import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
from PIL import Image

# ----------------------------- REPRODUCIBILITY -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- ARGPARSE -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--kfolds", type=int, default=5, help="Number of folds")
args = parser.parse_args()

EPOCHS = args.epochs
K = args.kfolds

# ----------------------------- SETTINGS -----------------------------
BATCH_SIZE = 32
LR = 1e-4
LOG_DIR = "./logs/mobilenetv2"
MODEL_DIR = "./model/mobilenetv2"
PLOT_DIR = "./plots/mobilenetv2"
DATA_DIR = "./data_fold/train"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- TRANSFORMS -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# ----------------------------- PLOTTING -----------------------------
def plot_history(history, fold):
    fold_plot_dir = os.path.join(PLOT_DIR, f"fold{fold}")
    os.makedirs(fold_plot_dir, exist_ok=True)

    pairs = [("train_loss", "val_loss", "Loss"), ("train_accuracy", "val_accuracy", "Accuracy")]
    for m1, m2, label in pairs:
        plt.figure(figsize=(10,5))
        plt.plot(history[m1], label=m1)
        plt.plot(history[m2], label=m2)
        plt.xlabel("Epochs"); plt.ylabel(label); plt.title(f"Fold {fold} {label}")
        plt.legend()
        plt.savefig(os.path.join(fold_plot_dir, f"{label.lower()}_fold{fold}.png"))
        plt.close()

    for m in ["precision", "recall", "f1"]:
        plt.figure(figsize=(10,5))
        plt.plot(history[m], label=m)
        plt.xlabel("Epochs"); plt.ylabel(m.capitalize()); plt.title(f"Fold {fold} {m.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join(fold_plot_dir, f"{m}_fold{fold}.png"))
        plt.close()

# ----------------------------- SAFE DATASET -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, ValueError):
                print(f"Skipping corrupted image: {self.imgs[index][0]}")
                index = (index + 1) % len(self.imgs)

# ----------------------------- MAIN SCRIPT -----------------------------
if __name__ == "__main__":
    set_seed(42)

    dataset = SafeImageFolder(DATA_DIR, transform=train_transform)
    targets = np.array([label for _, label in dataset.imgs])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        print(f"\n--- Fold {fold}/{K} ---")

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ----------------------------- MODEL -----------------------------
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features

# --- Add dropout to reduce overfitting ---
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),  # 30% dropout
    nn.Linear(num_features, len(dataset.classes))
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Add early stopping parameters ---
best_val_loss = float("inf")
best_val_acc = 0.0         # Track best validation accuracy
patience = 3               # Stop if no improvement in 'patience' epochs
trigger_times = 0          

history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
           "precision": [], "recall": [], "f1": []}

# ------------------ TRAIN/VAL ------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0
    all_train_preds, all_train_labels = [], []

    for imgs, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch} [Train]", leave=True):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_train_preds.extend(preds)
        all_train_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(all_train_labels, all_train_preds)

    # ------------------ VALIDATION ------------------
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Fold {fold} Epoch {epoch} [Val]", leave=True):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    pred_classes = np.argmax(all_preds, axis=1)

    acc = accuracy_score(all_labels, pred_classes)
    prec = precision_score(all_labels, pred_classes, average='weighted', zero_division=0)
    rec = recall_score(all_labels, pred_classes, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, pred_classes, average='weighted', zero_division=0)

    # log metrics
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(acc)
    history["precision"].append(prec)
    history["recall"].append(rec)
    history["f1"].append(f1)

    # ------------------ SAVE BEST MODEL ------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = acc           # Update best accuracy
        best_epoch = epoch           # Store which epoch gave best val
        trigger_times = 0            # Reset patience counter
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_model_fold{fold}.pt"))
    else:
        trigger_times += 1

    print(f"Epoch {epoch} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # --- Early stopping check ---
    if trigger_times >= patience:
        print(f"Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        break

    # ------------------ Save history + plots ------------------
    with open(os.path.join(LOG_DIR, f"history_fold{fold}.json"), "w") as f:
        json.dump(history, f, indent=4)

    plot_history(history, fold)
    fold_metrics.append(history)

    # ----------------------------- FINAL SUMMARY -----------------------------
    def summarize(metric_name):
        values = [hist[metric_name][-1] for hist in fold_metrics]
        return np.mean(values), np.std(values)

    mean_acc, std_acc = summarize("val_accuracy")
    mean_prec, std_prec = summarize("precision")
    mean_rec, std_rec = summarize("recall")
    mean_f1, std_f1 = summarize("f1")

    summary_text = (
        f"Final Validation Accuracy across {K} folds:\n"
        f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n"
        f"Precision: {mean_prec:.4f} ± {std_prec:.4f}\n"
        f"Recall: {mean_rec:.4f} ± {std_rec:.4f}\n"
        f"F1-score: {mean_f1:.4f} ± {std_f1:.4f}\n"
    )
    print(summary_text)

    with open(os.path.join(LOG_DIR, "final_summary.txt"), "w") as f:
        f.write(summary_text + "\n")
