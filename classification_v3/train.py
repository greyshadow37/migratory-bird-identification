import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from itertools import cycle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from model.resnet18 import create_model, freeze_backbone, unfreeze_finetune_layers

# ----------------------------- REPRODUCIBILITY -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- ARGPARSE -----------------------------
parser = argparse.ArgumentParser(description="Train ResNet18 for migratory bird classification")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for head-only training")
parser.add_argument("--kfolds", type=int, required=True, help="Number of folds for cross-validation")
parser.add_argument("--fine_tune_epochs", type=int, default=5, help="Number of fine-tuning epochs")
args = parser.parse_args()

EPOCHS = args.epochs
K = args.kfolds
FINE_TUNE_EPOCHS = args.fine_tune_epochs

# ----------------------------- FIXED SETTINGS -----------------------------
BATCH_SIZE = 32
LR = 1e-4
FINE_TUNE_LR = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

LOG_DIR = "./logs/resnet18"
MODEL_DIR = "./weights/resnet18"
PLOT_DIR = "./plots/resnet18"
DATA_DIR = "./data"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# ----------------------------- TRANSFORMS (ImageNet norm) -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ----------------------------- PLOTTING -----------------------------
def plot_history(history, fold, classes, prefix=""):
    fold_plot_dir = os.path.join(PLOT_DIR, f"fold{fold}")
    os.makedirs(fold_plot_dir, exist_ok=True)

    # Plot loss and accuracy
    pairs = [("train_loss", "val_loss", "Loss"), ("train_accuracy", "val_accuracy", "Accuracy")]
    for m1, m2, label in pairs:
        plt.figure(figsize=(10,5))
        plt.plot(history[m1], label=m1)
        plt.plot(history[m2], label=m2)
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.title(f"Fold {fold} {label}")
        plt.legend()
        plt.savefig(os.path.join(fold_plot_dir, f"{label.lower()}_{prefix}_fold{fold}.png"))
        plt.close()
        print(f"Saved {label.lower()}_{prefix}_fold{fold}.png")

    # Plot precision, recall, f1
    for m in ["precision", "recall", "f1"]:
        plt.figure(figsize=(10,5))
        plt.plot(history[m], label=m)
        plt.xlabel("Epochs")
        plt.ylabel(m.capitalize())
        plt.title(f"Fold {fold} {m.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join(fold_plot_dir, f"{m}_{prefix}_fold{fold}.png"))
        plt.close()
        print(f"Saved {m}_{prefix}_fold{fold}.png")

    # Plot confusion matrix
    cm = history["confusion_matrix"]
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_plot_dir, f"confusion_matrix_{prefix}_fold{fold}.png"))
    plt.close()
    print(f"Saved confusion_matrix_{prefix}_fold{fold}.png")

    # Plot ROC curves (one-vs-rest)
    fpr, tpr, roc_auc = history["roc"]
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {classes[i]}, AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Fold {fold} ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fold_plot_dir, f"roc_curve_{prefix}_fold{fold}.png"))
    plt.close()
    print(f"Saved roc_curve_{prefix}_fold{fold}.png")

    # Plot PR curves
    precision, recall, pr_auc = history["pr"]
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(classes)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR curve (class {classes[i]}, AUC = {pr_auc[i]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Fold {fold} Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fold_plot_dir, f"pr_curve_{prefix}_fold{fold}.png"))
    plt.close()
    print(f"Saved pr_curve_{prefix}_fold{fold}.png")

# ----------------------------- PLOT OVERALL METRICS -----------------------------
def plot_overall_metrics(fold_metrics, classes, prefix, plot_dir):
    n_classes = len(classes)

    # Overall confusion matrix (sum across folds)
    overall_cm = sum(hist["confusion_matrix"] for hist in fold_metrics)
    plt.figure(figsize=(10, 8))
    plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Overall Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(overall_cm[i, j], 'd'), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{prefix}_overall.png"))
    plt.close()
    print(f"Saved confusion_matrix_{prefix}_overall.png")

    # Mean ROC curves
    base_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros((n_classes, 100))
    mean_roc_auc = []
    for i in range(n_classes):
        tpr_list = []
        for hist in fold_metrics:
            fpr, tpr, roc_auc = hist["roc"]
            interp_tpr = np.interp(base_fpr, fpr[i], tpr[i])
            tpr_list.append(interp_tpr)
        mean_tpr[i] = np.mean(tpr_list, axis=0)
        mean_roc_auc.append(auc(base_fpr, mean_tpr[i]))

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(base_fpr, mean_tpr[i], color=color, lw=2, label=f'Mean ROC (class {classes[i]}, AUC = {mean_roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Mean ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f"roc_curve_{prefix}_overall.png"))
    plt.close()
    print(f"Saved roc_curve_{prefix}_overall.png")

    # Mean PR curves
    base_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros((n_classes, 100))
    mean_pr_auc = []
    for i in range(n_classes):
        precision_list = []
        for hist in fold_metrics:
            precision, recall, pr_auc = hist["pr"]
            interp_precision = np.interp(base_recall, recall[i][::-1], precision[i][::-1])
            precision_list.append(interp_precision)
        mean_precision[i] = np.mean(precision_list, axis=0)
        mean_pr_auc.append(auc(base_recall, mean_precision[i]))

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(classes)), colors):
        plt.plot(base_recall, mean_precision[i], color=color, lw=2, label=f'Mean PR (class {classes[i]}, AUC = {mean_pr_auc[i]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Mean Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(plot_dir, f"pr_curve_{prefix}_overall.png"))
    plt.close()
    print(f"Saved pr_curve_{prefix}_overall.png")

# ----------------------------- SAVE COMPLEX METRICS -----------------------------
def save_complex_metrics(history, fold, prefix, log_dir):
    np.save(os.path.join(log_dir, f"confusion_matrix_{prefix}_fold{fold}.npy"), history["confusion_matrix"])
    print(f"Saved confusion_matrix_{prefix}_fold{fold}.npy")
    
    fpr, tpr, roc_auc = history["roc"]
    for i in range(len(roc_auc)):
        np.save(os.path.join(log_dir, f"roc_fpr_{prefix}_fold{fold}_class{i}.npy"), fpr[i])
        np.save(os.path.join(log_dir, f"roc_tpr_{prefix}_fold{fold}_class{i}.npy"), tpr[i])
        np.save(os.path.join(log_dir, f"roc_auc_{prefix}_fold{fold}_class{i}.npy"), np.array(roc_auc[i]))
        print(f"Saved roc_data_fold{fold}_class{i}.npy")
    
    precision, recall, pr_auc = history["pr"]
    for i in range(len(pr_auc)):
        np.save(os.path.join(log_dir, f"pr_precision_{prefix}_fold{fold}_class{i}.npy"), precision[i])
        np.save(os.path.join(log_dir, f"pr_recall_{prefix}_fold{fold}_class{i}.npy"), recall[i])
        np.save(os.path.join(log_dir, f"pr_auc_{prefix}_fold{fold}_class{i}.npy"), np.array(pr_auc[i]))
        print(f"Saved pr_data_fold{fold}_class{i}.npy")

# ----------------------------- SAFE DATASET -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, ValueError):
                print(f"Skipping corrupted image: {self.samples[index][0]}")
                index = (index + 1) % len(self.samples)

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    set_seed(42)
    dataset = SafeImageFolder(DATA_DIR, transform=train_transform)
    targets = np.array([label for _, label in dataset.samples])
    classes = dataset.classes
    n_classes = len(classes)
    counts = np.bincount(targets, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = (targets.shape[0] / (n_classes * counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    fold_metrics_head = []
    fold_metrics_finetune = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        print(f"\nFold {fold}/{K}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        model = create_model(n_classes, DEVICE)
        freeze_backbone(model)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience = 3
        trigger_times = 0
        best_epoch = 0
        history_head = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
                        "precision": [], "recall": [], "f1": [], "roc": None, "pr": None, "confusion_matrix": None}

        for epoch in range(1, EPOCHS + 1):
            print(f"Head Training Epoch {epoch}/{EPOCHS}")
            model.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_train_preds, all_train_labels)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
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

            cm = confusion_matrix(all_labels, pred_classes)
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, pr_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_preds[:, i])
                pr_auc[i] = auc(recall[i], precision[i])

            history_head["train_loss"].append(train_loss)
            history_head["val_loss"].append(val_loss)
            history_head["train_accuracy"].append(train_acc)
            history_head["val_accuracy"].append(acc)
            history_head["precision"].append(prec)
            history_head["recall"].append(rec)
            history_head["f1"].append(f1)
            history_head["confusion_matrix"] = cm
            history_head["roc"] = (fpr, tpr, roc_auc)
            history_head["pr"] = (precision, recall, pr_auc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = acc
                best_epoch = epoch
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_head_model_fold{fold}.pt"))
                print(f"Saved best_head_model_fold{fold}.pt (Val Acc: {best_val_acc:.4f})")
            else:
                trigger_times += 1

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break

        with open(os.path.join(LOG_DIR, f"history_head_fold{fold}.json"), "w") as f:
            json.dump({k: v for k, v in history_head.items() if k not in ["roc", "pr", "confusion_matrix"]}, f, indent=4)
        print(f"Saved history_head_fold{fold}.json")
        save_complex_metrics(history_head, fold, "head", LOG_DIR)
        plot_history(history_head, fold, classes, prefix="head")

        print(f"\nFine-Tuning Fold {fold}/{K}")
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"best_head_model_fold{fold}.pt")))
        unfreeze_finetune_layers(model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = torch.amp.GradScaler("cuda")

        best_val_loss = float("inf")
        best_val_acc = 0.0
        trigger_times = 0
        best_epoch = 0
        history_finetune = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
                            "precision": [], "recall": [], "f1": [], "roc": None, "pr": None, "confusion_matrix": None}

        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            print(f"Fine-Tuning Epoch {epoch}/{FINE_TUNE_EPOCHS}")
            model.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_train_preds, all_train_labels)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
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

            cm = confusion_matrix(all_labels, pred_classes)
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, pr_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_preds[:, i])
                pr_auc[i] = auc(recall[i], precision[i])

            history_finetune["train_loss"].append(train_loss)
            history_finetune["val_loss"].append(val_loss)
            history_finetune["train_accuracy"].append(train_acc)
            history_finetune["val_accuracy"].append(acc)
            history_finetune["precision"].append(prec)
            history_finetune["recall"].append(rec)
            history_finetune["f1"].append(f1)
            history_finetune["confusion_matrix"] = cm
            history_finetune["roc"] = (fpr, tpr, roc_auc)
            history_finetune["pr"] = (precision, recall, pr_auc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = acc
                best_epoch = epoch
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"best_finetune_model_fold{fold}.pt"))
                print(f"Saved best_finetune_model_fold{fold}.pt (Val Acc: {best_val_acc:.4f})")
            else:
                trigger_times += 1

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break

        with open(os.path.join(LOG_DIR, f"history_finetune_fold{fold}.json"), "w") as f:
            json.dump({k: v for k, v in history_finetune.items() if k not in ["roc", "pr", "confusion_matrix"]}, f, indent=4)
        print(f"Saved history_finetune_fold{fold}.json")
        save_complex_metrics(history_finetune, fold, "finetune", LOG_DIR)
        plot_history(history_finetune, fold, classes, prefix="finetune")

        fold_metrics_head.append(history_head)
        fold_metrics_finetune.append(history_finetune)

    # ----------------------------- FINAL SUMMARY -----------------------------
    def summarize(metric_name, fold_metrics):
        values = [hist[metric_name][-1] for hist in fold_metrics]
        return np.mean(values), np.std(values), values

    mean_acc_head, std_acc_head, acc_head_list = summarize("val_accuracy", fold_metrics_head)
    mean_prec_head, std_prec_head, prec_head_list = summarize("precision", fold_metrics_head)
    mean_rec_head, std_rec_head, rec_head_list = summarize("recall", fold_metrics_head)
    mean_f1_head, std_f1_head, f1_head_list = summarize("f1", fold_metrics_head)
    mean_acc_finetune, std_acc_finetune, acc_finetune_list = summarize("val_accuracy", fold_metrics_finetune)
    mean_prec_finetune, std_prec_finetune, prec_finetune_list = summarize("precision", fold_metrics_finetune)
    mean_rec_finetune, std_rec_finetune, rec_finetune_list = summarize("recall", fold_metrics_finetune)
    mean_f1_finetune, std_f1_finetune, f1_finetune_list = summarize("f1", fold_metrics_finetune)

    mean_roc_auc_head, std_roc_auc_head = [], []
    mean_pr_auc_head, std_pr_auc_head = [], []
    mean_roc_auc_finetune, std_roc_auc_finetune = [], []
    mean_pr_auc_finetune, std_pr_auc_finetune = [], []
    for i in range(n_classes):
        roc_aucs_head = [hist["roc"][2][i] for hist in fold_metrics_head]
        pr_aucs_head = [hist["pr"][2][i] for hist in fold_metrics_head]
        roc_aucs_finetune = [hist["roc"][2][i] for hist in fold_metrics_finetune]
        pr_aucs_finetune = [hist["pr"][2][i] for hist in fold_metrics_finetune]
        mean_roc_auc_head.append(np.mean(roc_aucs_head))
        std_roc_auc_head.append(np.std(roc_aucs_head))
        mean_pr_auc_head.append(np.mean(pr_aucs_head))
        std_pr_auc_head.append(np.std(pr_aucs_head))
        mean_roc_auc_finetune.append(np.mean(roc_aucs_finetune))
        std_roc_auc_finetune.append(np.std(roc_aucs_finetune))
        mean_pr_auc_finetune.append(np.mean(pr_aucs_finetune))
        std_pr_auc_finetune.append(np.std(pr_aucs_finetune))

    overall_mean_roc_auc_head = np.mean(mean_roc_auc_head) * 100
    overall_mean_roc_auc_finetune = np.mean(mean_roc_auc_finetune) * 100
    overall_mean_pr_auc_head = np.mean(mean_pr_auc_head) * 100
    overall_mean_pr_auc_finetune = np.mean(mean_pr_auc_finetune) * 100

    plot_overall_metrics(fold_metrics_head, classes, "head", PLOT_DIR)
    plot_overall_metrics(fold_metrics_finetune, classes, "finetune", PLOT_DIR)

    summary_text = "Final Summary:\n\n"
    summary_text += "Per-Fold Validation Metrics (Head-Only):\n"
    summary_text += f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
    summary_text += "-" * 46 + "\n"
    for i in range(K):
        summary_text += f"{i+1:<6} {acc_head_list[i]*100:.2f}%     {prec_head_list[i]*100:.2f}%     {rec_head_list[i]*100:.2f}%     {f1_head_list[i]*100:.2f}%\n"

    summary_text += "\nPer-Fold Validation Metrics (Fine-Tuned):\n"
    summary_text += f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
    summary_text += "-" * 46 + "\n"
    for i in range(K):
        summary_text += f"{i+1:<6} {acc_finetune_list[i]*100:.2f}%     {prec_finetune_list[i]*100:.2f}%     {rec_finetune_list[i]*100:.2f}%     {f1_finetune_list[i]*100:.2f}%\n"

    summary_text += (
        f"\nHead-Only Validation Metrics across {K} folds:\n"
        f"Accuracy: {mean_acc_head*100:.2f}% ± {std_acc_head*100:.2f}\n"
        f"Precision: {mean_prec_head*100:.2f}% ± {std_prec_head*100:.2f}\n"
        f"Recall: {mean_rec_head*100:.2f}% ± {std_rec_head*100:.2f}\n"
        f"F1-score: {mean_f1_head*100:.2f}% ± {std_f1_head*100:.2f}\n"
        f"Mean ROC AUC: {overall_mean_roc_auc_head:.2f}\n"
        f"Mean PR AUC: {overall_mean_pr_auc_head:.2f}\n"
        f"\nFine-Tuned Validation Metrics across {K} folds:\n"
        f"Accuracy: {mean_acc_finetune*100:.2f}% ± {std_acc_finetune*100:.2f}\n"
        f"Precision: {mean_prec_finetune*100:.2f}% ± {std_prec_finetune*100:.2f}\n"
        f"Recall: {mean_rec_finetune*100:.2f}% ± {std_rec_finetune*100:.2f}\n"
        f"F1-score: {mean_f1_finetune*100:.2f}% ± {std_f1_finetune*100:.2f}\n"
        f"Mean ROC AUC: {overall_mean_roc_auc_finetune:.2f}\n"
        f"Mean PR AUC: {overall_mean_pr_auc_finetune:.2f}\n"
        f"\nComparison (Fine-Tuned vs. Head-Only):\n"
        f"Accuracy Improvement: {(mean_acc_finetune - mean_acc_head)*100:.2f}\n"
        f"Precision Improvement: {(mean_prec_finetune - mean_prec_head)*100:.2f}\n"
        f"Recall Improvement: {(mean_rec_finetune - mean_rec_head)*100:.2f}\n"
        f"F1-score Improvement: {(mean_f1_finetune - mean_f1_head)*100:.2f}\n"
        f"Mean ROC AUC Improvement: {(overall_mean_roc_auc_finetune - overall_mean_roc_auc_head):.2f}\n"
        f"Mean PR AUC Improvement: {(overall_mean_pr_auc_finetune - overall_mean_pr_auc_head):.2f}\n"
    )
    for i, cls in enumerate(classes):
        summary_text += (
            f"\nClass {cls}:\n"
            f"Head-Only ROC AUC: {mean_roc_auc_head[i]*100:.2f} ± {std_roc_auc_head[i]*100:.2f}\n"
            f"Fine-Tuned ROC AUC: {mean_roc_auc_finetune[i]*100:.2f} ± {std_roc_auc_finetune[i]*100:.2f}\n"
            f"Head-Only PR AUC: {mean_pr_auc_head[i]*100:.2f} ± {std_pr_auc_head[i]*100:.2f}\n"
            f"Fine-Tuned PR AUC: {mean_pr_auc_finetune[i]*100:.2f} ± {std_pr_auc_finetune[i]*100:.2f}\n"
        )

    print("\nFinal Summary:")
    print(summary_text)
    with open(os.path.join(LOG_DIR, "final_summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"Saved final_summary.txt")