# ---------------- IMPORTS ----------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score
)
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---------------- SETTINGS ----------------
MODEL_DIR = "./model/efficientnetb0"     # folder with saved models
DATA_DIR = "./data_fold/test"            # or test/val data
PLOT_BASE = "./plots_testing/efficientnetb0"
LOG_DIR = "./logs/efficientnetb0"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FOLDS = 5

os.makedirs(PLOT_BASE, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------- TRANSFORMS ----------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- DATASET ----------------
print("📂 Loading dataset...")
dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
num_classes = len(dataset.classes)
print(f"✅ Dataset loaded with {len(dataset)} images, {num_classes} classes: {dataset.classes}")

# ---------------- HELPER FUNCTIONS ----------------
def plot_and_save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def load_model(fold):
    """Load best EfficientNetB0 model for given fold"""
    print(f"🔹 Loading model for fold {fold}...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, f"best_model_fold{fold}.pt"), map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model for fold {fold} loaded.")
    return model

def eval_fold(model, val_idx, fold, save_dir):
    """Evaluate model on validation/test subset and plot metrics"""
    print(f"📊 Evaluating Fold {fold}...")

    val_subset = Subset(dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    pred_classes = np.argmax(all_preds, axis=1)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(all_labels, pred_classes)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=dataset.classes, yticklabels=dataset.classes, ax=ax)
    ax.set_title(f"Fold {fold} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plot_and_save(fig, os.path.join(save_dir, "confusion_matrix.png"))

    # ---------------- Classification Report ----------------
    report = classification_report(all_labels, pred_classes, target_names=dataset.classes, output_dict=True)
    df_report = pd.DataFrame(report).iloc[:-1, :-1].T
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_report, annot=True, cmap="Blues", ax=ax)
    ax.set_title(f"Fold {fold} - Classification Report")
    plot_and_save(fig, os.path.join(save_dir, "classification_report.png"))

    # ---------------- Multi-class ROC ----------------
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((all_labels==i).astype(int), all_preds[:, i])
        ax.plot(fpr, tpr, label=f"{dataset.classes[i]} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], "k--")
    ax.set_title(f"Fold {fold} - ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plot_and_save(fig, os.path.join(save_dir, "roc_curves.png"))

    # ---------------- Multi-class Precision-Recall ----------------
    fig, ax = plt.subplots(figsize=(8,6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve((all_labels==i).astype(int), all_preds[:, i])
        ax.plot(recall, precision, label=f"{dataset.classes[i]}")
    ax.set_title(f"Fold {fold} - Precision-Recall Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plot_and_save(fig, os.path.join(save_dir, "pr_curves.png"))

    # ---------------- Metrics ----------------
    acc = accuracy_score(all_labels, pred_classes)
    prec = precision_score(all_labels, pred_classes, average="macro")
    rec = recall_score(all_labels, pred_classes, average="macro")
    f1 = f1_score(all_labels, pred_classes, average="macro")

    print(f"✅ Fold {fold} Results: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    return acc, prec, rec, f1

# ---------------- MAIN ----------------
print("🚀 Starting cross-fold evaluation...")
targets = np.array([label for _, label in dataset.imgs])
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

for fold, (_, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
    fold_dir = os.path.join(PLOT_BASE, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    model = load_model(fold)
    acc, prec, rec, f1 = eval_fold(model, val_idx, fold, fold_dir)

    fold_metrics["accuracy"].append(acc)
    fold_metrics["precision"].append(prec)
    fold_metrics["recall"].append(rec)
    fold_metrics["f1"].append(f1)

# ---------------- SUMMARY ----------------
mean_acc, std_acc = np.mean(fold_metrics["accuracy"]), np.std(fold_metrics["accuracy"])
mean_prec, std_prec = np.mean(fold_metrics["precision"]), np.std(fold_metrics["precision"])
mean_rec, std_rec = np.mean(fold_metrics["recall"]), np.std(fold_metrics["recall"])
mean_f1, std_f1 = np.mean(fold_metrics["f1"]), np.std(fold_metrics["f1"])

summary = f"""
📊 Cross-Fold Testing Summary (EfficientNetB0)
---------------------------------------------
Accuracies per fold: {fold_metrics["accuracy"]}
Precisions per fold: {fold_metrics["precision"]}
Recalls per fold:    {fold_metrics["recall"]}
F1s per fold:        {fold_metrics["f1"]}

Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}
Mean Precision: {mean_prec:.4f} ± {std_prec:.4f}
Mean Recall   : {mean_rec:.4f} ± {std_rec:.4f}
Mean F1       : {mean_f1:.4f} ± {std_f1:.4f}
"""

print(summary)

with open(os.path.join(LOG_DIR, "test_summary.txt"), "w") as f:
    f.write(summary)

print(f"✅ All plots saved in {PLOT_BASE}")
print(f"📝 Summary log saved in {LOG_DIR}/test_summary.txt")
