# kd_train_full_with_kd_plots.py
import matplotlib
matplotlib.use('Agg')
import os
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve, auc)
from itertools import cycle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model.resnet18 import create_model as create_resnet
from model.mobilenetv2 import create_model as create_mobilenetv2

# ----------------------------- REPRODUCIBILITY -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- ARGPARSE -----------------------------
parser = argparse.ArgumentParser(description="Knowledge Distillation Training (ResNet18 -> EfficientNetB0)")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for KD training")
parser.add_argument("--kfolds", type=int, required=True, help="Number of folds for cross-validation")
parser.add_argument("--alpha", type=float, default=0.3, help="Weight for CE loss vs KD loss (alpha*CE + (1-alpha)*KD)")
parser.add_argument("--temperature", type=float, default=4.0, help="Softmax temperature for KD")
args = parser.parse_args()

EPOCHS = args.epochs
K = args.kfolds
ALPHA = args.alpha
TEMP = args.temperature

# ----------------------------- FIXED SETTINGS -----------------------------
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

LOG_DIR = "./logs/kd_mobilenetv2"
MODEL_DIR = "./weights/kd_mobilenetv2"
PLOT_DIR = "./plots/kd_mobilenetv2"
DATA_DIR = "./data"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# ----------------------------- TRANSFORMS -----------------------------
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

# ----------------------------- KD LOSS COMPONENTS -----------------------------
ce_loss_fn = nn.CrossEntropyLoss()

def kd_components(student_logits, teacher_logits, labels, alpha=ALPHA, temp=TEMP, device=DEVICE):
    """
    Returns (combined_loss, ce_loss_value, kd_loss_value, kl_div_value)
    where kd_loss_value and kl_div_value use the standard scaling (T^2 * KL)
    """
    # CE (hard labels)
    ce = ce_loss_fn(student_logits, labels)

    # KD term: scaled KL between softened predictions
    # log_s = log_softmax(student / T), soft_t = softmax(teacher / T)
    log_s = F.log_softmax(student_logits / temp, dim=1)
    soft_t = F.softmax(teacher_logits / temp, dim=1)
    kd_term = F.kl_div(log_s, soft_t, reduction='batchmean') * (temp * temp)

    # Combined
    combined = alpha * ce + (1.0 - alpha) * kd_term

    # Also compute "raw" KL divergence at temperature=1 for monitoring if desired
    kl_raw = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1), reduction='batchmean')

    # Return torch scalars
    return combined, ce.detach().cpu().item(), kd_term.detach().cpu().item(), kl_raw.detach().cpu().item()

# ----------------------------- SAFE DATASET -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, ValueError):
                print(f"Skipping corrupted image: {self.samples[index][0]}")
                index = (index + 1) % len(self.samples)

# ----------------------------- PLOTTING -----------------------------
def plot_history(history, fold, classes, prefix="kd"):
    fold_plot_dir = os.path.join(PLOT_DIR, f"fold{fold}")
    os.makedirs(fold_plot_dir, exist_ok=True)

    # Loss and accuracy (loss is combined KD loss)
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

    # CE vs KD (components) (may have different lengths if early stop, use lists)
    plt.figure(figsize=(10,5))
    plt.plot(history.get("ce_loss", []), label="CE Loss (hard labels)")
    plt.plot(history.get("kd_loss", []), label="KD Loss (soft targets, scaled)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Fold {fold} Loss Components")
    plt.legend()
    plt.savefig(os.path.join(fold_plot_dir, f"loss_components_{prefix}_fold{fold}.png"))
    plt.close()

    # KL divergence curve
    plt.figure(figsize=(10,5))
    plt.plot(history.get("kl_div", []), label="KL Divergence (raw)")
    plt.xlabel("Epochs")
    plt.ylabel("KL Div (batchmean)")
    plt.title(f"Fold {fold} KL Divergence (teacher vs student)")
    plt.legend()
    plt.savefig(os.path.join(fold_plot_dir, f"kl_divergence_{prefix}_fold{fold}.png"))
    plt.close()

    # Precision, recall, f1
    for m in ["precision", "recall", "f1"]:
        plt.figure(figsize=(10,5))
        plt.plot(history[m], label=m)
        plt.xlabel("Epochs")
        plt.ylabel(m.capitalize())
        plt.title(f"Fold {fold} {m.capitalize()}")
        plt.legend()
        plt.savefig(os.path.join(fold_plot_dir, f"{m}_{prefix}_fold{fold}.png"))
        plt.close()

    # Confusion matrix
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
            plt.text(j, i, format(int(cm[i, j]), 'd'), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_plot_dir, f"confusion_matrix_{prefix}_fold{fold}.png"))
    plt.close()

    # ROC curves
    fpr, tpr, roc_auc = history["roc"]
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC (class {classes[i]}, AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"Fold {fold} ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fold_plot_dir, f"roc_curve_{prefix}_fold{fold}.png"))
    plt.close()

    # PR curves
    precision, recall, pr_auc = history["pr"]
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(classes)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR (class {classes[i]}, AUC={pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Fold {fold} PR Curves")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fold_plot_dir, f"pr_curve_{prefix}_fold{fold}.png"))
    plt.close()

# ----------------------------- PLOT OVERALL METRICS -----------------------------
def _pad_and_stack(list_of_lists):
    """Pad lists to the same length by repeating last value, then stack into array (n_folds, max_len)."""
    max_len = max(len(l) for l in list_of_lists)
    padded = []
    for l in list_of_lists:
        if len(l) < max_len:
            if len(l) == 0:
                padded.append([0.0]*max_len)
            else:
                padded.append(list(l) + [l[-1]] * (max_len - len(l)))
        else:
            padded.append(list(l))
    return np.vstack(padded)

def plot_overall_metrics(fold_metrics, classes, prefix, plot_dir):
    n_classes = len(classes)
    os.makedirs(plot_dir, exist_ok=True)

    # Overall confusion matrix (sum across folds)
    overall_cm = sum(hist["confusion_matrix"] for hist in fold_metrics)
    plt.figure(figsize=(10, 8))
    plt.imshow(overall_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Overall Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(int(overall_cm[i, j]), 'd'), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{prefix}_overall.png"))
    plt.close()

    # Mean ROC curves
    base_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros((n_classes, 100))
    mean_roc_auc = []
    for i in range(n_classes):
        tpr_list = []
        for hist in fold_metrics:
            fpr, tpr, roc_auc = hist["roc"]
            # Ensure arrays are valid
            try:
                interp_tpr = np.interp(base_fpr, fpr[i], tpr[i])
            except Exception:
                interp_tpr = np.zeros_like(base_fpr)
            tpr_list.append(interp_tpr)
        mean_tpr[i] = np.mean(tpr_list, axis=0)
        mean_roc_auc.append(auc(base_fpr, mean_tpr[i]))

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(base_fpr, mean_tpr[i], color=color, lw=2, label=f'Mean ROC (class {classes[i]}, AUC={mean_roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Mean ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, f"roc_curve_{prefix}_overall.png"))
    plt.close()

    # Mean PR curves
    base_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros((n_classes, 100))
    mean_pr_auc = []
    for i in range(n_classes):
        precision_list = []
        for hist in fold_metrics:
            precision, recall, pr_auc = hist["pr"]
            try:
                interp_precision = np.interp(base_recall, recall[i][::-1], precision[i][::-1])
            except Exception:
                interp_precision = np.zeros_like(base_recall)
            precision_list.append(interp_precision)
        mean_precision[i] = np.mean(precision_list, axis=0)
        mean_pr_auc.append(auc(base_recall, mean_precision[i]))

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(classes)), colors):
        plt.plot(base_recall, mean_precision[i], color=color, lw=2, label=f'Mean PR (class {classes[i]}, AUC={mean_pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("Mean PR Curves")
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(plot_dir, f"pr_curve_{prefix}_overall.png"))
    plt.close()

    # Mean CE vs KD loss across folds (align lengths)
    ce_lists = [hist.get("ce_loss", []) for hist in fold_metrics]
    kd_lists = [hist.get("kd_loss", []) for hist in fold_metrics]
    kl_lists = [hist.get("kl_div", []) for hist in fold_metrics]

    ce_arr = _pad_and_stack(ce_lists)  # shape (n_folds, epochs)
    kd_arr = _pad_and_stack(kd_lists)
    kl_arr = _pad_and_stack(kl_lists)

    mean_ce = np.mean(ce_arr, axis=0)
    mean_kd = np.mean(kd_arr, axis=0)
    mean_kl = np.mean(kl_arr, axis=0)
    epochs = np.arange(1, mean_ce.shape[0] + 1)

    plt.figure(figsize=(10,5))
    plt.plot(epochs, mean_ce, label="Mean CE Loss")
    plt.plot(epochs, mean_kd, label="Mean KD Loss (scaled)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean Loss Components Across Folds")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"loss_components_{prefix}_overall.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, mean_kl, label="Mean KL Divergence (raw)")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    plt.title("Mean KL Divergence Across Folds")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"kl_divergence_{prefix}_overall.png"))
    plt.close()

# ----------------------------- SAVE COMPLEX METRICS -----------------------------
def save_complex_metrics(history, fold, prefix, log_dir):
    np.save(os.path.join(log_dir, f"confusion_matrix_{prefix}_fold{fold}.npy"), history["confusion_matrix"])
    fpr, tpr, roc_auc = history["roc"]
    for i in range(len(roc_auc)):
        np.save(os.path.join(log_dir, f"roc_fpr_{prefix}_fold{fold}_class{i}.npy"), fpr[i])
        np.save(os.path.join(log_dir, f"roc_tpr_{prefix}_fold{fold}_class{i}.npy"), tpr[i])
        np.save(os.path.join(log_dir, f"roc_auc_{prefix}_fold{fold}_class{i}.npy"), np.array(roc_auc[i]))
    precision, recall, pr_auc = history["pr"]
    for i in range(len(pr_auc)):
        np.save(os.path.join(log_dir, f"pr_precision_{prefix}_fold{fold}_class{i}.npy"), precision[i])
        np.save(os.path.join(log_dir, f"pr_recall_{prefix}_fold{fold}_class{i}.npy"), recall[i])
        np.save(os.path.join(log_dir, f"pr_auc_{prefix}_fold{fold}_class{i}.npy"), np.array(pr_auc[i]))

    # Save CE/KD/KL arrays
    np.save(os.path.join(log_dir, f"ce_loss_{prefix}_fold{fold}.npy"), np.array(history.get("ce_loss", [])))
    np.save(os.path.join(log_dir, f"kd_loss_{prefix}_fold{fold}.npy"), np.array(history.get("kd_loss", [])))
    np.save(os.path.join(log_dir, f"kl_div_{prefix}_fold{fold}.npy"), np.array(history.get("kl_div", [])))

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    set_seed(42)
    dataset = SafeImageFolder(DATA_DIR, transform=train_transform)
    targets = np.array([label for _, label in dataset.samples])
    classes = dataset.classes
    n_classes = len(classes)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    fold_metrics = []

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

        # Load teacher (ResNet18)
        teacher = create_resnet(n_classes, DEVICE)
        teacher_path = f"./weights/resnet18/best_finetune_model_fold{fold}.pt"
        teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
        teacher.to(DEVICE)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # Create student 
        student = create_mobilenetv2(n_classes, DEVICE)
        student.to(DEVICE)

        optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")

        best_val_acc = 0.0
        patience, trigger_times = 3, 0

        history = {
            "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
            "precision": [], "recall": [], "f1": [], "roc": None, "pr": None, "confusion_matrix": None,
            "ce_loss": [], "kd_loss": [], "kl_div": []
        }

        for epoch in range(1, EPOCHS + 1):
            print(f"KD Training Epoch {epoch}/{EPOCHS}")
            student.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            epoch_ce_sum = 0.0
            epoch_kd_sum = 0.0
            epoch_kl_sum = 0.0
            epoch_samples = 0

            # ---------- TRAIN ----------
            for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    s_logits = student(imgs)
                    with torch.no_grad():
                        t_logits = teacher(imgs)
                    combined_loss, ce_val, kd_val, kl_val = kd_components(s_logits, t_logits, labels, alpha=ALPHA, temp=TEMP)
                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size = imgs.size(0)
                running_loss += combined_loss.item() * batch_size
                epoch_ce_sum += ce_val * batch_size
                epoch_kd_sum += kd_val * batch_size
                epoch_kl_sum += kl_val * batch_size
                epoch_samples += batch_size

                preds = torch.argmax(s_logits, dim=1).cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Average CE/KD/KL for epoch
            if epoch_samples > 0:
                history["ce_loss"].append(epoch_ce_sum / epoch_samples)
                history["kd_loss"].append(epoch_kd_sum / epoch_samples)
                history["kl_div"].append(epoch_kl_sum / epoch_samples)
            else:
                history["ce_loss"].append(0.0)
                history["kd_loss"].append(0.0)
                history["kl_div"].append(0.0)

            # ---------- VALIDATE ----------
            student.eval()
            val_loss = 0.0
            all_preds_proba = []
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    s_logits = student(imgs)
                    t_logits = teacher(imgs)
                    combined_loss, ce_val, kd_val, kl_val = kd_components(s_logits, t_logits, labels, alpha=ALPHA, temp=TEMP)
                    batch_size = imgs.size(0)
                    val_loss += combined_loss.item() * batch_size

                    probs = F.softmax(s_logits, dim=1).cpu().numpy()
                    all_preds_proba.append(probs)
                    all_preds.extend(np.argmax(probs, axis=1))
                    all_labels.extend(labels.cpu().numpy())

            if len(all_preds_proba) == 0:
                print("Warning: validation set empty for this fold/epoch")
                continue

            all_probs = np.vstack(all_preds_proba)
            all_labels_arr = np.array(all_labels)

            history["val_loss"].append(val_loss / len(val_loader.dataset))
            acc = accuracy_score(all_labels_arr, all_preds)
            prec = precision_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
            rec = recall_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
            cm = confusion_matrix(all_labels_arr, all_preds)

            # ROC / PR
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, pr_auc = {}, {}, {}
            for i in range(n_classes):
                try:
                    fpr[i], tpr[i], _ = roc_curve(all_labels_arr == i, all_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                except Exception:
                    fpr[i], tpr[i], roc_auc[i] = np.array([0.0]), np.array([0.0]), 0.0
                try:
                    precision[i], recall[i], _ = precision_recall_curve(all_labels_arr == i, all_probs[:, i])
                    pr_auc[i] = auc(recall[i], precision[i])
                except Exception:
                    precision[i], recall[i], pr_auc[i] = np.array([0.0]), np.array([0.0]), 0.0

            history["val_accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1"].append(f1)
            history["confusion_matrix"] = cm
            history["roc"] = (fpr, tpr, roc_auc)
            history["pr"] = (precision, recall, pr_auc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            # ---------- CHECKPOINT / EARLY STOP ----------
            if acc > best_val_acc:
                best_val_acc = acc
                trigger_times = 0
                save_path = os.path.join(MODEL_DIR, f"best_kd_model_fold{fold}.pt")
                torch.save(student.state_dict(), save_path)
                print(f"Saved KD student model: {save_path}")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered")
                    break

        # Save fold logs and plots
        with open(os.path.join(LOG_DIR, f"kd_history_fold{fold}.json"), "w") as f:
            json.dump({k: v for k, v in history.items() if k not in ["roc", "pr", "confusion_matrix"]}, f, indent=4)
        save_complex_metrics(history, fold, "kd", LOG_DIR)
        plot_history(history, fold, classes, prefix="kd")
        fold_metrics.append(history)
        print(f"Saved history and plots for fold {fold}")

    # ----------------------------- FINAL SUMMARY & OVERALL PLOTS -----------------------------
    def summarize(metric_name, fold_metrics_local):
        values = []
        for hist in fold_metrics_local:
            arr = hist.get(metric_name)
            if arr is None or len(arr) == 0:
                values.append(0.0)
            else:
                values.append(arr[-1])
        return np.mean(values), np.std(values), values

    mean_acc, std_acc, acc_list = summarize("val_accuracy", fold_metrics)
    mean_prec, std_prec, prec_list = summarize("precision", fold_metrics)
    mean_rec, std_rec, rec_list = summarize("recall", fold_metrics)
    mean_f1, std_f1, f1_list = summarize("f1", fold_metrics)

    # Per-class mean ROC & PR AUC across folds
    mean_roc = []
    mean_pr = []
    for i in range(n_classes):
        roc_aucs = []
        pr_aucs = []
        for hist in fold_metrics:
            try:
                roc_aucs.append(hist["roc"][2][i])
            except Exception:
                roc_aucs.append(0.0)
            try:
                pr_aucs.append(hist["pr"][2][i])
            except Exception:
                pr_aucs.append(0.0)
        mean_roc.append(np.mean(roc_aucs))
        mean_pr.append(np.mean(pr_aucs))

    # Save & plot overall metrics
    plot_overall_metrics(fold_metrics, classes, "kd", PLOT_DIR)

    summary_text = "Final KD Summary:\n\n"
    summary_text += "Per-Fold Validation Metrics (student):\n"
    summary_text += f"{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
    summary_text += "-" * 56 + "\n"
    for i in range(len(fold_metrics)):
        a = acc_list[i] * 100
        p = prec_list[i] * 100
        r = rec_list[i] * 100
        f = f1_list[i] * 100
        summary_text += f"{i+1:<6} {a:6.2f}%     {p:6.2f}%     {r:6.2f}%     {f:6.2f}%\n"

    summary_text += (
        f"\nKD Student Validation Metrics across {K} folds:\n"
        f"Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}\n"
        f"Precision: {mean_prec*100:.2f}% ± {std_prec*100:.2f}\n"
        f"Recall: {mean_rec*100:.2f}% ± {std_rec*100:.2f}\n"
        f"F1-score: {mean_f1*100:.2f}% ± {std_f1*100:.2f}\n"
    )

    for i, cls in enumerate(classes):
        summary_text += f"\nClass {cls}:\n  Mean ROC AUC: {mean_roc[i]*100:.2f}\n  Mean PR AUC: {mean_pr[i]*100:.2f}\n"

    print("\nFinal Summary:")
    print(summary_text)
    with open(os.path.join(LOG_DIR, "final_summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"Saved final_summary.txt in {LOG_DIR}")
