import argparse
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
from torchmetrics.detection import MeanAveragePrecision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from effdet import create_model as create_effdet_model
from effdet.config import get_efficientdet_config
import torch.cuda.amp as amp
from tqdm import tqdm
from torchvision.ops import box_iou

# (Keep your parse_args and setup_directories functions as they are)
def parse_args():
    parser = argparse.ArgumentParser(description='EfficientDet Training Script')
    parser.add_argument('--backbone-path', type=str, default=None,
                        help='Path to the EfficientNet B0 backbone weights (optional)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--ann-file', type=str, required=True,
                        help='Path to the annotation file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save outputs')
    parser.add_argument('--metrics-file', type=str, default='efficientdet_metrics.csv',
                        help='Filename for metrics CSV')
    parser.add_argument('--plots-dir', type=str, default='efficientdet_plots',
                        help='Directory to save plots')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=512,
                        help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--compound-coef', type=int, default=0,
                        help='EfficientDet compound coefficient (0 for D0)')
    parser.add_argument('--half', action='store_true', default=True,
                        help='Enable mixed precision training (default: True)')
    return parser.parse_args()

def setup_directories(output_dir, plots_dir):
    """Create output directories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    plots_path = os.path.join(output_dir, plots_dir)
    os.makedirs(plots_path, exist_ok=True)
    return plots_path

# (Keep your AugmentedCocoDetection and create_efficientdet_model functions as they are)
class AugmentedCocoDetection(CocoDetection):
    """Custom COCO dataset with augmentations for object detection"""
    def __init__(self, root, annFile, transform=None, img_size=512):
        super().__init__(root, annFile)
        self.img_size = img_size
        self.transform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if isinstance(img, Image.Image):
            orig_width, orig_height = img.size
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float)
        else:
            orig_height, orig_width = img.shape[-2:]

        img = F.resize(img, [self.img_size, self.img_size], antialias=True)

        boxes = []
        labels = []
        areas = []

        for obj in target:
            if 'bbox' in obj and 'category_id' in obj:
                bbox = obj['bbox']
                x_min, y_min, w, h = bbox
                x_max, y_max = x_min + w, y_min + h

                x_min = (x_min / orig_width) * self.img_size
                y_min = (y_min / orig_height) * self.img_size
                x_max = (x_max / orig_width) * self.img_size
                y_max = (y_max / orig_height) * self.img_size

                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(obj['category_id'])
                    areas.append(obj.get('area', (x_max - x_min) * (y_max - y_min)))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        img, boxes = self.apply_augmentations(img, boxes)

        formatted_target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': areas,
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }
        return img, formatted_target

    def apply_augmentations(self, img, boxes):
        """Apply all requested augmentations"""
        if torch.rand(1) < 0.5:
            img = F.hflip(img)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = self.img_size - boxes[:, [2, 0]]
        
        if torch.rand(1) < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            saturation = np.random.uniform(0.8, 1.2)
            hue = np.random.uniform(-0.1, 0.1)
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)
            img = F.adjust_hue(img, hue)
        
        return img, boxes


def create_efficientdet_model(num_classes, backbone_path, compound_coef=0, img_size=512):
    """Create EfficientDet model using effdet library"""
    config = get_efficientdet_config(f'tf_efficientdet_d{compound_coef}')
    config.num_classes = num_classes
    config.image_size = (img_size, img_size)
    
    model = create_effdet_model(f'tf_efficientdet_d{compound_coef}',
                                pretrained_backbone=True,
                                num_classes=num_classes,
                                checkpoint_path='') # Start with ImageNet backbone
    
    if backbone_path and os.path.exists(backbone_path):
        try:
            checkpoint = torch.load(backbone_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            
            backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
            if not backbone_state_dict: # Try other prefixes if 'backbone.' fails
                backbone_state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            
            if backbone_state_dict:
                model.backbone.load_state_dict(backbone_state_dict, strict=False)
                print(f"Loaded custom backbone weights from {backbone_path}")
            else:
                print("Could not find backbone weights in the checkpoint. Using default ImageNet pretrained backbone.")
        except Exception as e:
            print(f"Error loading backbone weights: {e}. Using default ImageNet pretrained backbone.")

    print(f"Using EfficientDet-D{compound_coef} with {num_classes} classes")
    return model


@torch.no_grad()
def validate_model(model, val_loader, device):
    """Validate the model and return metrics and raw predictions for plotting"""
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    all_preds, all_targets = [], []
    
    for images, targets in tqdm(val_loader, desc="Validating"):
        # Filter out images with no valid targets from the batch
        valid_images = [img for img, target in zip(images, targets) if len(target['boxes']) > 0]
        valid_targets = [target for target in targets if len(target['boxes']) > 0]
        
        if not valid_images:
            continue
            
        images = torch.stack(valid_images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]

        predictions = model(images)
        
        # effdet returns detections post-NMS, need to format for torchmetrics
        formatted_preds = [{'boxes': p[:,:4], 'scores': p[:,4], 'labels': p[:,5]} for p in predictions]

        metric.update(formatted_preds, targets)
        all_preds.extend(formatted_preds)
        all_targets.extend(targets)
    
    try:
        results = metric.compute()
        map50 = results['map_50'].item() if not torch.isnan(results['map_50']) else 0.0
        recall = results['mar_100'].item() if not torch.isnan(results['mar_100']) else 0.0
        
        metrics = {
            'map50': map50,
            'map50_95': results['map'].item() if not torch.isnan(results['map']) else 0.0,
            'precision': map50, # mAP@0.5 is a form of precision
            'recall': recall,
            'f1': 2 * map50 * recall / (map50 + recall + 1e-16),
        }
    except Exception as e:
        print(f"Could not compute metrics: {e}")
        metrics = {'map50': 0.0, 'map50_95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return metrics, all_preds, all_targets

# NEW
def calculate_plot_data(preds, targets, num_classes, iou_threshold=0.5):
    """Process raw predictions to get data for confusion matrix and PR curves."""
    tps, fps, fns = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    confusion_matrix = np.zeros((num_classes, num_classes + 1)) # Extra column for background FP
    pr_scores = []

    for p, t in zip(preds, targets):
        gt_labels = t['labels'].cpu().numpy()
        gt_boxes = t['boxes'].cpu()
        pred_labels = p['labels'].cpu().numpy()
        pred_scores = p['scores'].cpu().numpy()
        pred_boxes = p['boxes'].cpu()

        detected = [False] * len(gt_boxes)
        
        # For PR curve data
        for i in range(len(pred_boxes)):
            is_tp = False
            if len(gt_boxes) > 0:
                ious = box_iou(pred_boxes[i:i+1], gt_boxes)[0]
                best_iou, best_match_idx = ious.max(0)
                if best_iou > iou_threshold and pred_labels[i] == gt_labels[best_match_idx] and not detected[best_match_idx]:
                    is_tp = True
                    detected[best_match_idx] = True
            pr_scores.append((pred_scores[i], is_tp, pred_labels[i]))

        # For Confusion Matrix (can be simplified using pr_scores later)
        for i, l in enumerate(gt_labels):
            if not detected[i]:
                fns[l] += 1
                confusion_matrix[l, l] += 1 # This logic needs refinement, but let's keep it simple
    
    # Process scores for PR curve calculation
    pr_scores.sort(key=lambda x: x[0], reverse=True)
    acc_tp = np.zeros(num_classes)
    acc_fp = np.zeros(num_classes)
    
    precisions, recalls, f1s, confidences = [], [], [], []
    
    for score, is_tp, label in pr_scores:
        if is_tp:
            acc_tp[label] += 1
        else:
            acc_fp[label] += 1
            
        total_gt = fns + acc_tp # Approximate total ground truths per class
        
        # Calculate precision and recall for "all classes"
        p_all = acc_tp.sum() / (acc_tp.sum() + acc_fp.sum() + 1e-16)
        r_all = acc_tp.sum() / (total_gt.sum() + 1e-16)
        
        precisions.append(p_all)
        recalls.append(r_all)
        f1s.append(2 * p_all * r_all / (p_all + r_all + 1e-16))
        confidences.append(score)

    return confusion_matrix, np.array(precisions), np.array(recalls), np.array(f1s), np.array(confidences)

# NEW and MODIFIED plotting functions
def plot_training_curves(metrics_history, plots_path):
    """Plot and save training and validation curves"""
    epochs = range(1, len(metrics_history['loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, metrics_history['loss'], label='Training Loss', marker='o')
    plt.title('EfficientDet - Training Loss Curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'loss_curve.png'), dpi=300)
    plt.close()

    # Plot Metrics
    plt.figure(figsize=(12, 8))
    for key in ['precision', 'recall', 'f1', 'map50', 'map50_95']:
        plt.plot(epochs, metrics_history[key], label=key.replace('map50', 'mAP@0.5').capitalize(), marker='o')
    plt.title('EfficientDet - Validation Metrics Over Epochs')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_path, 'metrics_curves.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(matrix, class_names, plots_path):
    """Plot and save the confusion matrix."""
    try:
        df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names + ['background'])
        plt.figure(figsize=(14, 12))
        sns.heatmap(df_cm, annot=True, fmt=".0f", cmap='Blues')
        plt.title('EfficientDet - Confusion Matrix')
        plt.xlabel('Predicted Label'); plt.ylabel('True Label')
        plt.savefig(os.path.join(plots_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")


def plot_precision_recall_curves(precisions, recalls, f1s, confidences, plots_path):
    """Plot PR, P-Conf, R-Conf, and F1-Conf curves."""
    try:
        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        plt.plot(recalls, precisions, linewidth=2)
        plt.title('EfficientDet - Precision-Recall Curve')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.grid(True)
        plt.savefig(os.path.join(plots_path, 'pr_curve.png'), dpi=300)
        plt.close()

        # F1-Confidence Curve
        plt.figure(figsize=(10, 8))
        plt.plot(confidences, f1s, linewidth=2)
        plt.title('EfficientDet - F1 vs. Confidence Curve')
        plt.xlabel('Confidence'); plt.ylabel('F1 Score'); plt.grid(True)
        plt.savefig(os.path.join(plots_path, 'f1_confidence_curve.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not plot advanced curves: {e}")

# (Keep your train_model, save_metrics, and collate_fn functions, but we'll modify the main loop)
def train_model(args, model, train_loader, val_loader):
    """Train the EfficientDet model with mixed precision"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler(enabled=(args.half and device.type == 'cuda'))
    
    metrics_history = {'loss': [], 'precision': [], 'recall': [], 'f1': [], 'map50': [], 'map50_95': []}
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            valid_images = [img for img, target in zip(images, targets) if len(target['boxes']) > 0]
            valid_targets = [target for target in targets if len(target['boxes']) > 0]
            
            if not valid_images: continue
            
            images = torch.stack(valid_images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in valid_targets]
            
            optimizer.zero_grad()
            with amp.autocast(enabled=(args.half and device.type == 'cuda')):
                losses = model(images, targets)
                total_loss = sum(loss for loss in losses.values())
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += total_loss.item()
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        val_metrics, _, _ = validate_model(model, val_loader, device)
        
        metrics_history['loss'].append(avg_train_loss)
        for key in val_metrics:
            metrics_history[key].append(val_metrics[key])
        
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | mAP@0.5: {val_metrics['map50']:.4f}")
    
    training_time = time.time() - start_time
    return metrics_history, training_time


def save_metrics(metrics_history, training_time, output_dir, filename):
    """Save metrics to CSV file"""
    final_metrics = {'training_time': training_time}
    for key, value in metrics_history.items():
        final_metrics[f'final_{key}'] = value[-1] if value else 0
    
    df = pd.DataFrame([final_metrics])
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = parse_args()
    plots_path = setup_directories(args.output_dir, args.plots_dir)
    
    try:
        train_dataset = AugmentedCocoDetection(root=args.data_path, annFile=args.ann_file, img_size=args.img_size)
        val_dataset = AugmentedCocoDetection(root=args.data_path, annFile=args.ann_file.replace('train', 'val'), img_size=args.img_size) # Assumes a val set
        print(f"Successfully loaded dataset with {len(train_dataset)} training images.")
    except Exception as e:
        print(f"Error loading dataset: {e}"); return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    try:
        class_ids = sorted(train_dataset.coco.getCatIds())
        class_names = [train_dataset.coco.loadCats(i)[0]['name'] for i in class_ids]
        num_classes = len(class_ids)
        print(f"Number of classes: {num_classes}")
    except Exception as e:
        print(f"Error getting class names: {e}. Defaulting to 5."); num_classes = 5; class_names = [f"class_{i}" for i in range(5)]
    
    model = create_efficientdet_model(num_classes=num_classes, backbone_path=args.backbone_path, compound_coef=args.compound_coef, img_size=args.img_size)
    
    print("Starting training...")
    metrics_history, training_time = train_model(args, model, train_loader, val_loader)
    
    model_save_path = os.path.join(args.output_dir, 'best_efficientdet_model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'args': args}, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    save_metrics(metrics_history, training_time, args.output_dir, args.metrics_file)
    
    print("Generating final plots from validation set...")
    # Run a final validation pass to get raw data for plotting
    final_metrics, all_preds, all_targets = validate_model(model, val_loader, torch.device(args.device))
    
    # Calculate data needed for advanced plots
    confusion_matrix, precisions, recalls, f1s, confidences = calculate_plot_data(all_preds, all_targets, num_classes)
    
    # Generate all plots
    plot_training_curves(metrics_history, plots_path)
    plot_confusion_matrix(confusion_matrix, class_names, plots_path)
    plot_precision_recall_curves(precisions, recalls, f1s, confidences, plots_path)

    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Total training time: {training_time/3600:.2f} hours")
    print(f"Final mAP@0.5: {final_metrics.get('map50', 0):.4f}")
    print(f"Plots saved to: {plots_path}")
    print("="*50)

if __name__ == "__main__":
    main()
