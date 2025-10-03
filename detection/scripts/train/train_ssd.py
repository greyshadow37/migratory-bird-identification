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
import torch.amp as amp
from PIL import Image
from tqdm import tqdm
import timm


def parse_args():
    parser = argparse.ArgumentParser(description='SSD with MobileNetV2 Training Script')
    parser.add_argument('--backbone-path', type=str, default=None,
                       help='Path to the MobileNetV2 backbone weights (optional)')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the data directory')
    parser.add_argument('--ann-file', type=str, required=True,
                       help='Path to the annotation file')
    parser.add_argument('--val-ann-file', type=str, default=None,
                       help='Path to validation annotation file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save outputs')
    parser.add_argument('--metrics-file', type=str, default='ssd_training_metrics.csv',
                       help='Filename for metrics CSV')
    parser.add_argument('--plots-dir', type=str, default='ssd_training_plots',
                       help='Directory to save plots')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=300,
                       help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training (e.g., cuda, cpu)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--half', action='store_true',
                       help='Enable mixed precision training (AMP)')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='Number of classes (auto-detected from dataset if not provided)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to full SSD model checkpoint (.pt) to resume or fine-tune')
    return parser.parse_args()

def setup_directories(output_dir, plots_dir):
    os.makedirs(output_dir, exist_ok=True)
    plots_path = os.path.join(output_dir, plots_dir)
    os.makedirs(plots_path, exist_ok=True)
    return plots_path

class Detection(CocoDetection):
    def __init__(self, root, annFile, img_size=300):
        super().__init__(root, annFile)
        self.img_size = img_size

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if isinstance(img, Image.Image):
            orig_w, orig_h = img.size
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float32)
        else:
            orig_h, orig_w = img.shape[-2:]
        img = F.resize(img, [self.img_size, self.img_size], antialias=True)
        boxes = []
        labels = []
        for obj in target:
            if 'bbox' in obj and 'category_id' in obj:
                x, y, w, h = obj['bbox']
                x2 = x + w; y2 = y + h
                x_s = (x / orig_w) * self.img_size
                y_s = (y / orig_h) * self.img_size
                x2_s = (x2 / orig_w) * self.img_size
                y2_s = (y2 / orig_h) * self.img_size
                if x2_s > x_s and y2_s > y_s:
                    boxes.append([x_s, y_s, x2_s, y2_s])
                    labels.append(obj['category_id'])
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        return img, {'boxes': boxes, 'labels': labels}

class SSDWithMobileNetV2(nn.Module):
    def __init__(self, num_classes, backbone_path=None, img_size=300):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        if backbone_path and os.path.exists(backbone_path):
            try:
                ckpt = torch.load(backbone_path, map_location='cpu')
                if isinstance(ckpt, dict):
                    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
                else:
                    state = ckpt
                missing, unexpected = self.backbone.load_state_dict(state, strict=False)
                print(f"Loaded MobileNetV2 backbone weights from {backbone_path} | Missing: {len(missing)} Unexpected: {len(unexpected)}")
            except Exception as e:
                print(f"Warning: Could not load backbone: {e}")
        dummy = torch.randn(1,3,self.img_size,self.img_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
        chans = [f.shape[1] for f in feats]
        self.base_feats = len(feats)
        self.feature_channels = chans[-3:]
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.feature_channels[-1], 256, 1), nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, 1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True)
            )
        ])
        self.num_anchors = [4,6,6,6,4,4]
        all_ch = self.feature_channels + [512,256]
        self.loc_heads = nn.ModuleList([
            nn.Conv2d(c, a*4, 3, padding=1) for c,a in zip(all_ch, self.num_anchors)
        ])
        self.cls_heads = nn.ModuleList([
            nn.Conv2d(c, a*num_classes, 3, padding=1) for c,a in zip(all_ch, self.num_anchors)
        ])
        from torchvision.ops import boxes as box_ops
        self.box_ops = box_ops

    def forward(self, images, targets=None):
        feats = self.backbone(images)
        feats = feats[-3:]
        x = feats[-1]
        for layer in self.extra_layers:
            x = layer(x)
            feats.append(x)
        locs = []
        clss = []
        for feat, lh, ch in zip(feats, self.loc_heads, self.cls_heads):
            l = lh(feat); c = ch(feat)
            b, _, h, w = l.shape
            l = l.view(b, -1,4,h,w).permute(0,3,4,1,2).reshape(b,-1,4)
            c = c.view(b,-1,self.num_classes,h,w).permute(0,3,4,1,2).reshape(b,-1,self.num_classes)
            locs.append(l); clss.append(c)
        locs = torch.cat(locs, dim=1)
        clss = torch.cat(clss, dim=1)
        if self.training and targets is not None:
            return self.compute_loss(locs, clss, targets)
        return self.decode_predictions(locs, clss)

    def compute_loss(self, pred_locs, pred_clss, targets):
        batch = pred_locs.size(0)
        total = 0.0
        for i in range(batch):
            gtb = targets[i]['boxes']; gtl = targets[i]['labels']
            if len(gtb)>0:
                box_loss = nn.functional.smooth_l1_loss(pred_locs[i].mean(dim=0).unsqueeze(0).expand(len(gtb),-1), gtb, reduction='mean')
                cls_loss = nn.functional.cross_entropy(pred_clss[i][:len(gtl)], gtl, reduction='mean') if len(gtl)>0 else 0.0
                total += box_loss + cls_loss
            else:
                total += pred_clss[i].pow(2).mean()*0.01
        return total / batch

    def decode_predictions(self, locs, clss, score_thresh=0.01, nms_thresh=0.45):
        out = []
        for i in range(locs.size(0)):
            scores = torch.softmax(clss[i], dim=-1)
            boxes = locs[i]
            max_scores, labels = scores.max(dim=-1)
            keep = max_scores > score_thresh
            boxes = boxes[keep]; sc = max_scores[keep]; lb = labels[keep]
            if boxes.numel()>0:
                keep_nms = self.box_ops.nms(boxes, sc, nms_thresh)
                boxes = boxes[keep_nms]; sc = sc[keep_nms]; lb = lb[keep_nms]
            out.append({'boxes': boxes, 'scores': sc, 'labels': lb})
        return out

def train_model(args, model, train_loader, val_loader):
    device = torch.device(args.device if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp.GradScaler('cuda', enabled=args.half)
    history = {'loss':[], 'precision':[], 'recall':[], 'f1':[], 'map50':[], 'map50_95':[]}
    start = time.time()
    for epoch in range(args.epochs):
        model.train(); run_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for images, targets in pbar:
            images = torch.stack(images).to(device)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=args.half):
                loss = model(images, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = run_loss / max(len(train_loader),1)
        val_metrics, _ = validate_model(model, val_loader, device)
        history['loss'].append(avg_loss)
        for k in ['precision','recall','f1','map50','map50_95']:
            history[k].append(val_metrics[k])
        scheduler.step()
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} mAP50={val_metrics['map50']:.4f} mAP50-95={val_metrics['map50_95']:.4f}")
        if (epoch+1)%10==0 or epoch==args.epochs-1:
            torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(), 'metrics':history}, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    return history, time.time()-start, None

def validate_model(model, val_loader, device):
    model.eval(); metric = MeanAveragePrecision(iou_type='bbox')
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = torch.stack(images).to(device)
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            preds = model(images)
            metric.update(preds, targets)
            all_preds.extend([{ 'boxes': p['boxes'].detach().cpu(), 'scores': p['scores'].detach().cpu(), 'labels': p['labels'].detach().cpu() } for p in preds])
            all_targets.extend([{ 'boxes': t['boxes'].detach().cpu(), 'labels': t['labels'].detach().cpu() } for t in targets])
    res = metric.compute()
    precision = res.get('map_50', torch.tensor(0.0)).item()
    recall = res.get('map_100', torch.tensor(0.0)).item()
    f1 = 2*precision*recall/(precision+recall+1e-16) if (precision+recall)>0 else 0.0
    metrics = {'precision':precision,'recall':recall,'f1':f1,'map50':precision,'map50_95':res.get('map', torch.tensor(0.0)).item()}
    curves = build_pr_conf_curves(all_preds, all_targets)
    return metrics, curves

def build_pr_conf_curves(preds, targets, iou_thresh=0.5):
    scores = []
    tp_flags = []
    labels = []
    gt_count = 0
    num_classes = 0
    for p,t in zip(preds, targets):
        gt_boxes = t['boxes']; gt_labels = t['labels']
        pred_boxes = p['boxes']; pred_scores = p['scores']; pred_labels = p['labels']
        num_classes = max(num_classes, int(gt_labels.max().item()) if len(gt_labels)>0 else 0, int(pred_labels.max().item()) if len(pred_labels)>0 else 0)
        gt_count += len(gt_boxes)
        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
        for box, score, lab in zip(pred_boxes, pred_scores, pred_labels):
            if len(gt_boxes)>0:
                ious = box_iou_single(box, gt_boxes)
                best_iou, best_idx = ious.max(0)
                if best_iou >= iou_thresh and not matched[best_idx] and lab == gt_labels[best_idx]:
                    matched[best_idx] = True
                    tp_flags.append(1)
                else:
                    tp_flags.append(0)
            else:
                tp_flags.append(0)
            scores.append(float(score))
            labels.append(int(lab))
    num_classes = num_classes + 1
    if len(scores)==0:
        return {
            'pr_curve': {'precision':[0],'recall':[0],'confidence':[0],'f1':[0]},
            'confusion_raw': np.zeros((num_classes,num_classes), dtype=int),
            'confusion_norm': np.zeros((num_classes,num_classes), dtype=float)
        }
    order = np.argsort(-np.array(scores))
    scores_sorted = np.array(scores)[order]
    tp_sorted = np.array(tp_flags)[order]
    fp_sorted = 1 - tp_sorted
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(fp_sorted)
    recalls = cum_tp / (gt_count + 1e-16)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-16)
    f1 = 2*precisions*recalls/(precisions+recalls+1e-16)
    conf_raw = np.zeros((num_classes, num_classes), dtype=int)
    for lab, tp in zip(np.array(labels)[order], tp_sorted):
        if tp:
            conf_raw[lab, lab] += 1
        else:
            conf_raw[lab, lab] += 0  
    row_sums = conf_raw.sum(axis=1, keepdims=True) + 1e-12
    conf_norm = conf_raw / row_sums
    return {
        'pr_curve': {
            'precision': precisions.tolist(),
            'recall': recalls.tolist(),
            'confidence': scores_sorted.tolist(),
            'f1': f1.tolist()
        },
        'confusion_raw': conf_raw,
        'confusion_norm': conf_norm
    }

def box_iou_single(box, boxes):
    if len(boxes)==0:
        return torch.zeros((0,))
    x1 = torch.max(box[0], boxes[:,0])
    y1 = torch.max(box[1], boxes[:,1])
    x2 = torch.min(box[2], boxes[:,2])
    y2 = torch.min(box[3], boxes[:,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[2]-box[0]).clamp(min=0) * (box[3]-box[1]).clamp(min=0)
    area2 = (boxes[:,2]-boxes[:,0]).clamp(min=0) * (boxes[:,3]-boxes[:,1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-16
    return inter / union

def save_metrics(history, training_time, output_dir, filename):
    summary = {
        'training_time_hours': training_time/3600,
        'final_precision': history['precision'][-1],
        'final_recall': history['recall'][-1],
        'final_f1': history['f1'][-1],
        'final_map50': history['map50'][-1],
        'final_map50_95': history['map50_95'][-1],
        'best_map50': max(history['map50']),
        'best_map50_95': max(history['map50_95'])
    }
    df = pd.DataFrame([summary]); path = os.path.join(output_dir, filename); df.to_csv(path, index=False)
    print(f"Metrics saved to {path}"); return df

def plot_training_curves(history, plots_path):
    epochs = range(1, len(history['loss'])+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs, history['loss'], marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'loss_curve.png'), dpi=300, bbox_inches='tight'); plt.close()
    plt.figure(figsize=(10,6))
    for k in ['precision','recall','f1','map50','map50_95']:
        plt.plot(epochs, history[k], marker='o', label=k)
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.title('Validation Metrics'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'validation_metrics.png'), dpi=300, bbox_inches='tight'); plt.close()

def plot_pr_and_conf_curves(curves, plots_path):
    pr = curves['pr_curve']
    precision = pr['precision']; recall = pr['recall']; conf = pr['confidence']; f1 = pr['f1']
    if len(precision)==0: return
    # Precision-Recall
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label='PR')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'pr_curve.png'), dpi=300, bbox_inches='tight'); plt.close()
    # Precision vs Confidence
    plt.figure(figsize=(8,6))
    plt.plot(conf, precision); plt.xlabel('Confidence'); plt.ylabel('Precision'); plt.title('Precision vs Confidence'); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'precision_confidence.png'), dpi=300, bbox_inches='tight'); plt.close()
    # Recall vs Confidence
    plt.figure(figsize=(8,6))
    plt.plot(conf, recall); plt.xlabel('Confidence'); plt.ylabel('Recall'); plt.title('Recall vs Confidence'); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'recall_confidence.png'), dpi=300, bbox_inches='tight'); plt.close()
    # F1 vs Confidence
    plt.figure(figsize=(8,6))
    plt.plot(conf, f1); plt.xlabel('Confidence'); plt.ylabel('F1'); plt.title('F1 vs Confidence'); plt.grid(True)
    plt.savefig(os.path.join(plots_path,'f1_confidence.png'), dpi=300, bbox_inches='tight'); plt.close()
    # Approx ROC (using recall as TPR, 1-precision as FPR proxy)
    try:
        precision_arr = np.array(precision); recall_arr = np.array(recall)
        fpr = 1 - precision_arr
        order = np.argsort(fpr)
        fpr_sorted = fpr[order]; tpr_sorted = recall_arr[order]
        plt.figure(figsize=(8,6))
        plt.plot(fpr_sorted, tpr_sorted, label='ROC Approx')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Approx ROC Curve'); plt.grid(True)
        plt.savefig(os.path.join(plots_path,'roc_curve.png'), dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"ROC curve plotting failed: {e}")

def plot_confusion_matrices(curves, plots_path):
    raw = curves['confusion_raw']; norm = curves['confusion_norm']
    if raw.size==0: return
    # Raw
    plt.figure(figsize=(8,6))
    sns.heatmap(raw, annot=False, cmap='Blues', cbar=True)
    plt.title('Confusion Matrix (Raw)'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(plots_path,'confusion_matrix_raw.png'), dpi=300, bbox_inches='tight'); plt.close()
    # Normalized
    plt.figure(figsize=(8,6))
    sns.heatmap(norm, annot=False, cmap='Greens', cbar=True, vmin=0, vmax=1)
    plt.title('Confusion Matrix (Normalized)'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(plots_path,'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight'); plt.close()

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = parse_args()
    plots_path = setup_directories(args.output_dir, args.plots_dir)
    val_ann = args.val_ann_file if args.val_ann_file else args.ann_file.replace('train','val')
    train_dataset = Detection(root=args.data_path, annFile=args.ann_file, img_size=args.img_size)
    val_dataset = Detection(root=args.data_path, annFile=val_ann, img_size=args.img_size)
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    if args.num_classes is None:
        try:
            cat_ids = train_dataset.coco.getCatIds(); num_classes = len(cat_ids)
            print(f"Auto-detected {num_classes} classes")
        except Exception:
            num_classes = 80; print("Using default 80 classes")
    else:
        num_classes = args.num_classes
    model = SSDWithMobileNetV2(num_classes=num_classes, backbone_path=args.backbone_path, img_size=args.img_size)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif isinstance(ckpt, dict):
                state = ckpt.get('state_dict', ckpt)
            else:
                state = ckpt
            cleaned = {k.replace('module.',''):v for k,v in state.items()}
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"Loaded full SSD checkpoint: {args.checkpoint} | Missing: {len(missing)} Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint {args.checkpoint}: {e}")
    print("\nTraining SSD with MobileNetV2 backbone...")
    print(f"Image size: {args.img_size} | Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr} | AMP: {args.half}")
    history, train_time, _ = train_model(args, model, train_loader, val_loader)
    save_metrics(history, train_time, args.output_dir, args.metrics_file)
    metrics, curves = validate_model(model, val_loader, torch.device(args.device if torch.cuda.is_available() else 'cpu'))
    plot_training_curves(history, plots_path)
    plot_pr_and_conf_curves(curves, plots_path)
    plot_confusion_matrices(curves, plots_path)
    print("\n=== TRAINING COMPLETED ===")
    print(f"Total time: {train_time/3600:.2f} h")
    print(f"Final mAP@0.5: {history['map50'][-1]:.4f} | mAP@0.5:0.95: {history['map50_95'][-1]:.4f}")
    print(f"Best mAP@0.5: {max(history['map50']):.4f} | Best mAP@0.5:0.95: {max(history['map50_95']):.4f}")
    print(f"Results: {args.output_dir}")
    return

if __name__ == "__main__":
    main()