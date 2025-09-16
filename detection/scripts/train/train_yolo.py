import argparse
import os
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection Training Script')
    parser.add_argument('--yolo-version', type=str, required=True,
                       choices=['yolov8', 'yolov9', 'yolov10', 'yolov11'],
                       help='YOLO version to use')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the YOLO model file (.pt)')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the data YAML file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save outputs')
    parser.add_argument('--metrics-file', type=str, default='training_metrics.csv',
                       help='Filename for metrics CSV')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for training (e.g., 0, 1, or cpu)')
    return parser.parse_args()


def setup_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_augmentation_config(version):
    augmentations = {
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'translate': 0.1, 'scale': 0.5, 'fliplr': 0.5,
        'degrees': 10.0, 'perspective': 0.001, 'shear': 2.0,
        'mosaic': 1.0 if version in ['yolov8', 'yolov9', 'yolov10', 'yolov11'] else 0.0,
        'erasing': 0.4
    }
    return augmentations


def get_model_for_version(version, model_path):
    model = YOLO(model_path)
    print(f"Using {version} for detection task")
    return model


def train_model(args, augmentations):
    start_time = time.time()
    model = get_model_for_version(args.yolo_version, args.model_path)

    # VALID ARGUMENTS ONLY — NO 'weight'
    train_args = {
        'data': args.data_path,
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': args.device,
        'augment': True,
        'save': True,
        'exist_ok': True,
        'project': args.output_dir,
        'name': f"{args.yolo_version}_detection_training",
        'verbose': True,
        'half': True,
        'cls': 1.0,           # Increase classification loss weight (default=0.5)
        'label_smoothing': 0.1,  # Reduce overconfidence on majority classes
        'save_period': 1,
        'workers': 4,
    }

    train_args.update(augmentations)

    results = model.train(**train_args)
    training_time = time.time() - start_time

    return model, results, training_time


def save_epoch_wise_metrics(results, output_dir, filename):
    """Save epoch-wise training metrics to CSV"""
    df = pd.DataFrame(results.results_dict)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"✅ Epoch-wise metrics saved to {filepath}")


def main():
    args = parse_args()
    output_dir = setup_directories(args.output_dir)

    augmentations = create_augmentation_config(args.yolo_version)

    print(f"🚀 Training {args.yolo_version} model for detection task...")
    model, results, training_time = train_model(args, augmentations)

    # Save full epoch-wise history
    save_epoch_wise_metrics(results, output_dir, args.metrics_file)

    # Extract final metrics
    final_map50 = results.results_dict.get('metrics/mAP50(B)', [0])[-1]
    final_map50_95 = results.results_dict.get('metrics/mAP50-95(B)', [0])[-1]
    final_precision = results.results_dict.get('metrics/precision(B)', [0])[-1]
    final_recall = results.results_dict.get('metrics/recall(B)', [0])[-1]

    print(f"\n📊 Training Summary:")
    print(f"YOLO Version: {args.yolo_version}")
    print(f"Task: Detection")
    print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"mAP@0.5: {final_map50:.4f}")
    print(f"mAP@0.5:0.95: {final_map50_95:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")

    print(f"\n✅ Training completed! Results saved to {output_dir}")
    print("💡 All plots (loss, PR, confusion matrix, etc.) were saved automatically by Ultralytics.")


if __name__ == "__main__":
    main()