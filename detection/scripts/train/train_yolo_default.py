from ultralytics import YOLO
import argparse

def train(model, data, epochs, batch_size, imgsz, lr0, output_dir):
    model.train(
        data=data,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr0,
        half=True,
        workers=2,
        save=True,
        plots=True,
        optimizer='AdamW',
        patience=5,
        project=output_dir,
        name='output',
        exist_ok=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with default settings.")
    parser.add_argument('--yolo-version', type=str, choices=['yolov8', 'yolov9','yolov10', 'yolov11'], help='YOLO version to use')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pre-trained model weights')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--img-size', type=int, default=512, help='Image size (default: 512)')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate (default: 0.01)')
    parser.add_argument('--output-dir', type=str, help='Output directory for results (default: results/yolo_run)')

    args = parser.parse_args()

    # Load model
    model = YOLO(args.model_path)

    # Train model
    train(
        model=model,
        data=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr0,
        output_dir=args.output_dir
    )
