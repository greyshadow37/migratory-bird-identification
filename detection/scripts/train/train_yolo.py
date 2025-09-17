import argparse
import os
import time
import pandas as pd
from pathlib import Path
import optuna
from ultralytics import YOLO
import shutil
import gc
import torch

# Suppress excessive logging if needed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection Training with Optuna Hyperparameter Tuning')
    parser.add_argument('--yolo-version', type=str, required=True,
                        choices=['yolov8', 'yolov9', 'yolov10', 'yolov11'],
                        help='YOLO version to use')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the YOLO model file (.pt)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the data YAML file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save outputs and study results')
    parser.add_argument('--metrics-file', type=str, default='training_metrics.csv',
                        help='Base filename for metrics CSV (trial-specific)')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs per trial')
    parser.add_argument('--patience', type=int, default=10,
                        help='Epochs to wait for improvement before early stopping')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use for training (e.g., 0, 1, or cpu)')
    parser.add_argument('--study-name', type=str, default='yolo-detection-optuna',
                        help='Name of the Optuna study')
    return parser.parse_args()


def setup_directories(output_dir):
    """Setup output directories and validate paths"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_inputs(args):
    """Validate input files and paths"""
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")


def get_model_for_version(version, model_path):
    """Initialize model for the specified YOLO version"""
    try:
        model = YOLO(model_path)
        print(f"Using {version} for detection task")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_path}: {e}")


def cleanup_resources():
    """Clean up GPU memory and force garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_get_metric(results_dict, metric_key, default=0.0):
    """Safely extract metric from results dictionary"""
    metric_values = results_dict.get(metric_key, [default])
    return metric_values[-1] if metric_values else default


def train_model(args, lr0, batch_size, img_size, cls_weight, dfl_weight, trial_number):
    """Train YOLO model with given hyperparameters"""
    start_time = time.time()
    model = None
    
    try:
        model = get_model_for_version(args.yolo_version, args.model_path)

        train_args = {
            'data': args.data_path,
            'epochs': args.epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': args.device,
            'augment': False,
            'save': True,
            'exist_ok': True,
            'project': args.output_dir,
            'name': f"{args.yolo_version}_trial_{trial_number}",
            'verbose': False,
            'half': True,
            'cls': cls_weight,
            'dfl': dfl_weight,
            'lr0': lr0,
            'label_smoothing': 0.1,
            'save_period': -1,
            'workers': 2,
            'patience': args.patience, 
            'cache': False
            }

        results = model.train(**train_args)
        training_time = time.time() - start_time
        return model, results, training_time
        
    except Exception as e:
        print(f"❌ Trial {trial_number} failed: {e}")
        raise optuna.TrialPruned()
    finally:
        # Clean up model and resources
        if model is not None:
            del model
        cleanup_resources()


def save_epoch_wise_metrics(results, output_dir, filename):
    """Save epoch-wise training metrics to CSV with error handling"""
    try:
        if hasattr(results, 'results_dict') and results.results_dict:
            df = pd.DataFrame(results.results_dict)
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"✅ Epoch-wise metrics saved to {filepath}")
        else:
            print(f"⚠️ No results data available for {filename}")
    except Exception as e:
        print(f"⚠️ Could not save metrics to {filename}: {e}")


def objective(trial, args):
    """Optuna objective function"""
    # Suggest hyperparameters
    lr0 = trial.suggest_float('lr0', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4])
    img_size = trial.suggest_categorical('img_size', [320, 480])
    cls_weight = trial.suggest_float('cls_weight', 0.5, 1.5)
    dfl_weight = trial.suggest_float('dfl_weight', 0.5, 1.5)

    print(f"\n🚀 Starting Trial {trial.number} with params:")
    print(f"  LR: {lr0}, Batch: {batch_size}, ImgSize: {img_size}")
    print(f"  Cls Weight: {cls_weight}, DFL Weight: {dfl_weight}")

    try:
        model, results, training_time = train_model(
            args, lr0, batch_size, img_size, cls_weight, dfl_weight, trial.number
        )

        # Save per-trial metrics
        metrics_file = f"trial_{trial.number}_{args.metrics_file}"
        save_epoch_wise_metrics(results, args.output_dir, metrics_file)

        # Safely extract metrics
        results_dict = results.results_dict if hasattr(results, 'results_dict') else {}
        
        # Objective: maximize mAP@0.5:0.95
        final_map50_95 = safe_get_metric(results_dict, 'metrics/mAP50-95(B)', 0.0)

        # Also log other metrics as trial user attributes
        trial.set_user_attr('mAP50', safe_get_metric(results_dict, 'metrics/mAP50(B)', 0.0))
        trial.set_user_attr('precision', safe_get_metric(results_dict, 'metrics/precision(B)', 0.0))
        trial.set_user_attr('recall', safe_get_metric(results_dict, 'metrics/recall(B)', 0.0))
        trial.set_user_attr('training_time', training_time)

        print(f"✅ Trial {trial.number} completed - mAP@0.5:0.95: {final_map50_95:.4f}")
        
        return final_map50_95

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"❌ Trial {trial.number} failed with exception: {e}")
        raise optuna.TrialPruned()


def copy_best_model(args, best_trial_number, output_dir):
    """Copy the best model weights with error handling"""
    try:
        best_model_path = os.path.join(
            output_dir, 
            f"{args.yolo_version}_trial_{best_trial_number}", 
            "weights", 
            "best.pt"
        )
        
        if os.path.exists(best_model_path):
            final_best_path = os.path.join(output_dir, "best_model.pt")
            shutil.copy(best_model_path, final_best_path)
            print(f"✅ Best model weights saved to {final_best_path}")
            return final_best_path
        else:
            print(f"⚠️ Best model weights not found at {best_model_path}")
            return None
    except Exception as e:
        print(f"⚠️ Failed to copy best model: {e}")
        return None


def main():
    args = parse_args()
    
    # Validate inputs
    try:
        validate_inputs(args)
    except FileNotFoundError as e:
        print(f"❌ Input validation failed: {e}")
        return
    
    output_dir = setup_directories(args.output_dir)

    # Create Optuna study with better error handling
    try:
        study = optuna.create_study(
            study_name=args.study_name,
            direction='maximize',
            load_if_exists=True
        )
    except Exception as e:
        print(f"❌ Failed to create Optuna study: {e}")
        return

    print(f"🎯 Starting Optuna hyperparameter optimization for {args.yolo_version}...")
    print(f"🧪 Running {args.n_trials} trials...")

    try:
        study.optimize(
            lambda trial: objective(trial, args), 
            n_trials=args.n_trials, 
            catch=(optuna.TrialPruned,)
        )
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # Check if we have any completed trials
    if not study.trials or not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print("❌ No trials completed successfully")
        return

    # Print best trial
    try:
        best_trial = study.best_trial
        print(f"\n🏆 Best trial: {best_trial.number}")
        print(f"📈 Best mAP@0.5:0.95: {best_trial.value:.4f}")
        print("🔧 Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print("\n📊 Additional metrics for best trial:")
        for key, value in best_trial.user_attrs.items():
            print(f"    {key}: {value}")

        # Save study results
        study_df = study.trials_dataframe()
        study_csv = os.path.join(output_dir, f"{args.study_name}_results.csv")
        study_df.to_csv(study_csv, index=False)
        print(f"\n✅ Study results saved to {study_csv}")

        # Copy best model weights
        copy_best_model(args, best_trial.number, output_dir)

        print(f"\n🎉 Hyperparameter tuning completed! All results in {output_dir}")
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")


if __name__ == "__main__":
    main()