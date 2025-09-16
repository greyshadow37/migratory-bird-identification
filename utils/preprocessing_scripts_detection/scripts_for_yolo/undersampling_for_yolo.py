import argparse
import random
from pathlib import Path

def undersample_classes(root_dir, target_count=452, seed=42):
    """
    Undersample all classes to the specified target count
    """
    random.seed(seed)
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    print(f"Undersampling all classes to {target_count} samples each")
    print(f"Using random seed: {seed}")
    
    # Process each class directory
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nProcessing {class_name}...")
        
        # Define paths
        images_dir = class_dir / 'images' / 'train'
        labels_dir = class_dir / 'labels' / 'train'
        
        # Check if directories exist
        if not images_dir.exists():
            print(f"  Images directory not found: {images_dir}")
            continue
            
        if not labels_dir.exists():
            print(f"  Labels directory not found: {labels_dir}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        # Filter only images with corresponding labels
        valid_image_files = []
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_image_files.append(img_path)
        
        current_count = len(valid_image_files)
        print(f"  Current count: {current_count}")
        
        if current_count <= target_count:
            print(f"  Already at or below target ({target_count}), skipping")
            continue
        
        # Randomly select target number of samples to keep
        files_to_keep = random.sample(valid_image_files, target_count)
        files_to_remove = set(valid_image_files) - set(files_to_keep)
        
        print(f"  Removing {len(files_to_remove)} files to reach target {target_count}")
        
        # Remove excess files
        removed_count = 0
        for img_path in files_to_remove:
            try:
                # Remove image file
                img_path.unlink()
                # Remove corresponding label file
                label_path = labels_dir / f"{img_path.stem}.txt"
                label_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {img_path.name}: {e}")
        
        print(f"  Successfully removed {removed_count} files")
        print(f"  Final count: {target_count}")

def print_class_distribution(root_dir):
    """
    Print current class distribution before undersampling
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    print("Current class distribution:")
    print("-" * 30)
    
    class_counts = {}
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images_dir = class_dir / 'images' / 'train'
        
        if images_dir.exists():
            image_count = 0
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_count += len(list(images_dir.glob(f"*{ext}")))
            class_counts[class_name] = image_count
            print(f"  {class_name}: {image_count} images")
    
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        avg_count = sum(class_counts.values()) / len(class_counts)
        
        print("-" * 30)
        print(f"Minimum: {min_count}")
        print(f"Maximum: {max_count}")
        print(f"Average: {avg_count:.1f}")
        print(f"Total: {sum(class_counts.values())}")
    
    return class_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Undersample classes to balance dataset')
    parser.add_argument('root_dir', type=str, help='Root directory containing class folders')
    parser.add_argument('--target', type=int, default=452, 
                       help='Target count for each class (default: 452)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually modifying files')
    
    args = parser.parse_args()
    
    # Show current distribution
    class_counts = print_class_distribution(args.root_dir)
    
    if not class_counts:
        print("No class directories found with images")
        exit(1)
    
    # Check if undersampling is needed
    min_count = min(class_counts.values())
    if args.target > min_count:
        print(f"\nWARNING: Target count ({args.target}) is higher than smallest class ({min_count})")
        print("Some classes will need oversampling instead of undersampling")
        response = input("Do you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            exit(0)
    
    if args.dry_run:
        print(f"\nDRY RUN: Would undersample all classes to {args.target} samples each")
        print("No files will be modified")
    else:
        # Safety confirmation
        print(f"\nWARNING: This will permanently delete files to balance classes to {args.target} samples each!")
        confirmation = input("Do you want to continue? This cannot be undone! (yes/no): ")
        
        if confirmation.lower() in ['yes', 'y']:
            undersample_classes(args.root_dir, args.target, args.seed)
            print("\nUndersampling completed!")
            
            # Show final distribution
            print("\nFinal class distribution:")
            print_class_distribution(args.root_dir)
        else:
            print("Operation cancelled.")