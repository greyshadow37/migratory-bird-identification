import os
import random
import shutil
from pathlib import Path

def split_dataset(root_dir, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, seed=42):
    """
    Split dataset into train, test, and validation sets by moving files.
    
    Args:
        root_dir (str): Root directory containing class folders
        train_ratio (float): Proportion for training set
        test_ratio (float): Proportion for test set
        val_ratio (float): Proportion for validation set
        seed (int): Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    # Process each class directory
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        print(f"\nProcessing class: {class_dir.name}")
        
        # Define paths
        images_train_dir = class_dir / 'images' / 'train'
        labels_train_dir = class_dir / 'labels' / 'train'
        
        # Create test and val directories
        images_test_dir = class_dir / 'images' / 'test'
        images_val_dir = class_dir / 'images' / 'val'
        labels_test_dir = class_dir / 'labels' / 'test'
        labels_val_dir = class_dir / 'labels' / 'val'
        
        # Create directories if they don't exist
        for dir_path in [images_test_dir, images_val_dir, labels_test_dir, labels_val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_train_dir.glob(f"*{ext}"))
        
        # Filter out images without corresponding labels
        valid_image_files = []
        for img_path in image_files:
            label_path = labels_train_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                valid_image_files.append(img_path)
            else:
                print(f"  Skipping {img_path.name} (no label file)")
        
        # Shuffle the files
        random.shuffle(valid_image_files)
        total_files = len(valid_image_files)
        
        if total_files == 0:
            print(f"  No valid image-label pairs found for {class_dir.name}")
            continue
        
        # Calculate split indices
        train_end = int(total_files * train_ratio)
        test_end = train_end + int(total_files * test_ratio)
        
        # Split into sets
        train_files = valid_image_files[:train_end]
        test_files = valid_image_files[train_end:test_end]
        val_files = valid_image_files[test_end:]
        
        print(f"  Total files: {total_files}")
        print(f"  Train: {len(train_files)}, Test: {len(test_files)}, Val: {len(val_files)}")
        
        # Move files to respective directories
        def move_files(files, dest_images_dir, dest_labels_dir):
            moved_count = 0
            for img_path in files:
                # Move image file
                dest_img_path = dest_images_dir / img_path.name
                shutil.move(str(img_path), str(dest_img_path))
                
                # Move corresponding label file
                label_path = labels_train_dir / f"{img_path.stem}.txt"
                dest_label_path = dest_labels_dir / f"{img_path.stem}.txt"
                shutil.move(str(label_path), str(dest_label_path))
                
                moved_count += 1
            return moved_count
        
        # Move files to their respective directories
        train_moved = move_files(train_files, images_train_dir, labels_train_dir)
        test_moved = move_files(test_files, images_test_dir, labels_test_dir)
        val_moved = move_files(val_files, images_val_dir, labels_val_dir)
        
        print(f"  Moved: {train_moved} train, {test_moved} test, {val_moved} val")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/test/val sets')
    parser.add_argument('root_dir', type=str, help='Root directory containing class folders')
    parser.add_argument('--train', type=float, default=0.8, help='Training set ratio (default: 0.8)')
    parser.add_argument('--test', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    parser.add_argument('--val', type=float, default=0.1, help='Validation set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1.0
    total_ratio = args.train + args.test + args.val
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Warning: Ratios sum to {total_ratio:.3f}, but should sum to 1.0")
        # Normalize the ratios
        args.train /= total_ratio
        args.test /= total_ratio
        args.val /= total_ratio
        print(f"Normalized ratios: Train={args.train:.3f}, Test={args.test:.3f}, Val={args.val:.3f}")
    
    print(f"Splitting dataset with ratios: Train={args.train}, Test={args.test}, Val={args.val}")
    print(f"Using random seed: {args.seed}")
    
    split_dataset(args.root_dir, args.train, args.test, args.val, args.seed)