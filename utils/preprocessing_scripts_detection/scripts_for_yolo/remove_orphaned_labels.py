import argparse
from pathlib import Path

def remove_orphaned_labels(root_dir):
    """
    Remove label files that don't have corresponding image files
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Process each class directory
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        print(f"\nProcessing class: {class_dir.name}")
        
        # Define paths for images and labels
        images_dir = class_dir / 'images' / 'train'
        labels_dir = class_dir / 'labels' / 'train'
        
        # Check if directories exist
        if not images_dir.exists():
            print(f"  Images directory not found: {images_dir}")
            continue
            
        if not labels_dir.exists():
            print(f"  Labels directory not found: {labels_dir}")
            continue
        
        # Get all image files (without extensions for comparison)
        image_files = set()
        for ext in image_extensions:
            for img_path in images_dir.glob(f"*{ext}"):
                image_files.add(img_path.stem)  # Filename without extension
        
        print(f"  Found {len(image_files)} images")
        
        # Get all label files
        label_files = list(labels_dir.glob("*.txt"))
        print(f"  Found {len(label_files)} label files")
        
        # Remove labels without corresponding images
        removed_count = 0
        kept_count = 0
        
        for label_path in label_files:
            label_stem = label_path.stem  # Filename without .txt extension
            
            if label_stem not in image_files:
                # This label has no corresponding image - remove it
                try:
                    label_path.unlink()
                    print(f"    Removed orphaned label: {label_path.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"    Error removing {label_path.name}: {e}")
            else:
                kept_count += 1
        
        print(f"  Removed {removed_count} orphaned labels, kept {kept_count} valid labels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove label files without corresponding images')
    parser.add_argument('root_dir', type=str, help='Root directory containing class folders')
    
    # Safety confirmation
    print("WARNING: This script will DELETE label files that don't have corresponding images!")
    print("This operation cannot be undone.")
    
    args = parser.parse_args()
    
    confirmation = input("\nDo you want to continue? (yes/no): ")
    if confirmation.lower() in ['yes', 'y']:
        remove_orphaned_labels(args.root_dir)
        print("\nOperation completed successfully!")
    else:
        print("Operation cancelled.")