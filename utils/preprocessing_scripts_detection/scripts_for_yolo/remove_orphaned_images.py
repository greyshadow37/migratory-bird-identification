import argparse
from pathlib import Path

def delete_images_without_labels(root_dir, confirm=True):
    """
    Delete images that don't have corresponding label files in YOLO dataset structure.
    
    Args:
        root_dir (str): Root directory containing class folders
        confirm (bool): Whether to ask for confirmation before deleting
    """
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"Root directory does not exist: {root_dir}")
        return
    
    # Safety check - confirm before proceeding
    if confirm:
        print("WARNING: This script will DELETE image files permanently!")
        print(f"Target directory: {root_path}")
        confirmation = input("Do you want to continue? (yes/no): ")
        
        if confirmation.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Walk through all class directories
    class_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
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
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        deleted_count = 0
        kept_count = 0
        
        # Check each image for corresponding label
        for image_path in image_files:
            image_name = image_path.stem  # Get filename without extension
            label_path = labels_dir / f"{image_name}.txt"
            
            if not label_path.exists():
                # Delete the image
                try:
                    image_path.unlink()  # Pathlib method to delete file
                    print(f"  Deleted: {image_path.name} (no label file)")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {image_path}: {e}")
            else:
                kept_count += 1
        
        print(f"  Summary: {deleted_count} images deleted, {kept_count} images kept")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Delete images without corresponding label files')
    parser.add_argument('root_dir', type=str, help='Root directory containing bird class folders')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt (use with caution)')
    
    args = parser.parse_args()
    
    delete_images_without_labels(args.root_dir, confirm=not args.no_confirm)