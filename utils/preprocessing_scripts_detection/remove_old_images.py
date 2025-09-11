import argparse
from pathlib import Path
import shutil

def cleanup_and_rename_images(root_dir):
    """
    Safely cleanup and rename images in the dataset.
    Step 1: Remove all images that don't start with 'clean_'
    Step 2: Rename 'clean_<filename>' back to '<filename>'
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
        
        # Define images directory path
        images_dir = class_dir / 'images' / 'train'
        
        # Check if images directory exists
        if not images_dir.exists():
            print(f"  Images directory not found: {images_dir}")
            continue
        
        # Step 1: Remove all non-clean images
        print("  Step 1: Removing non-clean images...")
        non_clean_removed = 0
        for ext in image_extensions:
            for img_path in images_dir.glob(f"*{ext}"):
                if not img_path.name.startswith('clean_'):
                    try:
                        img_path.unlink()
                        print(f"    Removed: {img_path.name}")
                        non_clean_removed += 1
                    except Exception as e:
                        print(f"    Error removing {img_path.name}: {e}")
        
        print(f"    Removed {non_clean_removed} non-clean images")
        
        # Step 2: Rename clean images back to original names
        print("  Step 2: Renaming clean images...")
        renamed_count = 0
        for ext in image_extensions:
            for clean_img_path in images_dir.glob(f"clean_*{ext}"):
                # Extract original filename (remove 'clean_' prefix)
                original_name = clean_img_path.name[6:]  # Remove first 6 characters 'clean_'
                original_path = images_dir / original_name
                
                try:
                    # Rename the file
                    shutil.move(str(clean_img_path), str(original_path))
                    print(f"    Renamed: {clean_img_path.name} -> {original_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"    Error renaming {clean_img_path.name}: {e}")
        
        print(f"    Renamed {renamed_count} clean images")
        
        # Final check
        remaining_images = sum(1 for _ in images_dir.glob(f"*{ext}") for ext in image_extensions)
        print(f"  Final count: {remaining_images} images remaining")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup and rename processed images')
    parser.add_argument('root_dir', type=str, help='Root directory containing class folders')
    
    # Safety confirmation
    print("WARNING: This script will:")
    print("1. DELETE all images that don't start with 'clean_'")
    print("2. Rename 'clean_<filename>' back to '<filename>'")
    print("3. Only affects images folders, labels are untouched")
    
    args = parser.parse_args()
    
    confirmation = input("\nDo you want to continue? This cannot be undone! (yes/no): ")
    if confirmation.lower() in ['yes', 'y']:
        cleanup_and_rename_images(args.root_dir)
        print("\nOperation completed successfully!")
    else:
        print("Operation cancelled.")