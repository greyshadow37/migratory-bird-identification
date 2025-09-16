import argparse
import json
from pathlib import Path
import shutil

def generate_split_json_files(root_dir):
    """
    Generate train, test, and val JSON files based on image splits
    and delete the old unsplit JSON file
    """
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
        images_dir = class_dir / 'images'
        annotations_dir = class_dir / 'annotations'
        
        # Check if directories exist
        if not images_dir.exists():
            print(f"  Images directory not found: {images_dir}")
            continue
            
        if not annotations_dir.exists():
            print(f"  Annotations directory not found: {annotations_dir}")
            continue
        
        # Look for the original JSON file (could be named various ways)
        original_json_files = list(annotations_dir.glob("*.json"))
        if not original_json_files:
            print(f"  No JSON files found in {annotations_dir}")
            continue
        
        # Use the first JSON file found as the source
        original_json_file = original_json_files[0]
        print(f"  Using source JSON: {original_json_file.name}")
        
        try:
            # Load the original COCO JSON
            with open(original_json_file, 'r') as f:
                coco_data = json.load(f)
            
            # Create mappings for quick lookup
            image_id_to_info = {img['id']: img for img in coco_data.get('images', [])}
            image_name_to_id = {img['file_name']: img['id'] for img in coco_data.get('images', [])}
            
            # Process each split
            splits = ['train', 'val', 'test']
            
            for split in splits:
                split_images_dir = images_dir / split
                
                if not split_images_dir.exists():
                    print(f"  {split} images directory not found: {split_images_dir}")
                    continue
                
                print(f"  Processing {split} split...")
                
                # Get all image files in this split
                image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                split_image_files = set()
                for ext in image_extensions:
                    for img_path in split_images_dir.glob(f"*{ext}"):
                        split_image_files.add(img_path.name)
                
                print(f"    Found {len(split_image_files)} images in {split} directory")
                
                # Filter images for this split
                split_images = []
                split_image_ids = set()
                
                for image_info in coco_data.get('images', []):
                    if image_info['file_name'] in split_image_files:
                        split_images.append(image_info)
                        split_image_ids.add(image_info['id'])
                
                # Filter annotations for this split
                split_annotations = []
                for annotation in coco_data.get('annotations', []):
                    if annotation['image_id'] in split_image_ids:
                        split_annotations.append(annotation)
                
                # Create new COCO data for this split
                split_coco_data = {
                    'info': coco_data.get('info', {}),
                    'licenses': coco_data.get('licenses', []),
                    'categories': coco_data.get('categories', []),
                    'images': split_images,
                    'annotations': split_annotations
                }
                
                # Save the split JSON file
                split_json_file = annotations_dir / f"{split}.json"
                with open(split_json_file, 'w') as f:
                    json.dump(split_coco_data, f, indent=2)
                
                print(f"    Created {split}.json with {len(split_images)} images and {len(split_annotations)} annotations")
            
            # Delete the original JSON file after successful creation of all splits
            try:
                original_json_file.unlink()
                print(f"  Deleted original JSON file: {original_json_file.name}")
            except Exception as e:
                print(f"  Error deleting original JSON file: {e}")
                
        except Exception as e:
            print(f"  Error processing JSON file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate split JSON files based on image directories')
    parser.add_argument('root_dir', type=str, help='Root directory containing class folders')
    
    print("This script will:")
    print("1. Create train.json, val.json, test.json based on images in train/val/test folders")
    print("2. Delete the original unsplit JSON file")
    print("3. Only affects annotation JSON files, not images")
    
    args = parser.parse_args()
    
    confirmation = input("\nDo you want to continue? (yes/no): ")
    if confirmation.lower() in ['yes', 'y']:
        generate_split_json_files(args.root_dir)
        print("\nOperation completed successfully!")
    else:
        print("Operation cancelled.")