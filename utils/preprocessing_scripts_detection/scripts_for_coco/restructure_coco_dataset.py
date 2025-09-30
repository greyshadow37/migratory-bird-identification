"""
COCO Dataset Restructuring Script

This script merges per-class COCO annotation files into a unified COCO dataset format
that is compatible with PyTorch detection training scripts (SSD, EfficientDet).

The script handles:
1. Merging annotations from multiple class-specific JSON files
2. Reorganizing images into unified train/val/test directories
3. Remapping image_id and annotation_id to avoid conflicts
4. Updating file paths in annotations
5. Creating a consolidated data.yaml file

"""

import argparse
import json
import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Restructure per-class COCO dataset into unified format for PyTorch training'
    )
    parser.add_argument(
        '--input-dir', 
        type=str, 
        required=True,
        help='Path to input directory containing per-class COCO datasets (e.g., data/undersampled_datasets/data-coco)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        required=True,
        help='Path to output directory for unified COCO dataset (will be created if it doesn\'t exist)'
    )
    parser.add_argument(
        '--splits', 
        nargs='+', 
        default=['train', 'val', 'test'],
        help='Dataset splits to process (default: train val test)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Perform a dry run without actually copying files or writing outputs'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def get_class_folders(input_dir: str) -> List[str]:
    """Get list of class folders in the input directory."""
    input_path = Path(input_dir)
    class_folders = []
    
    for item in input_path.iterdir():
        if item.is_dir() and item.name != '.git' and not item.name.startswith('.'):
            # Check if it has the expected structure (annotations and images folders)
            if (item / 'annotations').exists() and (item / 'images').exists():
                class_folders.append(item.name)
    
    logger.info(f"Found {len(class_folders)} class folders: {class_folders}")
    return sorted(class_folders)


def load_coco_annotation(file_path: str) -> Dict:
    """Load COCO annotation file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def get_unique_categories(input_dir: str, class_folders: List[str]) -> List[Dict]:
    """Extract unique categories from all annotation files."""
    categories_map = {}
    
    # Load categories from the first available annotation file
    for class_folder in class_folders:
        for split in ['train', 'val', 'test']:
            ann_file = Path(input_dir) / class_folder / 'annotations' / f'{split}.json'
            if ann_file.exists():
                data = load_coco_annotation(str(ann_file))
                if data and 'categories' in data:
                    for cat in data['categories']:
                        categories_map[cat['id']] = cat
                    break
        if categories_map:
            break
    
    # Return sorted categories by ID
    categories = [categories_map[cat_id] for cat_id in sorted(categories_map.keys())]
    logger.info(f"Found {len(categories)} categories: {[cat['name'] for cat in categories]}")
    return categories


def merge_annotations_for_split(
    input_dir: str, 
    class_folders: List[str], 
    split: str, 
    categories: List[Dict]
) -> Tuple[Dict, Dict[str, str]]:
    """
    Merge annotations from all class folders for a specific split.
    
    Returns:
        merged_data: Complete COCO annotation structure
        image_mapping: Dictionary mapping old file paths to new file paths
    """
    merged_data = {
        "info": {
            "contributor": "Migratory Bird Detection Project",
            "date_created": "2025-09-30",
            "description": f"Unified {split} dataset for migratory bird detection",
            "url": "",
            "version": "1.0",
            "year": "2025"
        },
        "licenses": [
            {
                "name": "Unknown",
                "id": 0,
                "url": ""
            }
        ],
        "categories": categories,
        "images": [],
        "annotations": []
    }
    
    current_image_id = 1
    current_annotation_id = 1
    image_id_mapping = {}  # Maps old image_id to new image_id
    image_mapping = {}  # Maps old file path to new file path
    
    for class_folder in class_folders:
        ann_file = Path(input_dir) / class_folder / 'annotations' / f'{split}.json'
        
        if not ann_file.exists():
            logger.warning(f"Annotation file not found: {ann_file}")
            continue
            
        logger.info(f"Processing {class_folder}/{split}.json")
        data = load_coco_annotation(str(ann_file))
        
        if not data:
            continue
            
        # Process images
        for img in data.get('images', []):
            old_image_id = img['id']
            new_image_id = current_image_id
            
            # Update image entry
            new_img = img.copy()
            new_img['id'] = new_image_id
            
            original_file_name = img['file_name']
            new_img['file_name'] = original_file_name
            
            merged_data['images'].append(new_img)
            
            image_id_mapping[old_image_id] = new_image_id
            old_path = f"{class_folder}/images/{split}/{original_file_name}"
            new_path = f"{split}/{original_file_name}"
            image_mapping[old_path] = new_path
            
            current_image_id += 1
        
        # Process annotations
        for ann in data.get('annotations', []):
            if ann['image_id'] in image_id_mapping:
                new_ann = ann.copy()
                new_ann['id'] = current_annotation_id
                new_ann['image_id'] = image_id_mapping[ann['image_id']]
                
                merged_data['annotations'].append(new_ann)
                current_annotation_id += 1
    
    logger.info(f"Merged {split}: {len(merged_data['images'])} images, {len(merged_data['annotations'])} annotations")
    return merged_data, image_mapping


def copy_images(input_dir: str, output_dir: str, image_mappings: Dict[str, Dict[str, str]], dry_run: bool = False):
    """Copy images from class-specific folders to unified directories."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    total_copied = 0
    skipped = 0
    
    for split, mapping in image_mappings.items():
        split_output_dir = output_path / split
        
        if not dry_run:
            split_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Copying {len(mapping)} images for {split} split...")
        
        for old_path, new_path in mapping.items():
            src_file = input_path / old_path
            dst_file = output_path / new_path
            
            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                skipped += 1
                continue
            
            if not dry_run:
                # Create destination directory if it doesn't exist
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                try:
                    shutil.copy2(src_file, dst_file)
                    total_copied += 1
                except Exception as e:
                    logger.error(f"Error copying {src_file} to {dst_file}: {e}")
                    skipped += 1
            else:
                logger.info(f"Would copy: {src_file} -> {dst_file}")
                total_copied += 1
    
    logger.info(f"Image copying complete. Copied: {total_copied}, Skipped: {skipped}")


def save_annotation_file(data: Dict, output_file: str, dry_run: bool = False):
    """Save merged annotation data to JSON file."""
    if dry_run:
        logger.info(f"Would save annotation file: {output_file}")
        return
    
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved annotation file: {output_file}")
    except Exception as e:
        logger.error(f"Error saving annotation file {output_file}: {e}")


def create_data_yaml(output_dir: str, categories: List[Dict], splits: List[str], dry_run: bool = False):
    """Create a YOLO-style data.yaml file for reference."""
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    
    data_yaml = {
        'path': str(Path(output_dir).absolute()),
        'train': 'train',
        'val': 'val',
        'test': 'test' if 'test' in splits else 'val',
        'nc': len(class_names),
        'names': class_names,
        'description': 'Unified COCO dataset for migratory bird detection',
        'created': '2025-09-30',
        'splits': splits
    }
    
    yaml_file = Path(output_dir) / 'data.yaml'
    
    if dry_run:
        logger.info(f"Would create data.yaml: {yaml_file}")
        logger.info(f"Content: {data_yaml}")
        return
    
    try:
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created data.yaml: {yaml_file}")
    except Exception as e:
        logger.error(f"Error creating data.yaml: {e}")


def validate_dataset(output_dir: str, splits: List[str]):
    """Validate the created dataset structure."""
    output_path = Path(output_dir)
    issues = []
    
    # Check directory structure
    for split in splits:
        split_dir = output_path / split
        ann_file = output_path / f'{split}.json'
        
        if not split_dir.exists():
            issues.append(f"Missing {split} directory")
        
        if not ann_file.exists():
            issues.append(f"Missing {split}.json file")
        else:
            # Validate annotation file
            try:
                data = load_coco_annotation(str(ann_file))
                if not data:
                    issues.append(f"Invalid {split}.json file")
                else:
                    # Check consistency
                    image_files = {img['file_name'] for img in data.get('images', [])}
                    actual_files = {f.name for f in split_dir.glob('*') if f.is_file()}
                    
                    missing_files = image_files - actual_files
                    extra_files = actual_files - image_files
                    
                    if missing_files:
                        issues.append(f"{split}: {len(missing_files)} images referenced in JSON but missing from directory")
                    
                    if extra_files:
                        issues.append(f"{split}: {len(extra_files)} extra images in directory not referenced in JSON")
            
            except Exception as e:
                issues.append(f"Error validating {split}.json: {e}")
    
    # Check data.yaml
    yaml_file = output_path / 'data.yaml'
    if not yaml_file.exists():
        issues.append("Missing data.yaml file")
    
    if issues:
        logger.warning("Validation issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Dataset validation passed!")


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory
    output_path = Path(args.output_dir)
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Would create output directory: {args.output_dir}")
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Processing splits: {args.splits}")
    logger.info(f"Dry run: {args.dry_run}")
    
    # Get class folders
    class_folders = get_class_folders(args.input_dir)
    if not class_folders:
        logger.error("No valid class folders found in input directory")
        return 1
    
    # Get categories
    categories = get_unique_categories(args.input_dir, class_folders)
    if not categories:
        logger.error("No categories found in annotation files")
        return 1
    
    # Process each split
    all_image_mappings = {}
    
    for split in args.splits:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {split} split")
        logger.info(f"{'='*50}")
        
        # Merge annotations
        merged_data, image_mapping = merge_annotations_for_split(
            args.input_dir, class_folders, split, categories
        )
        
        # Save annotation file
        output_ann_file = output_path / f'{split}.json'
        save_annotation_file(merged_data, str(output_ann_file), args.dry_run)
        
        # Store image mappings for later copying
        all_image_mappings[split] = image_mapping
    
    # Copy all images
    logger.info(f"\n{'='*50}")
    logger.info("Copying images")
    logger.info(f"{'='*50}")
    copy_images(args.input_dir, args.output_dir, all_image_mappings, args.dry_run)
    
    # Create data.yaml
    logger.info(f"\n{'='*50}")
    logger.info("Creating data.yaml")
    logger.info(f"{'='*50}")
    create_data_yaml(args.output_dir, categories, args.splits, args.dry_run)
    
    # Validate dataset (only if not dry run)
    if not args.dry_run:
        logger.info(f"\n{'='*50}")
        logger.info("Validating dataset")
        logger.info(f"{'='*50}")
        validate_dataset(args.output_dir, args.splits)
    
    logger.info(f"\n{'='*50}")
    logger.info("Dataset restructuring complete!")
    logger.info(f"{'='*50}")
    
    if not args.dry_run:
        logger.info(f"Unified dataset available at: {args.output_dir}")
        logger.info("You can now use this dataset with the training scripts:")
        logger.info(f"  --data-path {args.output_dir}")
        logger.info(f"  --ann-file {args.output_dir}/train.json")
        logger.info(f"  --val-ann-file {args.output_dir}/val.json")
    
    return 0


if __name__ == '__main__':
    exit(main())