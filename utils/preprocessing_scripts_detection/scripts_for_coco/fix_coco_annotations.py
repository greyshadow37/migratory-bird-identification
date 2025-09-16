import os
import json
import argparse
from pathlib import Path

SPLITS = ['train', 'test', 'val']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def get_image_stems(directory):
    """
    Return set of image filenames without extensions from directory.
    Example: "ML100845971.jpg" → "ML100845971"
    """
    stems = set()
    for filename in os.listdir(directory):
        if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(Path(filename).stem)  # removes extension
    return stems

def sync_annotation_json(class_dir, split):
    """
    For a given class and split:
    - Read image stems from images/{split}/
    - Load annotations/{split}.json
    - Filter images and annotations to match only existing images
    - Write back the cleaned JSON
    """
    img_split_path = os.path.join(class_dir, 'images', split)
    ann_json_path = os.path.join(class_dir, 'annotations', f"{split}.json")

    if not os.path.exists(img_split_path):
        print(f"⚠️  Image directory missing: {img_split_path}")
        return
    if not os.path.exists(ann_json_path):
        print(f"⚠️  Annotation file missing: {ann_json_path}")
        return

    # Step 1: Get valid image stems
    valid_stems = get_image_stems(img_split_path)
    print(f"\n🔍 Syncing {class_dir.split(os.sep)[-1]} / {split} ({len(valid_stems)} images)")

    # Step 2: Load JSON
    with open(ann_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_image_count = len(data['images'])
    original_ann_count = len(data['annotations'])

    # Step 3: Filter images — keep only those with file_name matching valid_stems
    filtered_images = []
    kept_image_ids = set()

    for img in data['images']:
        stem = Path(img['file_name']).stem
        if stem in valid_stems:
            filtered_images.append(img)
            kept_image_ids.add(img['id'])

    # Step 4: Filter annotations — keep only those referencing kept image_ids
    filtered_annotations = [
        ann for ann in data['annotations'] if ann['image_id'] in kept_image_ids
    ]

    # Step 5: Update data
    data['images'] = filtered_images
    data['annotations'] = filtered_annotations

    # Step 6: Write back
    with open(ann_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    removed_images = original_image_count - len(filtered_images)
    removed_anns = original_ann_count - len(filtered_annotations)

    print(f"   ✅ Kept {len(filtered_images)} images ({removed_images} removed)")
    print(f"   ✅ Kept {len(filtered_annotations)} annotations ({removed_anns} removed)")

def main():
    parser = argparse.ArgumentParser(description="Sync JSON annotations with actual image files per class and split.")
    parser.add_argument('root_dir', type=str, help="Root directory containing class folders with 'images/' and 'annotations/'")

    args = parser.parse_args()
    root_dir = args.root_dir

    if not os.path.exists(root_dir):
        print(f"❌ Root directory '{root_dir}' does not exist.")
        return

    # Get all class directories
    class_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if not class_dirs:
        print("⚠️  No class directories found.")
        return

    # Process each class and each split
    for cls_dir in class_dirs:
        full_class_path = os.path.join(root_dir, cls_dir)
        for split in SPLITS:
            sync_annotation_json(full_class_path, split)

    print("\n🎉 All annotations synced with image files successfully!")

if __name__ == "__main__":
    main()