import os
import argparse
import random
from pathlib import Path

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def get_image_files(directory):
    """Return list of image files in directory with supported extensions."""
    return [f for f in os.listdir(directory) if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

def get_annotation_file(image_file):
    """Given an image filename, return expected annotation filename (.json)."""
    return Path(image_file).with_suffix('.json').name

def undersample_split_by_class(root_dir, split_name):
    """
    Process one split (train/test/val) across all classes.
    Each class has: images/{split_name}/ and annotations/
    Annotation file must match image name and be in 'annotations/' dir.
    """
    print(f"\n📊 Processing split: '{split_name}'")

    # Get all class directories (e.g., "asian-green-bee-eater", "cattle-egret", etc.)
    class_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if not class_dirs:
        print(f"⚠️  No class directories found in {root_dir}. Skipping.")
        return

    # Dictionary to store count of images per class for this split
    class_counts = {}

    for cls in class_dirs:
        img_split_path = os.path.join(root_dir, cls, 'images', split_name)
        ann_path = os.path.join(root_dir, cls, 'annotations')

        if not os.path.exists(img_split_path):
            print(f"   ⚠️  Missing images/{split_name} for class '{cls}'. Skipping class.")
            continue

        if not os.path.exists(ann_path):
            print(f"   ⚠️  Missing annotations/ for class '{cls}'. Skipping class.")
            continue

        images = get_image_files(img_split_path)
        class_counts[cls] = len(images)

    if not class_counts:
        print(f"   ⚠️  No valid images found in any class for split '{split_name}'. Skipping.")
        return

    min_count = min(class_counts.values())
    print(f"   Class counts: {class_counts}")
    print(f"   Minimum count across classes: {min_count}")

    # If already balanced, skip
    if all(count == min_count for count in class_counts.values()):
        print(f"   ✅ Already balanced. No changes needed.")
        return

    # For each class, remove excess images randomly
    for cls in class_dirs:
        img_split_path = os.path.join(root_dir, cls, 'images', split_name)
        ann_path = os.path.join(root_dir, cls, 'annotations')

        if not os.path.exists(img_split_path) or not os.path.exists(ann_path):
            continue

        images = get_image_files(img_split_path)
        num_to_remove = len(images) - min_count

        if num_to_remove <= 0:
            continue

        random.shuffle(images)
        images_to_remove = images[:num_to_remove]

        removed_count = 0
        for img in images_to_remove:
            img_path = os.path.join(img_split_path, img)
            ann_file = get_annotation_file(img)
            ann_full_path = os.path.join(ann_path, ann_file)

            # Remove image
            os.remove(img_path)
            removed_count += 1

            # Remove corresponding annotation if exists
            if os.path.exists(ann_full_path):
                os.remove(ann_full_path)
                # Optional: Log removal
                # print(f"      Removed annotation: {ann_file}")
            else:
                print(f"   ⚠️  Annotation missing for {img} in {cls} — but image was still removed.")

        print(f"   ✅ {cls}: removed {removed_count} image-annotation pairs (was {len(images)}, now {min_count})")

def main():
    parser = argparse.ArgumentParser(description="Undersample dataset with JSON annotations. Removes image + matching .json annotation.")
    parser.add_argument('root_dir', type=str, help="Root directory containing class folders with 'images/{train,test,val}/' and 'annotations/'")

    args = parser.parse_args()
    root_dir = args.root_dir

    if not os.path.exists(root_dir):
        print(f"❌ Root directory '{root_dir}' does not exist.")
        return

    splits = ['train', 'test', 'val']

    for split in splits:
        undersample_split_by_class(root_dir, split)

    print("\n🎉 Undersampling completed successfully!")

if __name__ == "__main__":
    main()