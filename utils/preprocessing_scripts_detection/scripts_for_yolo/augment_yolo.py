import os
import random
import argparse
import cv2
import numpy as np
from pathlib import Path

def load_yolo_label(label_path):
    """Load YOLO label file and return list of [class_id, x_center, y_center, width, height]"""
    if not os.path.exists(label_path):
        return []
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels

def save_yolo_label(label_path, labels):
    """Save list of labels to YOLO .txt file"""
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def augment_image_and_labels(image, labels, img_size=(640, 640), augment_prob=0.8):
    """
    Apply geometric and color augmentations to image and bounding boxes.
    Returns: augmented_image, augmented_labels
    """
    h, w = image.shape[:2]
    original_image = image.copy()
    original_labels = [label[:] for label in labels]  # Deep copy

    # --- COLOR AUGMENTATIONS ---
    if random.random() < augment_prob:
        # Brightness jitter (±20%)
        alpha_b = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha_b, beta=0)

    if random.random() < augment_prob:
        # Contrast jitter (±20%)
        alpha_c = random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=alpha_c, beta=0)

    if random.random() < augment_prob:
        # Saturation jitter (±20%) - convert to HSV, adjust S, then back to BGR
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.8, 1.2)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- GEOMETRIC AUGMENTATIONS ---

    # 1. Horizontal Flip (50% chance)
    if random.random() < augment_prob:
        image = cv2.flip(image, 1)
        for label in labels:
            label[1] = 1.0 - label[1]  # Flip x_center

    # 2. Rotation (±15 degrees)
    if random.random() < augment_prob:
        angle = random.uniform(-15, 15)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Rotate bounding boxes: transform centers
        for label in labels:
            x, y = label[1] * w, label[2] * h
            # Apply rotation matrix
            x_new = (x - center[0]) * np.cos(np.radians(angle)) - (y - center[1]) * np.sin(np.radians(angle)) + center[0]
            y_new = (x - center[0]) * np.sin(np.radians(angle)) + (y - center[1]) * np.cos(np.radians(angle)) + center[1]
            label[1] = x_new / w
            label[2] = y_new / h

    # 3. Scaling (0.8x to 1.2x)
    if random.random() < augment_prob:
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad or crop to original size
        pad_w = max(0, (w - new_w) // 2)
        pad_h = max(0, (h - new_h) // 2)
        padded = np.zeros((h, w, 3), dtype=np.uint8)
        padded[pad_h:min(pad_h+new_h, h), pad_w:min(pad_w+new_w, w)] = resized_img[:min(new_h, h), :min(new_w, w)]
        image = padded

        # Adjust bounding box coordinates for scaling and padding
        for label in labels:
            label[1] = (label[1] * w - pad_w) / w
            label[2] = (label[2] * h - pad_h) / h
            label[3] = label[3] * scale
            label[4] = label[4] * scale
            # Clamp to [0,1]
            label[1] = np.clip(label[1], 0, 1)
            label[2] = np.clip(label[2], 0, 1)
            label[3] = np.clip(label[3], 0, 1)
            label[4] = np.clip(label[4], 0, 1)

    # 4. Random Crop (20% chance)
    if random.random() < 0.2:
        crop_ratio = random.uniform(0.7, 0.95)
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        x_start = random.randint(0, w - crop_w)
        y_start = random.randint(0, h - crop_h)

        image = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Update labels: shift and rescale
        for label in labels:
            x_center = label[1] * w
            y_center = label[2] * h
            bw = label[3] * w
            bh = label[4] * h

            # Check if bbox center is inside crop
            if x_start <= x_center <= x_start + crop_w and y_start <= y_center <= y_start + crop_h:
                # Shift and normalize
                label[1] = (x_center - x_start) / w
                label[2] = (y_center - y_start) / h
                label[3] = bw / w
                label[4] = bh / h
            else:
                # Discard if center outside crop
                label[1] = label[2] = label[3] = label[4] = -1  # Mark for removal

        # Remove invalid labels
        labels = [l for l in labels if l[1] != -1]

    return image, labels


def main():
    parser = argparse.ArgumentParser(description="Augment YOLO dataset with geometric and color transformations.")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing images/ and labels/ subfolders')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for augmented data (default: data_dir_aug)')
    parser.add_argument('--num_aug_per_image', type=int, default=1, help='Number of augmented versions per image (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if args.output_dir is None:
        output_dir = data_dir.parent / (data_dir.name + "_aug")
    else:
        output_dir = Path(args.output_dir)

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Supported extensions
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # Traverse train/val/test subdirectories
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        src_img_subdir = images_dir / subset
        src_lbl_subdir = labels_dir / subset
        dst_img_subdir = output_images_dir / subset
        dst_lbl_subdir = output_labels_dir / subset

        dst_img_subdir.mkdir(parents=True, exist_ok=True)
        dst_lbl_subdir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {subset}...")

        # Get all image files
        img_files = [f for f in src_img_subdir.iterdir() if f.suffix.lower() in exts]
        total = len(img_files)

        for i, img_file in enumerate(img_files, 1):
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"Warning: Could not load {img_file}")
                continue

            # Load corresponding label
            label_file = src_lbl_subdir / (img_file.stem + ".txt")
            labels = load_yolo_label(label_file)

            # Generate N augmented versions
            for aug_idx in range(args.num_aug_per_image):
                # Copy image and labels for augmentation
                aug_image, aug_labels = augment_image_and_labels(image.copy(), labels.copy())

                # Save augmented image
                aug_img_name = f"{img_file.stem}_aug{aug_idx}{img_file.suffix}"
                aug_img_path = dst_img_subdir / aug_img_name
                cv2.imwrite(str(aug_img_path), aug_image)

                # Save augmented labels
                aug_lbl_name = f"{img_file.stem}_aug{aug_idx}.txt"
                aug_lbl_path = dst_lbl_subdir / aug_lbl_name
                save_yolo_label(aug_lbl_path, aug_labels)

            orig_img_path = dst_img_subdir / img_file.name
            orig_lbl_path = dst_lbl_subdir / label_file.name
            cv2.imwrite(str(orig_img_path), image)
            save_yolo_label(orig_lbl_path, labels)

            if i % 100 == 0:
                print(f"  Progress: {i}/{total}")

    print(f"\n✅ Augmentation completed! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()