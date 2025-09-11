import os
import shutil
import random

def split_dataset(input_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    # Create output dirs
    for split in ["train", "val", "test"]:
        split_path = os.path.join(output_folder, split)
        os.makedirs(split_path, exist_ok=True)

    # Loop through classes
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        print(f"📂 {class_name}: {n_train} train, {n_val} val, {n_test} test")

        # Copy files
        for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_class_path = os.path.join(output_folder, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)
            for img in split_imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(split_class_path, img))


# Example usage
input_dir = "../data_augmented"        # original dataset folder (with class subfolders)
output_dir = "../data_split_augmented"     # new folder with train/val/test
split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
