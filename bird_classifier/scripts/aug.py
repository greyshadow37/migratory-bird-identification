import os
from PIL import Image

def augment_images(input_folder, output_folder):
    # Create output folder if it doesn’t exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through class subfolders (if dataset is organized by classes)
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        # Make subfolder in output
        output_class_path = os.path.join(output_folder, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = Image.open(img_path).convert("RGB")

                base_name, ext = os.path.splitext(img_name)

                # Save original
                img.save(os.path.join(output_class_path, f"{base_name}_orig{ext}"))

                # Horizontal Flip
                img.transpose(Image.FLIP_LEFT_RIGHT).save(
                    os.path.join(output_class_path, f"{base_name}_hflip{ext}")
                )

                # Vertical Flip
                img.transpose(Image.FLIP_TOP_BOTTOM).save(
                    os.path.join(output_class_path, f"{base_name}_vflip{ext}")
                )

                # Rotate 90° clockwise
                img.rotate(-90, expand=True).save(
                    os.path.join(output_class_path, f"{base_name}_rot90{ext}")
                )

                # Rotate 180°
                img.rotate(180, expand=True).save(
                    os.path.join(output_class_path, f"{base_name}_rot180{ext}")
                )

            except Exception as e:
                print(f"❌ Error with {img_path}: {e}")


# Example usage
input_dir = "../data_split/train"          # your dataset folder
output_dir = "../data_augmented"     # new augmented folder
augment_images(input_dir, output_dir)
