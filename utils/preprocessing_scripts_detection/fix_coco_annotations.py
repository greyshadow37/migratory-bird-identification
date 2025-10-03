# tools/fix_coco.py
import json
import argparse
import shutil
from pathlib import Path

CATEGORY_METADATA = {
    "asian-green-bee-eater": {"id": 0, "name": "Asian Green Bee-Eater", "supercategory": "bird"},
    "cattle-egret": {"id": 1, "name": "Cattle Egret", "supercategory": "bird"},
    "gray-wagtail": {"id": 2, "name": "Gray Wagtail", "supercategory": "bird"},
    "indian-pitta": {"id": 3, "name": "Indian Pitta", "supercategory": "bird"},
    "ruddy-shelduck": {"id": 4, "name": "Ruddy Shelduck", "supercategory": "bird"}
}

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_split_filenames(images_dir):
    splits = {}
    for split in ("train", "test", "val"):
        split_path = images_dir / split
        if not split_path.exists():
            splits[split] = set()
            continue
        splits[split] = {f.name for f in split_path.iterdir() if f.is_file()}
    return splits

def ensure_output_dirs(output_dir, images_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test", "val"):
        src = images_dir / split
        dst = output_dir / split
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def build_filename_split_lookup(split_filenames):
    exact_lookup = {}
    alternate_lookup = {}

    for split, names in split_filenames.items():
        for name in names:
            if name in exact_lookup and exact_lookup[name][0] != split:
                raise ValueError(f"Image {name} appears in multiple splits: {exact_lookup[name][0]} and {split}")
            exact_lookup[name] = (split, name)

            base_name = Path(name).name
            stem = Path(name).stem
            suffix = name.split("__", 1)[1] if "__" in name else None

            alt_keys = {
                base_name,
                base_name.lower(),
                stem,
                stem.lower(),
            }

            if suffix:
                alt_keys.add(suffix)
                alt_keys.add(suffix.lower())
                suffix_stem = Path(suffix).stem
                alt_keys.add(suffix_stem)
                alt_keys.add(suffix_stem.lower())
                alt_keys.add(f"{name.split('__', 1)[0]}__{suffix}")
                alt_keys.add(f"{name.split('__', 1)[0]}__{suffix}".lower())

            for key in alt_keys:
                alternate_lookup.setdefault(key, set()).add((split, name))

    alternate_lookup = {key: list(values) for key, values in alternate_lookup.items()}
    return exact_lookup, alternate_lookup


def resolve_image_filename(class_name, basename, exact_lookup, alternate_lookup):
    sanitized_variants = [
        class_name,
        class_name.replace("-", "_"),
        class_name.replace("-", ""),
    ]

    candidate_keys = [
        basename,
        basename.lower(),
    ]

    for prefix in sanitized_variants:
        candidate_keys.append(f"{prefix}__{basename}")
        candidate_keys.append(f"{prefix}__{basename}".lower())

    for key in candidate_keys:
        if key in exact_lookup:
            return exact_lookup[key]

    alternate_candidates = []
    for key in candidate_keys + [Path(basename).stem, Path(basename).stem.lower()]:
        alternate_candidates.extend(alternate_lookup.get(key, []))

    if not alternate_candidates:
        return None

    unique_candidates = {}
    for split, name in alternate_candidates:
        unique_candidates[name] = (split, name)
    candidates = list(unique_candidates.values())

    def matches_prefix(name: str) -> bool:
        lowered = name.lower()
        return any(lowered.startswith(f"{variant.lower()}__") for variant in sanitized_variants)

    preferred = [candidate for candidate in candidates if matches_prefix(candidate[1])]

    if preferred:
        splits = {split for split, _ in preferred}
        if len(splits) > 1:
            raise ValueError(f"Image '{basename}' for class '{class_name}' appears in multiple splits: {splits}")
        return preferred[0]

    splits = {split for split, _ in candidates}
    if len(splits) > 1:
        raise ValueError(f"Ambiguous image filename '{basename}' for class '{class_name}' across splits: {splits}")

    return candidates[0]

def normalize_per_class_json(data, class_name):
    # Expect COCO-like dict with images / annotations / categories
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    # Ignore categories from source; we'll regenerate.
    return images, annotations

def main():
    ap = argparse.ArgumentParser(description="Fix and consolidate per-class COCO-like JSONs into split-based COCO datasets.")
    ap.add_argument("--images-dir", required=True, help="Path to images directory containing train/, test/, val/")
    ap.add_argument("--ann-dir", required=True, help="Path to annotations directory with per-class subfolders")
    ap.add_argument("--output-dir", required=True, help="Output directory (e.g. data-coco-clean)")
    ap.add_argument("--info", default="{}",
                    help="Optional JSON string for 'info' field")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    ann_dir = Path(args.ann_dir)
    output_dir = Path(args.output_dir)

    try:
        info_field = json.loads(args.info)
    except json.JSONDecodeError:
        info_field = {}

    split_filenames = collect_split_filenames(images_dir)

    # Prepare aggregation structures
    split_data = {
        "train": {"images": [], "annotations": []},
        "test": {"images": [], "annotations": []},
        "val": {"images": [], "annotations": []}
    }

    categories = [meta for meta in sorted(CATEGORY_METADATA.values(), key=lambda m: m["id"])]
    next_image_id = 1
    next_annotation_id = 1

    exact_filename_lookup, alternate_filename_lookup = build_filename_split_lookup(split_filenames)
    image_lookup = {}  # filename -> {"id": int, "split": str}

    aggregated_info = info_field if info_field else None
    aggregated_licenses = []
    seen_license_keys = set()

    missing_images = []
    orphan_annotations = 0
    duplicate_images = 0

    # Map (original image unique key) -> new id if needed (not strictly necessary across per-class since we reassign)
    for class_dir in sorted(ann_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CATEGORY_METADATA:
            continue
        json_file = class_dir / "instances_default.json"
        if not json_file.exists():
            continue
        try:
            data = load_json(json_file)
        except Exception as e:
            print(f"Skipping {json_file}: {e}")
            continue

        if aggregated_info is None or not aggregated_info:
            aggregated_info = data.get("info") or {}

        for lic in data.get("licenses", []):
            lic_key = (lic.get("id"), lic.get("name"), lic.get("url"))
            if lic_key not in seen_license_keys:
                seen_license_keys.add(lic_key)
                aggregated_licenses.append(lic)

        images, annotations = normalize_per_class_json(data, class_name)
        # Build lookup old_id -> original filename
        id_to_original_name = {}
        id_to_resolved_name = {}
        for img in images:
            fn = img.get("file_name") or img.get("filename")
            if not fn:
                continue
            # Normalize to just basename to match actual file presence
            basename = Path(fn).name
            id_to_original_name[img.get("id")] = basename

        # Process annotations referencing existing files & matching splits
        for img in images:
            orig_id = img.get("id")
            basename = id_to_original_name.get(orig_id)
            if not basename:
                continue
            resolved = resolve_image_filename(class_name, basename, exact_filename_lookup, alternate_filename_lookup)
            if resolved is None:
                missing_images.append((class_name, basename))
                continue

            target_split, filename = resolved

            id_to_resolved_name[orig_id] = filename

            if filename in image_lookup:
                duplicate_images += 1
                img_new_id = image_lookup[filename]["id"]
            else:
                new_image = {
                    "id": next_image_id,
                    "file_name": f"{target_split}/{filename}"
                }
                for opt in ("width", "height", "license", "coco_url", "flickr_url", "date_captured"):
                    if opt in img:
                        new_image[opt] = img[opt]
                split_data[target_split]["images"].append(new_image)
                image_lookup[filename] = {"id": next_image_id, "split": target_split}
                img_new_id = next_image_id
                next_image_id += 1

        # Now process annotations
        for ann in annotations:
            old_image_id = ann.get("image_id")
            filename = id_to_resolved_name.get(old_image_id)
            if filename is None:
                orphan_annotations += 1
                continue
            mapped = image_lookup.get(filename)
            if mapped is None:
                orphan_annotations += 1
                continue

            target_split = mapped["split"]

            new_ann = dict(ann)  # shallow copy
            new_ann["id"] = next_annotation_id
            new_ann["image_id"] = mapped["id"]
            new_ann["category_id"] = CATEGORY_METADATA[class_name]["id"]
            # Remove segmentation/area/bbox validation if malformed
            if "categoryId" in new_ann:
                del new_ann["categoryId"]
            split_data[target_split]["annotations"].append(new_ann)
            next_annotation_id += 1

    # Write outputs
    ensure_output_dirs(output_dir, images_dir)
    for split in ("train", "test", "val"):
        out_json = {
            "info": aggregated_info or info_field or {},
            "licenses": aggregated_licenses,
            "images": split_data[split]["images"],
            "annotations": split_data[split]["annotations"],
            "categories": categories
        }
        with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Wrote {split} dataset: {len(out_json['images'])} images, {len(out_json['annotations'])} annotations")

    if missing_images:
        missing_details = {f"{cls}:{name}" for cls, name in missing_images}
        print(f"Skipped {len(missing_details)} images missing from images directory")
    if orphan_annotations:
        print(f"Skipped {orphan_annotations} annotations referencing missing images")
    if duplicate_images:
        print(f"Detected {duplicate_images} duplicate image references across class JSON files")

if __name__ == "__main__":
    main()