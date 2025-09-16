from pathlib import Path
import argparse
import shutil
import json
import datetime
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("merge_classes")

def detect_structure(root: Path):
    """Detect dataset structure: 'unified_yolo', 'class_yolo', 'class_coco', or 'unknown'."""
    if (root / "images").exists() and (root / "labels").exists():
        return "unified_yolo"
    # look for class subdirs
    class_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not class_dirs:
        return "unknown"
    first = class_dirs[0]
    if (first / "images").exists() and (first / "labels").exists():
        return "class_yolo"
    if (first / "images").exists() and (first / "annotations").exists():
        return "class_coco"
    return "unknown"

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path, dry_run: bool):
    safe_mkdir(dst.parent)
    if dry_run:
        log.info(f"[DRY] copy {src} -> {dst}")
    else:
        shutil.copy2(src, dst)

def remap_yolo_label_lines(lines: List[str], new_class_idx: int) -> List[str]:
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        # expect class x y w h
        if len(parts) < 5:
            continue
        parts[0] = str(new_class_idx)
        out.append(" ".join(parts))
    return out

def convert_coco_json_to_yolo_labels(coco_json_path: Path, class_idx: int, out_labels_dir: Path, images_dir: Path, rename_prefix: str, dry_run: bool):
    """Convert one COCO json (assumed to contain annotations for a single class) into YOLO labels."""
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data.get("images", [])}
    anns_by_image = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)
    for img_id, img in images.items():
        file_name = img.get("file_name")
        if not file_name:
            continue
        img_w = img.get("width")
        img_h = img.get("height")
        if not img_w or not img_h:
            # try to infer from actual file
            img_path = images_dir / file_name
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                log.warning(f"Could not get size for {img_path}, skipping")
                continue
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue
        label_lines = []
        for ann in anns:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            x_c = x + w / 2.0
            y_c = y + h / 2.0
            x_cn = x_c / img_w
            y_cn = y_c / img_h
            w_n = w / img_w
            h_n = h / img_h
            # clamp
            x_cn = max(0.0, min(1.0, x_cn))
            y_cn = max(0.0, min(1.0, y_cn))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))
            label_lines.append(f"{class_idx} {x_cn:.6f} {y_cn:.6f} {w_n:.6f} {h_n:.6f}")
        if not label_lines:
            continue
        # determine output label file name (apply rename prefix if set)
        out_image_name = f"{rename_prefix}{file_name}" if rename_prefix else file_name
        stem = Path(out_image_name).stem
        out_label_path = out_labels_dir / f"{stem}.txt"
        safe_mkdir(out_label_path.parent)
        if dry_run:
            log.info(f"[DRY] write labels for {out_label_path} ({len(label_lines)} anns)")
        else:
            out_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

def merge_class_yolo(class_dir: Path, class_idx: int, out_images: Path, out_labels: Path, rename_prefix: str, dry_run: bool):
    imgs_root = class_dir / "images"
    labels_root = class_dir / "labels"
    if not imgs_root.exists():
        log.warning(f"No images/ in {class_dir}, skipping")
        return
    # handle splits if present
    splits = [p for p in imgs_root.iterdir() if p.is_dir()] or [imgs_root]
    for sp in splits:
        rel = sp.name if sp.is_dir() and sp.name in ("train", "val", "test") else ""
        dst_img_dir = (out_images / rel) if rel else out_images
        dst_lbl_dir = (out_labels / rel) if rel else out_labels
        for img_path in sp.glob("*.*"):
            if not img_path.is_file():
                continue
            target_name = f"{rename_prefix}{img_path.name}" if rename_prefix else img_path.name
            dst_img_path = dst_img_dir / target_name
            copy_file(img_path, dst_img_path, dry_run)
            # copy label (same stem .txt)
            lbl_src = labels_root / rel / f"{img_path.stem}.txt" if rel else labels_root / f"{img_path.stem}.txt"
            if lbl_src.exists():
                try:
                    lines = lbl_src.read_text(encoding="utf-8").splitlines()
                    new_lines = remap_yolo_label_lines(lines, class_idx)
                    dst_lbl_path = dst_lbl_dir / f"{Path(target_name).stem}.txt"
                    safe_mkdir(dst_lbl_path.parent)
                    if dry_run:
                        log.info(f"[DRY] write label {dst_lbl_path} ({len(new_lines)} lines)")
                    else:
                        dst_lbl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                except Exception as e:
                    log.warning(f"Failed to process label {lbl_src}: {e}")
            else:
                # label may not exist if dataset uses per-class jsons or missing labels
                log.debug(f"No label for {img_path.name} at expected {lbl_src}")

def merge_unified_yolo(root: Path, out_images: Path, out_labels: Path, dry_run: bool):
    imgs = root / "images"
    lbls = root / "labels"
    if not imgs.exists() or not lbls.exists():
        log.error("Unified YOLO layout requires images/ and labels/ at root")
        return
    # copy splits or all
    splits = [p for p in imgs.iterdir() if p.is_dir()] or [imgs]
    for sp in splits:
        rel = sp.name if sp.is_dir() and sp.name in ("train", "val", "test") else ""
        dst_img_dir = (out_images / rel) if rel else out_images
        dst_lbl_dir = (out_labels / rel) if rel else out_labels
        for img_path in sp.glob("*.*"):
            if not img_path.is_file():
                continue
            dst_img_path = dst_img_dir / img_path.name
            copy_file(img_path, dst_img_path, dry_run)
            lbl_src = (lbls / rel / f"{img_path.stem}.txt") if rel else (lbls / f"{img_path.stem}.txt")
            if lbl_src.exists():
                dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"
                copy_file(lbl_src, dst_lbl_path, dry_run)
            else:
                log.debug(f"Missing label for {img_path.name}")

def merge_class_coco(class_dir: Path, class_idx: int, out_images: Path, out_labels: Path, out_annotations_dir: Path, rename_prefix: str, dry_run: bool, convert_coco: bool):
    imgs_root = class_dir / "images"
    ann_root = class_dir / "annotations"
    if not imgs_root.exists():
        log.warning(f"No images/ in {class_dir}, skipping")
        return
    # copy image files
    for img_path in imgs_root.glob("*.*"):
        if not img_path.is_file():
            continue
        target_name = f"{rename_prefix}{img_path.name}" if rename_prefix else img_path.name
        dst_img_path = out_images / target_name
        copy_file(img_path, dst_img_path, dry_run)
    # handle annotations
    if not ann_root.exists():
        return
    safe_mkdir(out_annotations_dir)
    for ann in ann_root.glob("*.json"):
        if convert_coco:
            convert_coco_json_to_yolo_labels(ann, class_idx, out_labels, imgs_root, rename_prefix, dry_run)
        else:
            # copy json to annotations/classname__json
            dst = out_annotations_dir / f"{class_dir.name}__{ann.name}"
            copy_file(ann, dst, dry_run)

def main():
    parser = argparse.ArgumentParser(description="Merge class-based datasets into a unified YOLO/annotations structure.")
    parser.add_argument("--data-root", "-r", required=True, type=Path, help="Root dir containing either unified dataset or class subfolders")
    parser.add_argument("--output-root", "-o", type=Path, default=None, help="Output root (default: <data-root>/../merged_dataset)")
    parser.add_argument("--classes", nargs="+", default=None, help="Optional ordered list of class names (used to map class ids for per-class YOLO/COCO). If omitted, class folder ordering is used.")
    parser.add_argument("--dry-run", action="store_true", help="Don't copy/write files, only print actions")
    parser.add_argument("--rename", choices=["none","class_prefix"], default="class_prefix", help="How to rename files to avoid collisions. 'class_prefix' will prepend '<class>__' to filenames.")
    parser.add_argument("--convert-coco", action="store_true", help="When merging per-class COCO annotations, convert to YOLO .txt labels (requires Pillow).")
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    if not data_root.exists():
        log.error(f"data-root does not exist: {data_root}")
        return 2

    out_root = args.output_root.resolve() if args.output_root else (data_root.parent / "merged_dataset")
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_annotations = out_root / "annotations"

    safe_mkdir(out_root)
    safe_mkdir(out_images)
    safe_mkdir(out_labels)

    structure = detect_structure(data_root)
    log.info(f"Detected structure: {structure}")

    # determine classes list
    if args.classes:
        classes = args.classes
    else:
        # if class dirs exist, list them
        if structure in ("class_yolo", "class_coco"):
            classes = [p.name for p in sorted([d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")])]
        else:
            classes = []

    log.info(f"Classes: {classes or '(none)'}")

    rename_prefix_template = ""
    if args.rename == "class_prefix":
        # will set per-file when copying

        def prefix_for(class_name):
            return f"{class_name}__"
    else:
        def prefix_for(class_name):
            return ""

    if structure == "unified_yolo":
        log.info("Merging unified YOLO dataset (will copy files to output).")
        merge_unified_yolo(data_root, out_images, out_labels, args.dry_run)
    elif structure == "class_yolo":
        log.info("Merging per-class YOLO folders.")
        for idx, cls_name in enumerate(classes):
            class_dir = data_root / cls_name
            if not class_dir.exists():
                log.warning(f"Class dir missing: {class_dir}, skipping")
                continue
            merge_class_yolo(class_dir, idx, out_images, out_labels, prefix_for(cls_name), args.dry_run)
    elif structure == "class_coco":
        log.info("Merging per-class COCO folders.")
        for idx, cls_name in enumerate(classes):
            class_dir = data_root / cls_name
            if not class_dir.exists():
                log.warning(f"Class dir missing: {class_dir}, skipping")
                continue
            merge_class_coco(class_dir, idx, out_images, out_labels, out_annotations, prefix_for(cls_name), args.dry_run, args.convert_coco)
    else:
        log.error("Unknown dataset layout. Expected unified YOLO (images/ + labels/) or class-based folders with images/ + labels/ or images/ + annotations/.")
        return 3

    log.info(f"Done. Output at: {out_root}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())