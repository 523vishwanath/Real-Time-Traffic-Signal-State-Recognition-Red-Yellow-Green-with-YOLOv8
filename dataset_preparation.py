"""
dataset_preparation.py
======================
Bounding box–guided cropping strategy for small traffic light detection.

Pipeline Overview:
  1. Reads YOLO-format annotations from the original dataset.
  2. Remaps class labels (merges the mislabeled 'off' class → 'red').
  3. Generates context-aware crops centered on each group of bounding boxes.
  4. Copies original full images alongside crops for scene diversity.
  5. Writes re-labeled YOLO annotation files for each output image.

Class Remapping (Original → New):
  0: Green   → 0: Green
  1: off     → 1: red      (mislabeled; merged into red)
  2: red     → 1: red
  3: wait_on → 2: wait_on
  4: yellow  → 3: yellow

Usage:
    python src/dataset_preparation.py \
        --dataset_root /path/to/Small_Traffic_Light.v1i.yolov11 \
        --output_root  /path/to/traffic_light_crops_dataset \
        --crop_expand  9 \
        --min_size     10
"""

import os
import cv2
import shutil
import argparse
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CLASS_REMAP = {
    0: 0,   # Green   → Green
    1: 1,   # off     → red   (mislabeled samples)
    2: 1,   # red     → red
    3: 2,   # wait_on → wait_on
    4: 3,   # yellow  → yellow
}

CLASS_NAMES = {0: "Green", 1: "red", 2: "wait_on", 3: "yellow"}

# ──────────────────────────────────────────────────────────────────────────────
# Coordinate Utilities
# ──────────────────────────────────────────────────────────────────────────────
def yolo_to_xyxy(box, img_w, img_h):
    """Convert YOLO normalized [xc, yc, bw, bh] → pixel [x1, y1, x2, y2]."""
    xc, yc, bw, bh = box
    x1 = int((xc - bw / 2) * img_w)
    y1 = int((yc - bh / 2) * img_h)
    x2 = int((xc + bw / 2) * img_w)
    y2 = int((yc + bh / 2) * img_h)
    return x1, y1, x2, y2


def xyxy_to_yolo(box, img_w, img_h):
    """Convert pixel [x1, y1, x2, y2] → YOLO normalized [xc, yc, bw, bh]."""
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [xc, yc, bw, bh]


def clip(val, low, high):
    return max(low, min(val, high))


# ──────────────────────────────────────────────────────────────────────────────
# Label I/O
# ──────────────────────────────────────────────────────────────────────────────
def read_yolo_labels(label_path):
    """
    Read a YOLO label file and return a list of (new_class_id, [xc,yc,bw,bh]).
    Applies CLASS_REMAP to every annotation.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id_original = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:])

            if cls_id_original not in CLASS_REMAP:
                print(f"  [WARN] Unknown class {cls_id_original} in {label_path} — skipped.")
                continue

            cls_id_new = CLASS_REMAP[cls_id_original]
            boxes.append((cls_id_new, [xc, yc, bw, bh]))
    return boxes


def write_yolo_labels(label_path, boxes):
    """Write a list of (class_id, [xc,yc,bw,bh]) to a YOLO label file."""
    with open(label_path, "w") as f:
        for cls_id, (xc, yc, bw, bh) in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Core Processing
# ──────────────────────────────────────────────────────────────────────────────
def process_split(split, dataset_root, output_root, crop_expand, min_size):
    """
    Process a single dataset split ('train', 'valid', or 'test').

    For every image:
      - Copy the original full image + re-labeled annotation.
      - Generate bounding box–guided crops and their re-labeled annotations.
        Crops that already contain multiple boxes share a single label file,
        preventing redundant duplicates.
    """
    img_dir = os.path.join(dataset_root, split, "images")
    lbl_dir = os.path.join(dataset_root, split, "labels")

    out_img_dir = os.path.join(output_root, split, "images")
    out_lbl_dir = os.path.join(output_root, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]

    crop_count = 0
    for img_file in tqdm(image_files, desc=f"Processing [{split}]"):
        stem = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [WARN] Could not read {img_path} — skipped.")
            continue

        h, w = img.shape[:2]

        # ── 1. Copy original image with remapped labels ──────────────────────
        out_img_path = os.path.join(out_img_dir, img_file)
        out_lbl_path = os.path.join(out_lbl_dir, stem + ".txt")
        shutil.copy(img_path, out_img_path)

        if os.path.exists(lbl_path):
            boxes = read_yolo_labels(lbl_path)
        else:
            boxes = []

        write_yolo_labels(out_lbl_path, boxes)

        # ── 2. Generate bounding box–guided crops ────────────────────────────
        if not boxes:
            continue

        visited = set()   # track which boxes are already inside a generated crop

        for i, (cls_i, box_i) in enumerate(boxes):
            if i in visited:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(box_i, w, h)

            # Expand crop region by CROP_EXPAND × object dimensions
            obj_w = x2 - x1
            obj_h = y2 - y1
            if obj_w < min_size or obj_h < min_size:
                continue

            cx1 = clip(x1 - obj_w * crop_expand, 0, w)
            cy1 = clip(y1 - obj_h * crop_expand, 0, h)
            cx2 = clip(x2 + obj_w * crop_expand, 0, w)
            cy2 = clip(y2 + obj_h * crop_expand, 0, h)

            crop = img[int(cy1):int(cy2), int(cx1):int(cx2)]
            ch, cw = crop.shape[:2]
            if cw < min_size or ch < min_size:
                continue

            # Collect all boxes that fall inside this crop
            crop_boxes = []
            for j, (cls_j, box_j) in enumerate(boxes):
                bx1, by1, bx2, by2 = yolo_to_xyxy(box_j, w, h)
                # Check if the box centre is within the crop
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    # Re-compute coordinates relative to crop
                    nx1 = clip(bx1 - cx1, 0, cw)
                    ny1 = clip(by1 - cy1, 0, ch)
                    nx2 = clip(bx2 - cx1, 0, cw)
                    ny2 = clip(by2 - cy1, 0, ch)
                    yolo_box = xyxy_to_yolo([nx1, ny1, nx2, ny2], cw, ch)
                    crop_boxes.append((cls_j, yolo_box))
                    visited.add(j)

            if not crop_boxes:
                continue

            crop_name = f"{stem}_crop{crop_count:04d}{os.path.splitext(img_file)[1]}"
            cv2.imwrite(os.path.join(out_img_dir, crop_name), crop)
            write_yolo_labels(
                os.path.join(out_lbl_dir, os.path.splitext(crop_name)[0] + ".txt"),
                crop_boxes,
            )
            crop_count += 1

    print(f"  → [{split}] {crop_count} crops generated.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Traffic Light Dataset Preparation")
    parser.add_argument("--dataset_root", required=True,
                        help="Path to the original YOLO dataset root.")
    parser.add_argument("--output_root", required=True,
                        help="Path where the processed dataset will be saved.")
    parser.add_argument("--crop_expand", type=int, default=9,
                        help="Multiplier for expanding crops around bounding boxes (default: 9).")
    parser.add_argument("--min_size", type=int, default=10,
                        help="Minimum object pixel size to include (default: 10).")
    args = parser.parse_args()

    for split in ["train", "valid", "test"]:
        split_path = os.path.join(args.dataset_root, split, "images")
        if os.path.isdir(split_path):
            process_split(split, args.dataset_root, args.output_root,
                          args.crop_expand, args.min_size)
        else:
            print(f"  [INFO] Split '{split}' not found — skipping.")

    print("\n✅ Dataset preparation complete.")
    print(f"   Output saved to: {args.output_root}")


if __name__ == "__main__":
    main()
