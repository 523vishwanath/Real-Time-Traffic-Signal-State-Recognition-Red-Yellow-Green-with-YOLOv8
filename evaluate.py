"""
evaluate.py
===========
Evaluate a trained YOLOv8 traffic light detection model.

Prints a full metrics report including:
  - Overall mAP50, mAP50-95, mAP75, Precision, Recall
  - Per-class mAP50-95 breakdown

Usage:
    python src/evaluate.py \
        --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
        --data    config/traffic_light_detection.yaml \
        --imgsz   960 \
        --conf    0.25 \
        --iou     0.50
"""

import argparse
from ultralytics import YOLO

CLASS_NAMES = ["Green", "red", "wait_on", "yellow"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Traffic Light Detection Model")
    parser.add_argument("--weights", required=True,
                        help="Path to trained .pt weights file.")
    parser.add_argument("--data",    default="config/traffic_light_detection.yaml",
                        help="Path to the dataset YAML config.")
    parser.add_argument("--imgsz",   type=int,   default=960,
                        help="Inference image size (must match training size).")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="IoU threshold for NMS (default: 0.50).")
    parser.add_argument("--save",    action="store_true",
                        help="Save validation images with predictions.")
    return parser.parse_args()


def print_metrics(metrics):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP75      : {metrics.box.map75:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print("-" * 60)
    print("  Per-Class mAP50-95:")
    for i, ap in enumerate(metrics.box.maps):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        bar = "█" * int(ap * 40)
        print(f"    {name:<12} {ap:.4f}  {bar}")
    print("=" * 60)


def main():
    args = parse_args()

    print(f"\n Loading model: {args.weights}")
    model = YOLO(args.weights)

    print(f" Running validation on: {args.data}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
    )

    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
