"""
train.py
========
YOLOv8l training script for real-time traffic signal state recognition.

Model   : YOLOv8l (large) — best balance of accuracy and inference speed.
Dataset : Bounding box–guided crops + original full images (see dataset_preparation.py).
Hardware: Tested on NVIDIA A100 (Google Colab); any CUDA-capable GPU works.

Key Hyperparameters (tuned for small-object detection):
  - imgsz  : 960   — higher resolution captures small/distant traffic lights.
  - batch  : 16    — adjust based on available VRAM.
  - hsv_h  : 0.01  — subtle hue shift to handle varying light conditions.
  - hsv_s  : 0.70  — saturation augmentation for overcast/night variation.
  - hsv_v  : 0.40  — brightness augmentation for glare and shadow.

Usage:
    python src/train.py \
        --data    config/traffic_light_detection.yaml \
        --epochs  100 \
        --imgsz   960 \
        --batch   16 \
        --project runs/detect \
        --name    traffic_light_yolov8l
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8l for Traffic Light Detection")
    parser.add_argument("--data",    default="config/traffic_light_detection.yaml",
                        help="Path to the dataset YAML config file.")
    parser.add_argument("--model",   default="yolov8l.pt",
                        help="YOLOv8 model variant to use (default: yolov8l.pt).")
    parser.add_argument("--epochs",  type=int,   default=100,
                        help="Number of training epochs (default: 100).")
    parser.add_argument("--imgsz",   type=int,   default=960,
                        help="Input image size in pixels (default: 960).")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (default: 16). Reduce if OOM.")
    parser.add_argument("--hsv_h",   type=float, default=0.01,
                        help="HSV hue augmentation (default: 0.01).")
    parser.add_argument("--hsv_s",   type=float, default=0.70,
                        help="HSV saturation augmentation (default: 0.70).")
    parser.add_argument("--hsv_v",   type=float, default=0.40,
                        help="HSV value/brightness augmentation (default: 0.40).")
    parser.add_argument("--project", default="runs/detect",
                        help="Output directory for training runs.")
    parser.add_argument("--name",    default="traffic_light_yolov8l",
                        help="Name for this training run.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Traffic Light Detection — YOLOv8 Training")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Data    : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Img size: {args.imgsz}px")
    print(f"  Batch   : {args.batch}")
    print(f"  Project : {args.project}/{args.name}")
    print("=" * 60)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        project=args.project,
        name=args.name,
    )

    best_weights = f"{args.project}/{args.name}/weights/best.pt"
    print("\n✅ Training complete.")
    print(f"   Best weights saved to: {best_weights}")
    return results


if __name__ == "__main__":
    main()
