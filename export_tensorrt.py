"""
export_tensorrt.py
==================
Export a trained YOLOv8 .pt model to TensorRT .engine format for
near real-time inference (~4.9 ms/image on A100).

Requirements:
  - NVIDIA GPU with CUDA installed
  - TensorRT installed (pip install tensorrt)
  - Ultralytics ≥ 8.0

FP16 half-precision is enabled by default for maximum throughput.

Usage:
    python src/export_tensorrt.py \
        --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
        --imgsz   960 \
        --half

The exported .engine file will be saved alongside the .pt file.
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to TensorRT .engine format"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to trained .pt weights file.")
    parser.add_argument("--imgsz",   type=int, default=960,
                        help="Export image size — must match training imgsz (default: 960).")
    parser.add_argument("--half",    action="store_true", default=True,
                        help="Enable FP16 half-precision (default: True).")
    parser.add_argument("--batch",   type=int, default=1,
                        help="Batch size for TensorRT engine (default: 1).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLOv8 → TensorRT Export")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Img size: {args.imgsz}px")
    print(f"  FP16    : {args.half}")
    print(f"  Batch   : {args.batch}")
    print("=" * 60)

    model = YOLO(args.weights)

    model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        batch=args.batch,
    )

    engine_path = args.weights.replace(".pt", ".engine")
    if os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 ** 2)
        print(f"\n✅ Export complete.")
        print(f"   Engine saved to : {engine_path}")
        print(f"   Engine size     : {size_mb:.1f} MB")
    else:
        print("\n⚠️  Export may have failed — .engine file not found.")
        print("   Ensure TensorRT is installed and a CUDA GPU is available.")


if __name__ == "__main__":
    main()
