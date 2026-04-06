"""
inference.py
============
Run inference using a trained YOLOv8 or TensorRT-optimized traffic light
detection model on images, video files, or a webcam stream.

Supports both:
  - .pt  weights (standard PyTorch)
  - .engine weights (TensorRT, ~4.9 ms/image on A100)

Usage examples:
    # Video inference with TensorRT
    python src/inference.py \
        --weights runs/detect/traffic_light_yolov8l/weights/best.engine \
        --source  assets/demo.mp4 \
        --conf    0.70 \
        --save

    # Webcam (real-time)
    python src/inference.py \
        --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
        --source  0 \
        --conf    0.60

    # Single image
    python src/inference.py \
        --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
        --source  assets/test_image.jpg \
        --conf    0.70 \
        --save
"""

import argparse
from ultralytics import YOLO


# Class name → colour mapping for on-screen display (BGR)
CLASS_COLORS = {
    "Green"   : (0,   200,  0),
    "red"     : (0,   0,   220),
    "wait_on" : (0,   165, 255),
    "yellow"  : (0,   220, 220),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Light Detection — Inference"
    )
    parser.add_argument("--weights", required=True,
                        help="Path to .pt or .engine weights file.")
    parser.add_argument("--source",  required=True,
                        help="Input source: image path, video path, or webcam index (e.g. 0).")
    parser.add_argument("--conf",    type=float, default=0.70,
                        help="Confidence threshold (default: 0.70).")
    parser.add_argument("--iou",     type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45).")
    parser.add_argument("--imgsz",   type=int,   default=960,
                        help="Inference image size (default: 960).")
    parser.add_argument("--save",    action="store_true",
                        help="Save output images/video with predictions drawn.")
    parser.add_argument("--name",    default="out_TrafficLightDemo",
                        help="Output folder name inside runs/detect/ (default: out_TrafficLightDemo).")
    parser.add_argument("--show",    action="store_true",
                        help="Display predictions in a live window (requires display).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Traffic Light Detection — Inference")
    print("=" * 60)
    print(f"  Weights : {args.weights}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print(f"  IoU     : {args.iou}")
    print(f"  Img size: {args.imgsz}px")
    print("=" * 60)

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        name=args.name,
    )

    print(f"\n✅ Inference complete.")
    if args.save:
        print(f"   Results saved to: runs/detect/{args.name}/")

    return results


if __name__ == "__main__":
    main()
