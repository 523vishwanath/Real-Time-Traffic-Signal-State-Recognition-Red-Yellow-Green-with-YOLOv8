# 🚦 Real-Time Traffic Signal State Recognition with YOLOv8

<p align="center">
  <img src="assets/demo.gif" alt="Traffic Light Detection Demo" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-YOLOv8l-blue?logo=pytorch" />
  <img src="https://img.shields.io/badge/Framework-Ultralytics-orange" />
  <img src="https://img.shields.io/badge/Runtime-TensorRT-green?logo=nvidia" />
  <img src="https://img.shields.io/badge/Speed-~4.9ms%2Fimage-brightgreen" />
  <img src="https://img.shields.io/badge/mAP50-0.963-blue" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Strategy & Methodology](#-strategy--methodology)
  - [Bounding Box–Guided Cropping](#1-bounding-boxguided-cropping)
  - [Retaining Original Full Images](#2-retaining-original-full-images)
  - [Class Remapping & Label Correction](#3-class-remapping--label-correction)
- [Performance Before & After](#-performance-before--after)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Prepare Dataset](#1-prepare-dataset)
  - [2. Train](#2-train)
  - [3. Evaluate](#3-evaluate)
  - [4. Export to TensorRT](#4-export-to-tensorrt)
  - [5. Run Inference](#5-run-inference)
- [Notebook](#-notebook)
- [Demo & Writeup](#-demo--writeup)

---

## 🔍 Overview

This project implements a **robust, real-time traffic light detection and classification system** capable of recognising four signal states:

| Class | Label | Description |
|-------|-------|-------------|
| 🟢 | `Green` | Go signal — safe to proceed |
| 🔴 | `red` | Stop signal (includes merged `off` state) |
| 🟠 | `wait_on` | Amber / prepare to stop |
| 🟡 | `yellow` | Caution signal |

The system is built on **YOLOv8l** with a strong emphasis on **data-centric improvements** rather than purely architectural changes. The core challenge — detecting **small and distant traffic lights** in real-world driving footage — is addressed through a custom bounding box–guided cropping pipeline, label correction, and class consolidation.

After training for **100 epochs on an NVIDIA A100 GPU**, the model is exported to **TensorRT FP16** for near real-time deployment at approximately **4.9 ms per image**.

---

## 📊 Key Results

### Final Model Performance (after label fix & retraining)

| Metric | Value |
|--------|-------|
| Precision | **0.967** |
| Recall | **0.932** |
| mAP50 | **0.963** |
| mAP75 | **0.516** |
| mAP50-95 | **0.542** |
| Inference Speed (TensorRT FP16) | **~4.9 ms/image** |

### Per-Class mAP50-95

| Class | mAP50-95 |
|-------|----------|
| Green | 0.595 |
| red | 0.568 |
| wait_on | 0.503 |
| yellow | 0.502 |

> **Benchmark improvement:** mAP50-95 improved from **0.436 → 0.542** after merging the noisy `off` class into `red`.

---

## 📁 Project Structure

```
traffic-light-detection/
│
├── 📓 TrafficLightDetection.ipynb    # End-to-end Colab notebook
│
├── 📂 src/
│   ├── dataset_preparation.py        # BBox-guided cropping + label remapping
│   ├── train.py                      # YOLOv8 training script
│   ├── evaluate.py                   # Validation & metrics reporting
│   ├── export_tensorrt.py            # Export .pt → TensorRT .engine
│   └── inference.py                  # Run predictions on image/video/webcam
│
├── 📂 config/
│   └── traffic_light_detection.yaml  # Dataset paths & class names
│
├── 📂 assets/
│   └── demo.gif                      # Inference demo (add your own)
│
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

**Source:** [Small Traffic Light Dataset (Roboflow)](https://universe.roboflow.com/) — YOLO format  
**Google Drive (preprocessed):** `1h_joqblzQaPWe4GDTiqfZvLFQgXibj9g`

### Original Class Distribution (before fix)

| Class ID | Name | Count | Issue |
|----------|------|-------|-------|
| 0 | Green | ✅ Normal | 1,330 |
| 1 | **off** | ~7 instances | ❌ Mislabeled & severely underrepresented |
| 2 | red | ✅ Normal | 1,892 |
| 3 | wait_on | ✅ Normal | 832 |
| 4 | yellow | ✅ Normal | 113 |

### Processed Class Distribution (after fix)

| Class ID | Name | Note |
|----------|------|------|
| 0 | Green | Unchanged |
| 1 | **red** | Merged `off` → `red` (corrected mislabels) |
| 2 | wait_on | Renumbered from 3 |
| 3 | yellow | Renumbered from 4 |

### Lighting & Weather Conditions
The dataset covers a diverse range of real-world driving conditions across three environmental settings:

| Condition | Characteristics |
|-----------|-----------------|
| 0 | Daytime | Clear natural light, high contrast, well-lit traffic lights against bright sky backgrounds |
| 1 | Nighttime | Low ambient light, strong light blooming from signals, high false-positive risk from streetlights and headlights |
| 2 | Rainy | Reduced visibility, lens flare, signal reflections on wet road surfaces, colour distortion |


The processed dataset combines:
- **Original full images** — preserve scene context, reduce false positives.
- **Bounding box–guided crops** — boost effective resolution for small objects.

---

## 🧠 Strategy & Methodology

### 1. Bounding Box–Guided Cropping

Standard grid-tiling splits images into equal-sized patches regardless of object locations. Many patches contain no traffic lights, offering no signal for small-object learning.

This project takes a different approach: **crops are generated by expanding outward from each annotated bounding box**, using a configurable multiplier (`CROP_EXPAND=9`). This places the traffic light at the centre of the crop with sufficient surrounding context, effectively increasing its pixel resolution within the training distribution.

**Multi-object handling:** When several bounding boxes lie within the same expanded crop region, they are all included in a single label file. The pipeline tracks which boxes have already been covered to avoid generating redundant duplicate crops.

```
Original image (1920×1080)         Crop around small traffic light
┌─────────────────────────┐        ┌──────────────┐
│                         │        │   context    │
│   🚦 (tiny, 8×20px)    │  ───▶  │   🚦 larger  │
│                         │        │   context    │
└─────────────────────────┘        └──────────────┘
```

### 2. Retaining Original Full Images

Training exclusively on object-centred crops introduces a **sampling bias**: the model begins to expect a traffic light in every frame, leading to false positives in background regions.

To counteract this, **all original full-resolution images are retained** in the training set. This exposes the model to:
- Frames where traffic lights occupy only a tiny fraction of the image.
- Frames with no traffic lights at all (background-only scenes from original images with empty label files).

This dual strategy — crops for resolution, originals for context — directly mirrors the benefits of tiling (background exposure) without its drawbacks (many irrelevant patches).

### 3. Class Remapping & Label Correction

The dataset originally included an `off` class with only **7 annotated instances** after preprocessing. This created two compounding problems:

1. **Severe class imbalance** — insufficient training signal.
2. **Mislabeled data** — visual inspection confirmed these instances were actually red traffic lights.

Simply duplicating the 7 samples would only amplify the incorrect labels. Instead, the `off` class was **merged into `red`** with a deterministic class remap:

```python
CLASS_REMAP = {
    0: 0,   # Green   → Green
    1: 1,   # off     → red  (mislabeled → corrected)
    2: 1,   # red     → red
    3: 2,   # wait_on → wait_on
    4: 3,   # yellow  → yellow
}
```

This produced an immediate and measurable improvement in model performance.

---

## 📈 Performance Before & After

### Before label fix (5 classes, `off` included)

| Class | mAP50-95 |
|-------|----------|
| Green | 0.593 |
| red | 0.553 |
| **off** | **0.007** ← near-zero learning |
| wait_on | — |
| yellow | 0.537 |
| **Overall** | **0.436** |

### After label fix (4 classes, `off` merged)

| Class | mAP50-95 |
|-------|----------|
| Green | 0.595 |
| red | 0.568 |
| wait_on | 0.503 |
| yellow | 0.502 |
| **Overall** | **0.542** ✅ |

> Removing the noisy class freed model capacity for meaningful pattern learning, improving overall mAP50-95 by **+10.6 points** and all other per-class scores simultaneously.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/traffic-light-detection.git
cd traffic-light-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install TensorRT for GPU-accelerated inference
pip install tensorrt
```

> **Note:** TensorRT export and inference require an NVIDIA GPU with CUDA installed. The training and evaluation scripts work on any CUDA-capable GPU.

---

## 🚀 Usage

### 1. Prepare Dataset

Download the raw dataset from Google Drive and run the preprocessing pipeline:

```bash
# Download dataset (requires gdown)
python - <<'EOF'
import gdown
gdown.download("https://drive.google.com/uc?id=1h_joqblzQaPWe4GDTiqfZvLFQgXibj9g",
               "dataset.zip", quiet=False)
EOF

# Unzip
unzip dataset.zip -d data/raw

# Run bounding box–guided cropping + class remapping
python src/dataset_preparation.py \
    --dataset_root data/raw/Small_Traffic_Light.v1i.yolov11 \
    --output_root  data/processed \
    --crop_expand  9 \
    --min_size     10
```

Update `config/traffic_light_detection.yaml` to point `path:` at your `data/processed` directory.

---

### 2. Train

```bash
python src/train.py \
    --data    config/traffic_light_detection.yaml \
    --model   yolov8l.pt \
    --epochs  100 \
    --imgsz   960 \
    --batch   16 \
    --name    traffic_light_yolov8l
```

> **Hardware note:** Training was performed on an NVIDIA A100 (Google Colab). Reduce `--batch` or `--imgsz` for smaller GPUs.

---

### 3. Evaluate

```bash
python src/evaluate.py \
    --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
    --data    config/traffic_light_detection.yaml \
    --imgsz   960
```

Sample output:
```
============================================================
  Evaluation Results
============================================================
  Precision  : 0.9670
  Recall     : 0.9320
  mAP50      : 0.9630
  mAP75      : ...
  mAP50-95   : 0.5420
------------------------------------------------------------
  Per-Class mAP50-95:
    Green        0.5950  ████████████████████████
    red          0.5680  ██████████████████████
    wait_on      0.5030  ████████████████████
    yellow       0.5020  ████████████████████
============================================================
```

---

### 4. Export to TensorRT

```bash
python src/export_tensorrt.py \
    --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
    --imgsz   960 \
    --half         # FP16 precision (~4.9 ms/image on A100)
```

---

### 5. Run Inference

```bash
# Video — TensorRT engine
python src/inference.py \
    --weights runs/detect/traffic_light_yolov8l/weights/best.engine \
    --source  your_video.mp4 \
    --conf    0.70 \
    --save

# Real-time webcam
python src/inference.py \
    --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
    --source  0 \
    --conf    0.60 \
    --show

# Single image
python src/inference.py \
    --weights runs/detect/traffic_light_yolov8l/weights/best.pt \
    --source  assets/test_image.jpg \
    --conf    0.70 \
    --save
```

---

## 📓 Notebook

The complete end-to-end workflow is available as a Google Colab notebook:

**[`TrafficLightDetection.ipynb`](TrafficLightDetection.ipynb)**

The notebook covers every step in sequence:
1. Dataset download & extraction
2. Dependency installation
3. Bounding box–guided cropping & class remapping
4. Dataset visualisation with FiftyOne
5. YAML config generation
6. YOLOv8l training (100 epochs, imgsz=960)
7. Training curve visualisation
8. Model evaluation & metrics
9. Saving best weights to Google Drive
10. TensorRT export
11. Inference on video

---

## 🎬 Demo & Writeup

| Resource | Link |
|----------|------|
| 🎥 Inference Demo Video | [YouTube — youtu.be/Iu9jzexylSs](https://youtu.be/Iu9jzexylSs) |
| 📄 Full Strategy Writeup | [Google Docs](https://docs.google.com/document/d/1BYa89EUTyGwMNhOo5IqY0mL8Tp3aesl__1rMTd0Ivbk/edit?usp=sharing) |

---

## 🔑 Key Takeaways

> *"Data quality beats data quantity."*

- **Bounding box–guided cropping** significantly improves small-object detection without changing the model architecture.
- **Retaining full-scene images** alongside crops is essential for generalization and false-positive suppression.
- **Correcting mislabeled data** (merging `off` → `red`) had a larger impact on mAP than any augmentation or architectural tweak.
- **TensorRT FP16** brings inference to ~4.9 ms/image — suitable for real-time automotive deployment.

---

## 📜 License

This project is released under the [MIT License](LICENSE).  
Dataset sourced from [Roboflow Universe](https://universe.roboflow.com/) under its respective license.
