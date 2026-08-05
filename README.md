<img width="3816" height="1928" alt="Fig  1  Overall workflow of the proposed DL-YOLO framework" src="https://github.com/user-attachments/assets/8e946b65-2f2f-485d-a5fb-6c1dbe3826bd" /># Detection_Localization_YOLO

**Official implementation of "A monocular vision-based method using DL-YOLO for fire detection and localization in road tunnels"**

---

## 📋 Overview

DL-YOLO is an enhanced YOLOv8n framework for fire detection and monocular geometric localization in tunnels, featuring SPD-Conv, GAM, P2 head, and Ghost modules.
![Uploading Fig. 1. Overall workflow of the proposed DL-YOLO framework.png…]()

---

## 🚀 Quick Start

```bash
git clone https://github.com/Deepstrong-cn/Detection_Localization_YOLO.git
cd Detection_Localization_YOLO
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Inference
```bash
python predict.py
```

### Example Usage
For users who want to verify the localization method, we provide complete examples with pre-configured camera parameters. These examples include all necessary camera calibration files and sample images to reproduce the results reported in the paper.

### Flame localization
```bash
cd examples/localization/flame
python calculate_flame_position.py
```

### Smoke front localization
```bash
cd examples/localization/flame
python calculate_smoke_front_position.py
```
---

## 📊 Performance

| Detection | Value | Localization | Error |
|-----------|-------|--------------|-------|
| mAP@0.5 | **96.9%** | Flame (Longitudinal) | 1.00 m (1.80%) |
| Recall | **93.6%** | Flame (Lateral) | 0.12 m (2.7%) |
| Precision | **93.9%** | Smoke Front | 0.51 m (1.55%) |

---

## 📁 Structure

```
Detection_Localization_YOLO/
│
├── train.py                          # Training script
├── predict.py                        # Inference script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── LICENSE                           # AGPL-3.0
├── .gitignore                        # Git ignore file
│
├── DATA/                             # Dataset configuration and splits
│   ├── yolov8.yaml                   # Dataset config
│   ├── annotation_guidelines.md      # Annotation guidelines
│   └── dataset/
│       ├── images/                   # Image files (not uploaded)
│       ├── labels/                   # Label files (not uploaded)
│       ├── train.txt                 # Training set image list
│       ├── val.txt                   # Validation set image list
│       ├── test.txt                  # Test set image list
│       ├── classes.txt               # Class names
│       └── generate_names.py         # Script to generate image lists
│
├── examples/                         # Standalone examples
│   └── localization/
│       ├── flame/
│       │   ├── calculate_flame_position.py
│       │   └── README.md
│       └── smoke/
│           ├── calculate_smoke_front_position.py
│           ├── camera_matrix.txt
│           ├── dist_coeffs.txt
│           ├── weights.pt
│           └── README.md
│   └── detection/
│       ├── Web-sourced/ # 5 web-sourced test images
│       └── Rail_robot_perspective/ # 5 rail-robot perspective test images
├── ultralytics/                      # Modified YOLO source (SPD-Conv, GAM, etc.)
│   └── cfg/
│       └── models/
│           └── v8/
│               └── yolov8-p2-spd-GAM-ghost.yaml
│
├── weights/                          # Pretrained weights
│   └── yolov8n.pt
│
└── runs/ # Training and inference outputs
```
---

## 📄 License

AGPL-3.0 License. This project builds upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

## Note: 
1. The full detection dataset is not included due to institutional restrictions. 2. During revision, the ground truth measurements for flame localization (Section 4.4.2) were found to be inaccurate, which we have clarified in our response to the reviewers. To further validate our method, we conducted extended experiments with 16 test points (A–P) covering 27–100 m and 0–6.6 m lateral offset (Section 4.4.4).
