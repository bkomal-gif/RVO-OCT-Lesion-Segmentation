# 🔬 RVO-OCT-Lesion-Segmentation

![OCT Segmentation Banner](results/sample_predictions/banner.png)

> Automated multi-class lesion segmentation in Retinal Vein Occlusion (RVO) from OCT B-scans using a ResNet34-backbone U-Net

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Colab](https://colab.research.google.com/drive/1EgJNlI8H_CrUUxuAjdVa_aKUrWtXZ8Zf#scrollTo=KRgtHwaUw-Wz)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🩺 Clinical Background

**Retinal Vein Occlusion (RVO)** is one of the most common retinal vascular disorders and a leading cause of vision impairment worldwide. It occurs when a vein in the retina becomes blocked, leading to fluid leakage and tissue damage.

**Optical Coherence Tomography (OCT)** is the gold-standard imaging modality for diagnosing and monitoring RVO. It produces high-resolution cross-sectional scans (B-scans) of the retina, revealing structural changes that indicate disease severity.

Key lesions visible on OCT in RVO include:
- **SRF** (Subretinal Fluid) — fluid accumulation beneath the retina
- **IRF** (Intraretinal Fluid) — fluid pockets within retinal layers
- **ELM** (External Limiting Membrane) — a structural boundary whose disruption predicts vision loss
- **EZ** (Ellipsoid Zone) — photoreceptor integrity layer; loss correlates with reduced visual acuity

Manual delineation of these lesions is time-consuming and subject to inter-rater variability. Automated segmentation can enable faster diagnosis, consistent monitoring, and scalable screening — especially important in resource-limited settings.

---

## 🎯 Project Overview

This project implements a **deep learning pipeline for automated multi-class OCT lesion segmentation** in RVO patients. The model simultaneously segments 5 classes — Background, SRF, IRF, ELM, and EZ — from full-resolution OCT B-scans.

**Capstone project** for the Executive Programme for AI in Healthcare, IIT Delhi — **Team 10**

| | |
|---|---|
| **Task** | Multi-class semantic segmentation |
| **Input** | OCT B-scans (384 × 576 px, greyscale → 3-channel) |
| **Output** | Per-pixel class mask (5 classes) |
| **Best Mean Dice** | **0.802** |
| **Framework** | TensorFlow / Keras |
| **Platform** | Google Colab Pro |

---

## 🏗️ Architecture

The model uses a **U-Net** architecture with a **ResNet34 ImageNet-pretrained encoder**.

```
Input (384×576×3)
       │
  ResNet34 Encoder (ImageNet weights)
  ├── Block 1 → Skip connection
  ├── Block 2 → Skip connection
  ├── Block 3 → Skip connection
  └── Block 4 → Skip connection (bottleneck)
       │
  Decoder (transposed convolutions + skip fusion)
       │
  Output (384×576×5) → Softmax
```

**Key design choices:**
- ResNet34 encoder chosen over ResNet50 for better performance-to-complexity trade-off (validated via ablation)
- Full-resolution training (384×576) outperformed patch-based (stride-64/32) approaches
- ImageNet pretraining accelerated convergence on limited labelled data

---

## 📊 Experiment Results

11 experiments were conducted systematically varying loss function, resolution strategy, stride, and backbone.

| Run | Backbone | Resolution | Loss | Mean Dice |
|-----|----------|------------|------|-----------|
| 1 | ResNet34 | Patch (stride-64) | CCE | 0.623 |
| 2 | ResNet34 | Patch (stride-64) | Dice | 0.641 |
| 3 | ResNet34 | Patch (stride-64) | Focal | 0.638 |
| 4 | ResNet34 | Patch (stride-64) | CCE + Dice | 0.659 |
| 5 | ResNet34 | Patch (stride-64) | CCE + Focal | 0.651 |
| 6 | ResNet34 | Patch (stride-32) | CCE | 0.672 |
| 7 | ResNet34 | Patch (stride-32) | Dice | 0.688 |
| 8 | ResNet34 | Patch (stride-32) | CCE + Dice | 0.694 |
| 9 | ResNet50 | Full resolution | CCE + Dice | 0.771 |
| 10 | ResNet34 | Full resolution | CCE + Dice | 0.789 |
| **11** | **ResNet34 (ImageNet)** | **Full resolution** | **CCE + Dice** | **0.802 ✅** |

> **Best model:** Run 11 — ResNet34 with ImageNet pretraining, full-resolution input, combined CCE + Dice loss

---

## 📁 Repository Structure

```
RVO-OCT-Lesion-Segmentation/
│
├── data/
│   └── sample_images/          # Sample OCT B-scans and ground truth masks
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb  ← Colab-ready demo
│
├── src/
│   ├── model.py                 # U-Net architecture definition
│   ├── dataset.py               # Data loading and preprocessing
│   ├── train.py                 # Training loop
│   ├── inference.py             # Inference and mask generation
│   └── metrics.py               # Dice score per class
│
├── results/
│   ├── experiment_log.csv       # All 11 runs summary
│   └── sample_predictions/      # Visual outputs (image | GT | pred)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow >= 2.10`
- `segmentation-models` (for ResNet34 U-Net backbone)
- `numpy`, `opencv-python`, `matplotlib`

### Training

```python
# Key configuration
SEED = 42
INPUT_SHAPE = (384, 576, 3)   # Greyscale repeated ×3 for ResNet34
BATCH_SIZE = 2
NUM_CLASSES = 5               # Background, SRF, IRF, ELM, EZ
```

```bash
python src/train.py
```

### Inference

```python
from tensorflow.keras.models import load_model

model = load_model("resnet34_unet_fullres.keras", compile=False)
# Input: (384, 576, 3) greyscale OCT B-scan repeated across channels
```

Or open the Colab demo notebook directly:

[![Open In Colab](https://colab.research.google.com/drive/1EgJNlI8H_CrUUxuAjdVa_aKUrWtXZ8Zf#scrollTo=KRgtHwaUw-Wz)](https://colab.research.google.com/)

---

## 🖼️ Sample Results

> *(Sample images and prediction masks will be added here)*

| OCT B-scan | Ground Truth | Model Prediction |
|:---:|:---:|:---:|
| ![input](results/sample_predictions/Input.png) | ![gt](results/sample_predictions/GT.png) | ![pred](results/sample_predictions/Pred.png) |

**Colour legend:**

| Class | Colour |
|---|---|
| Background | Black |
| SRF | Blue |
| IRF | Green |
| ELM | Yellow |
| EZ | Red |

---

## 👥 Team

Capstone project by **Team 10**, Executive Programme for AI in Healthcare, IIT Delhi

Alfred Thomas · Heidrun Zeug · Kiran Kamble · **Komal** · Muneesh Kapoor · Nitika Jesingh · Saptarshi Paul Choudhury · Shivani Sheth · Shuvadeep Ganguly · Sumit Talwar

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- IIT Delhi Executive Programme for AI in Healthcare
- [segmentation-models](https://github.com/qubvel/segmentation_models) library by Pavel Iakubovskii
- Google Colab Pro for compute support
