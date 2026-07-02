# 🛢️ Automated Detection and Vessel Attribution of Illegal Bilge Dumping
### Using Sentinel-1 SAR Imagery and AIS Data Fusion

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-brightgreen?style=flat-square)
![VTU](https://img.shields.io/badge/VTU-Final%20Year%20Project-green?style=flat-square)
![SDG](https://img.shields.io/badge/SDG%2014-Life%20Below%20Water-0a97d9?style=flat-square)

**SMVITM, Visvesvaraya Technological University — 2026-27**

</div>

---

## 📌 Overview

Ships illegally discharge oily bilge water at sea — an environmental crime that produces thin, dark streaks visible in Synthetic Aperture Radar (SAR) imagery. Existing systems lack an end-to-end pipeline to detect spills, reject natural look-alikes, and attribute the dump to a specific vessel.

This project builds a 4-module pipeline to automatically detect illegal spills and trace them back to the source.

## 🏗️ System Architecture

```text
Sentinel-1 SAR Input
        ↓
┌───────────────────────────────────────┐
│  MODULE 1 — SAR Oil Spill Segmentation│  ✅ Complete
│  EfficientNet-B4 + scSE Attention     │
└───────────────────────────────────────┘
        ↓  Binary spill mask
┌───────────────────────────────────────┐
│  MODULE 2 — Look-alike Rejection      │  ✅ Complete
│  Random Forest on Geometric Features  │
└───────────────────────────────────────┘
        ↓  Confirmed spill + location
┌───────────────────────────────────────┐
│  MODULE 3 — AIS Vessel Filtering      │  ❌ Planned
│  Isolation Forest + 3D DBSCAN         │
└───────────────────────────────────────┘
        ↓  Ranked suspect vessels
┌───────────────────────────────────────┐
│  MODULE 4 — Drift Attribution         │  ❌ Planned
│  Bidirectional Lagrangian Model       │
└───────────────────────────────────────┘
        ↓
Output: Spill Map + Vessel ID + Confidence Score
```

## 📊 Final Evaluation Results (Modules 1 & 2)

After upgrading our architecture and testing on a combined dataset (SOS + MKLab), we evaluated our baseline model (MobileNetV3) against our final model (EfficientNet-B4) across 949 test images.

| Backbone Model | M1 IoU | M1 Dice | M2 Precision | M2 Recall | Pipeline F1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| MobileNetV3-Large (256px) | 0.5673 | 0.6848 | **0.9582** | **0.3599** | **0.5232** |
| **EfficientNet-B4 (512px)** | **0.6349** | **0.7423** | 0.9379 | 0.3386 | 0.4975 |

**🏆 Conclusion:**
**EfficientNet-B4** is our final selected model. It achieved a massive **+6.7% improvement in segmentation accuracy (IoU)** by effectively detecting thin, broken oil streaks that the baseline missed. The Look-alike Classifier (Module 2) acts as a strict filter, maintaining an exceptional ~94% Precision in rejecting natural look-alikes.

---

## 🔬 Technical Highlights

### Module 1: Oil Spill Segmentation
* **Architecture**: DeepLabV3+ with an **EfficientNet-B4** encoder (19M params).
* **Attention Mechanism**: Custom **scSE (Spatial and Channel Squeeze-and-Excitation)** block to heavily weight SAR backscatter anomalies.
* **Loss Function**: Focal (0.4) + Tversky (0.6) to penalize false negatives (missed oil).
* **Resolution**: Trained on high-res 512×512 random crops.

### Module 2: Look-Alike Rejection
* **Feature Extraction**: Extracts 13 geometric, morphological, and contextual features (e.g., area, elongation, compactness, standard deviation of intensity).
* **Classifier**: Random Forest trained on real look-alike labels (MKLab dataset) to achieve > 90% precision.

---

## 👥 Team

| Name | USN | Role |
|---|---|---|
| Rohith Sheregar | 4MW23CS120 | ML Pipeline, SAR Processing, AIS Integration |
| Reynol D'Souza | 4MW23CS119 | Team Lead, System Architecture |
| Prajwal Shanbhag | 4MW23CS095 | Data Processing, Evaluation |
| Nishith R Poojary | 4MW23CS087 | Drift Modelling, Visualization |

**Institution:** Shri Madhwa Vadiraja Institute of Technology and Management (VTU)

---

## ⚙️ Quick Start

```bash
git clone https://github.com/rohith-sheregar/Oil-Spill-Detection.git
cd Oil-Spill-Detection
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 📥 Data & Models Setup
1. **Dataset**: Download the SOS & MKLab datasets from [Google Drive (Dataset)](https://drive.google.com/file/d/12grU_EAPbW75eyyHj-U5pOfnwQzm0MFw/view) and extract them into the `data/` folder.
2. **Pre-trained Models**: Download the trained `.pth` and `.pkl` weights from [Google Drive (Models)](https://drive.google.com/drive/folders/1ybKpImQJs8WbQ1ZABUUoE5a8M-jEJLMC?usp=drive_link).
3. Place the downloaded model files into the `outputs/checkpoints/` directory so they look exactly like this:
```text
outputs/checkpoints/
├── best_model_efficientnet.pth
├── best_model_mobilenet.pth
└── module2_classifier.pkl
```

### Key Commands

```bash
# Compare both final models side-by-side
python evaluate.py --backbone both --dataset combined

# Run Full Integrated Pipeline on a specific image
python predict_pipeline.py --image data/sos/test/sentinel/image/0.png

# Visualise raw Module 1 Predictions
python -m src.visualize
```

---

<div align="center">
<b>SDG 14 — Life Below Water</b><br>
Supporting marine ecosystem protection by enabling automated detection and forensic attribution of illegal oil pollution.
</div>
