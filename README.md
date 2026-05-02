# 🧠 Module 1 — SAR Oil Spill Segmentation

---

## 📌 Problem

Detect oil spill regions from SAR imagery where:

* oil appears as dark patches
* severe class imbalance exists
* noise and contrast variation present

---

## 🏗️ Model Architecture

```
Input: 256×256×3
↓
MobileNet Backbone
↓
scSE Attention
↓
DeepLabv3+ ASPP
↓
Output: Binary Mask
```

---

## 🧠 Design Justification

| Component  | Reason                    |
| ---------- | ------------------------- |
| MobileNet  | Lightweight + high IoU    |
| scSE       | Focus on relevant regions |
| DeepLabv3+ | Multi-scale detection     |
| BCE + Dice | Handles imbalance         |

---

## 📊 Dataset

| Dataset      | Images | Sensor |
| ------------ | ------ | ------ |
| SOS Sentinel | 4193   | C-band |
| SOS PALSAR   | 3877   | L-band |

Total: 6455 train / 1615 test

---

## ⚙️ Training Configuration

* Optimizer: AdamW
* LR: 1e-4
* Scheduler: Cosine Annealing
* Loss: BCE + Dice
* Epochs: 50

---

## 📈 Results

* IoU: 0.64
* Dice: 0.75

Training trend:

```
Epoch 1 → 0.62
Epoch 11 → 0.66 (best)
Epoch 30 → plateau
```

---

## 📂 Code Structure

* dataset.py → loading
* model.py → architecture
* train.py → training
* utils.py → metrics
* visualize.py → outputs

---

## ⚙️ Usage

```bash id="m1run"
git checkout module-1
pip install -r requirements.txt
python -m src.train
```

---

## 🚧 Limitations

* Confuses look-alikes
* Needs post-processing

---

## 🎯 Output

Binary segmentation mask for next module
