# 🕸️ GCN Fraud Detection on the Elliptic Bitcoin Dataset

Graph-based fraud detection using a **Graph Convolutional Network (GCN)** on the Elliptic Bitcoin transaction dataset.

This project implements a production-style, modular, and reproducible graph machine learning pipeline for detecting illicit cryptocurrency transactions.

---

## 🚀 Overview

The Elliptic dataset represents a **Bitcoin transaction graph** where:

- **Nodes** → Bitcoin transactions  
- **Edges** → Flow of BTC between transactions  
- **Task** → Classify transactions as **licit** or **illicit**

This repository includes:

- Config-driven GCN model (PyTorch Geometric)
- Logits-based binary classification (`BCEWithLogitsLoss`)
- Full-graph supervised training
- GPU-safe execution with automatic CUDA fallback
- TensorBoard logging
- Artifact management (weights, runs, visualizations)
- CLI-driven training & visualization
- Production-ready `src/` package layout

---

## 📊 Dataset

The **Elliptic Bitcoin Dataset** maps transactions to real-world licit and illicit entities.

### Graph Statistics

- **Nodes:** 203,769  
- **Edges:** 234,355  
- **Illicit (class1):** 4,545 (~2%)  
- **Licit (class2):** 42,019 (~21%)  
- **Unknown:** Remaining (~77%)

The classification task is highly imbalanced, making graph structure particularly important.

### Features

Each node has **166 features**:

- **94 Local features**
  - Transaction fee
  - Number of inputs/outputs
  - BTC volume
  - Time step (1–49)
- **72 Aggregated features**
  - One-hop neighbor statistics
  - Max/min/std/correlation of transaction properties

Time steps represent ~2-week intervals.  
Each time step forms a connected component with no edges across time steps.

---

## 📥 Dataset Download

The dataset (~880MB) is not included in this repository.

Download it from:

👉 https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

After downloading, place the files inside:

```
data/
  elliptic_txs_features.csv
  elliptic_txs_edgelist.csv
  elliptic_txs_classes.csv
```
---

## 🏗️ Project Structure
```bash
gcn-fraud-detection/
├── configs/
├── scripts/
├── src/
│   └── fraud_detection/
│       ├── data/
│       ├── models/
│       └── training/
├── artifacts/
│   ├── weights/
│   ├── runs/
│   └── visualizations/
└── requirements.txt
```

Design principles:

- Modular `src/` structure
- Separation of dataset, model, and training logic
- Config-driven experimentation
- Reproducible splits
- No raw data or artifacts committed

---

## 🧠 Model

### Graph Convolutional Network (GCN)

Architecture:
```
Input → GCNConv → tanh →
        GCNConv → tanh →
        Linear → Logits
```

- Binary classification
- `BCEWithLogitsLoss`
- Float32 tensor pipeline
- Sigmoid applied only during metric computation

---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 🏋️ Training

From project root:
```bash
PYTHONPATH=src python scripts/train.py --config configs/gcn.yaml
```

Configuration

All experiments are driven by:

```bash
configs/gcn.yaml
```

Configurable parameters include:

- Model dimensions
- Learning rate
- Weight decay
- Epoch count
- Device (cpu / cuda)
- Artifact directories

CUDA fallback is automatic if unavailable.

---
## 📈 Logging

Artifacts are saved to:

- Weights: `artifacts/weights/`
- TensorBoard logs: `artifacts/runs/`
- Visualizations: `artifacts/visualizations/`

Launch TensorBoard:
```bash
tensorboard --logdir artifacts/runs/
```

---
## 📊 Visualization

Visualize predictions for a specific time step:

```bash 
PYTHONPATH=src python scripts/visualize.py \
    --config configs/gcn.yaml \
    --step 10
```
---

### 📊 Visualization Legend

**Color legend:**

- 🟢 **Green** → True licit  
- 🔴 **Red** → True illicit  
- 🟠 **Orange** → Predicted illicit  
- 🔵 **Blue** → Predicted licit  

---

## 🔬 Evaluation Metrics

During training, the following metrics are tracked:

- Accuracy  
- F1-score (micro and macro)  
- Recall  
- Precision  
- Confusion Matrix  

Given the heavy class imbalance (~2% illicit), **F1-score is more informative than raw accuracy**.

---

## 🛠️ Tech Stack

- PyTorch  
- PyTorch Geometric  
- OmegaConf  
- NetworkX  
- TensorBoard  
- scikit-learn  

---

## 🔮 Future Improvements

- Add GraphSAGE / GAT architectures  
- Temporal graph modeling  
- Node embedding export  
- Stratified time-based validation  
- Early stopping  
- Hyperparameter sweeps  

---

## 📄 License

This project is intended for educational and research purposes.  
Dataset ownership belongs to Elliptic.
