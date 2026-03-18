# Feature-Based Structural Damage Detection (FSDD)

> "Advanced ensemble deep learning for automated structural damage assessment in post-disaster environments using UAV aerial imagery."

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-YOLOv8%20%7C%20Detectron2%20%7C%20YOLOv5-orange.svg)](https://github.com/ultralytics/ultralytics)
[![mAP](https://img.shields.io/badge/mAP-0.91-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Proposed System – FSDD Architecture](#proposed-system--fsdd-architecture)
4. [Key Results](#key-results)
5. [Dataset](#dataset)
6. [Technologies Used](#technologies-used)
7. [Notebooks](#notebooks)
8. [Quick-Start](#quick-start)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Future Work](#future-work)
11. [Contributing](#contributing)
12. [Citation](#citation)
13. [Authors](#authors)

---

## Overview

Natural disasters such as earthquakes, hurricanes, and floods cause catastrophic damage to buildings and critical infrastructure, endangering thousands of lives. Rapid and accurate structural damage assessment is essential for rescue prioritisation, resource allocation, and reconstruction planning.

**FSDD** is a novel **Feature-Based Structural Damage Detection** system that harnesses the power of **ensemble deep learning** combined with **active learning** and **UAV (drone) aerial imagery** to automate and significantly improve damage assessment in post-disaster environments.

The system achieves a state-of-the-art **mAP of 0.91**, **Precision of 0.93**, **Recall of 0.91**, and **F1-Score of 0.92** — outperforming each individual model it is built upon.

For the full theoretical background and methodology, see [`FSDD_Thesis.pdf`](./FSDD_Thesis.pdf).

---

## Problem Statement

Traditional structural damage assessment methods are:
- **Slow** – reliant on manual, on-ground inspection by human experts.
- **Dangerous** – requiring surveyors to enter potentially unstable structures.
- **Inconsistent** – subject to human error and limited scope in disaster zones.
- **Not scalable** – unable to cover large disaster-affected regions quickly.

Existing single deep-learning model solutions (YOLOv8, Detectron2, YOLOv5) each have their own limitations in precision, recall, or generalisation. The core challenge is:

> *How can we accurately classify and localise multiple levels of structural damage across a large number of UAV images in real-time, with minimal labelled data, while maximising model performance?*

FSDD directly addresses this by combining the complementary strengths of three state-of-the-art CNN architectures through ensemble learning.

---

## Proposed System – FSDD Architecture

### 1. Data Acquisition
- Aerial images captured using **UAV drones** over disaster-hit areas.
- Dataset: **ISBDA (~160 images)** with 5 damage classes:
  - `no_d` – No damage
  - `slight` – Slight damage
  - `slight_h` – Slight damage (high confidence)
  - `severe` – Severe damage
  - `severe_h` – Severe damage (high confidence)

### 2. Preprocessing & Annotation
- Images preprocessed using **OpenCV** (resize, contrast enhancement, normalisation).
- Annotations created with **Roboflow** (bounding boxes, class labels).
- Train / Test split: **70% training | 30% testing**.

### 3. Feature Extraction
- Gradient-based features using **Sobel kernels** and the **Harris corner detector**.
- Structure tensor components computed and smoothed with a Gaussian filter (sigma = 1).
- Harris response: `R = det(A) - k x trace(A)^2` (k = 0.05).

### 4. Deep Learning Models

| Model | Architecture | Role |
|---|---|---|
| YOLOv8 | CNN – anchor-free, real-time | Primary detector (highest single-model mAP) |
| Detectron2 | R-CNN – region-proposal | Fine-grained localisation |
| YOLOv5 | CNN – multi-scale detection | Speed vs. accuracy trade-off |

### 5. Active Learning
- Iteratively selects the **most uncertain / informative samples** from unlabelled data for expert annotation.
- Reduces labelling cost while improving model generalisation.
- Grid search optimises hyperparameters for each base model.

### 6. Ensemble Learning
- All three CNN models are trained on the **same ISBDA dataset**.
- Predictions are combined via **weighted average / majority voting**.
- Final FSDD prediction = best-of-ensemble across classes.

```
UAV Images --> Preprocessing --> Feature Extraction
                                        |
              +--------------------------+-----------------------+
              |                          |                       |
           YOLOv8                  Detectron2                YOLOv5
              |                          |                       |
              +--------------------------+-----------------------+
                                        |
                             Ensemble Voting / Averaging
                                        |
                              FSDD Damage Classification
                          (no_d | slight | severe | etc.)
```

---

## Key Results

### Performance Comparison

| Model | mAP | Precision | Recall | F1-Score |
|---|---|---|---|---|
| YOLOv8 | 0.89 | 0.89 | 0.85 | 0.87 |
| YOLOv5 | 0.84 | 0.82 | 0.84 | 0.83 |
| Detectron2 | 0.83 | 0.80 | 0.84 | 0.82 |
| **FSDD (Ensemble)** | **0.91** | **0.93** | **0.91** | **0.92** |

**FSDD outperforms every individual baseline model** across all metrics, demonstrating the clear advantage of ensemble learning for structural damage detection.

Ranking by mAP: **FSDD > YOLOv8 > YOLOv5 > Detectron2**

---

## Dataset

**ISBDA (Image Set for Building Damage Assessment)**
- Approximately 160 UAV aerial images
- 5 damage severity classes
- Annotated using Roboflow
- Balanced class distribution across train/test splits
- Training: 70% | Testing: 30%

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.9+ | Core programming language |
| YOLOv8 / Ultralytics | Real-time object detection |
| Detectron2 | Region-proposal CNN |
| YOLOv5 | Multi-scale detection |
| OpenCV | Image preprocessing |
| scikit-image / scipy | Feature extraction (Sobel, Harris) |
| Matplotlib | Visualisations and training curves |
| Roboflow | Dataset management and annotation |
| Google Colab | Cloud training environment |
| CUDA / NVIDIA Tesla K80 | GPU acceleration |
| NumPy | Numerical computation |

---

## Notebooks

### `3DSDD.ipynb` – Feature Extraction and Visualisation
- Loads aerial images and converts to grayscale.
- Computes X/Y gradients (Sobel kernels via `scipy.signal.convolve2d`).
- Builds structure tensor, Gaussian-smoothed components, and Harris response.
- Visualises: gradient magnitude, Harris heat-map, corner/edge overlays.

### `Ensemble_predictions.ipynb` – Ensemble Classification
- Loads feature sets from `3DSDD.ipynb`.
- Trains and combines base classifiers (Logistic Regression, Random Forest, SVM) using `VotingClassifier`.
- Reports Precision, Recall, and F1-Score on a held-out test set.
- Plots ROC curves and confusion matrices.

---

## Quick-Start

```bash
# Clone the repository
git clone https://github.com/Joevita20/Feature-Based-Structural-Damage-Detection.git
cd Feature-Based-Structural-Damage-Detection

# Set up a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Or manually:
pip install numpy scipy scikit-image matplotlib jupyter torch torchvision

# Launch notebooks
jupyter notebook
```

Open `3DSDD.ipynb` first for feature extraction, then `Ensemble_predictions.ipynb` for classification.

---

## Evaluation Metrics

| Metric | Formula | Description |
|---|---|---|
| mAP | mean of AP per class | Overall detection performance |
| Precision | TP / (TP + FP) | Exactness of predictions |
| Recall | TP / (TP + FN) | Completeness of predictions |
| F1-Score | 2 x P x R / (P + R) | Harmonic mean of precision and recall |
| Avg. Loss | sum(loss) / iterations | Model convergence (lower = better) |

---

## Future Work

- Real-time deployment on edge devices (Jetson Nano / Raspberry Pi) for on-field drone use.
- Larger datasets including multi-disaster type coverage (floods, earthquakes, typhoons).
- 3-D point-cloud integration for volumetric damage estimation.
- Transformer-based architectures (e.g., DETR, ViT) as additional ensemble members.
- Semi-supervised learning to further reduce reliance on labelled data.
- Mobile app for first responders to capture images and receive instant damage classification.

---

## Contributing

Contributions are welcome!
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Ensure all notebooks run end-to-end without errors.
4. Follow PEP 8 style and update the README for any major additions.
5. Open a pull request with a clear description of your changes.

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If you use FSDD in your academic work, please cite:

```bibtex
@misc{fsdd2024,
  author    = {Joevita Faustina Doss, F. and Sasana, R. and Rakshitha, P.},
  title     = {Feature-Based Structural Damage Detection using Ensemble Deep Learning},
  year      = {2024},
  school    = {Anna University, MIT Campus – Department of Computer Technology},
  url       = {https://github.com/Joevita20/Feature-Based-Structural-Damage-Detection},
  note      = {CIP Final Year Thesis, Version 1.0}
}
```

---

## Authors

| Name | Role | GitHub |
|---|---|---|
| **F. Joevita Faustina Doss** | Lead Developer and Researcher | [@Joevita20](https://github.com/Joevita20) |
| R. Sasana | Co-Author and Researcher | — |
| P. Rakshitha | Co-Author and Researcher | — |

**Guided by:**
Dr. Raja Kathiroli and Dr. V. P. Jayachitra
*Department of Computer Technology, Anna University, MIT Campus*

---

*Built at Anna University, MIT Campus — Department of Computer Technology*
