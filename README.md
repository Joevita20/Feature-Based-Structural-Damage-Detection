# Feature‑Based Structural Damage Detection (FSDD)

**A Python toolbox for detecting structural damage in 3‑D data using gradient‑based feature extraction and ensemble classification.**

---

## 📖 Overview
FSDD provides a complete, reproducible pipeline:
1. **Feature extraction** – Sobel & Harris corner detection on 3‑D images (see `3DSDD.ipynb`).
2. **Ensemble classification** – Combine multiple descriptors with a voting classifier for robust damage prediction (`Ensemble_predictions.ipynb`).
3. **Visualization** – Interactive plots of keypoints, edge maps, and classification results.

The notebooks are self‑contained and can be run on any image dataset (e.g., the sample `resources/box.jpg`).

---

## 🚀 Quick‑Start
```bash
# Clone the repository
git clone https://github.com/Sauce16/FSDD.git && cd FSDD

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt   # or: pip install numpy scipy scikit-image matplotlib jupyter

# Launch the notebooks
jupyter notebook
```
Open `3DSDD.ipynb` first, then `Ensemble_predictions.ipynb`. All cells are executable from top to bottom.

---

## 📂 Project Structure
```
FSDD/
│
├─ 3DSDD.ipynb                # Feature extraction & visualisation
├─ Ensemble_predictions.ipynb # Ensemble classification demo
├─ README.md                  # (this file)
└─ resources/
   └─ box.jpg                # Sample image used in the notebooks
```
Add a `data/` folder for your own datasets.

---

## 🛠️ Notebooks at a Glance
### `3DSDD.ipynb`
* Loads an image, converts to grayscale, and computes X/Y gradients using Sobel kernels.
* Builds the structure‑tensor, calculates the Harris response (`k = 0.05`), and visualises corners (red) and edges (green).
* **Suggested improvements** – vectorised masking, `ipywidgets` for interactive `sigma`/`k`, and a reusable `compute_harris` function.

### `Ensemble_predictions.ipynb`
* Instantiates several base classifiers (Logistic Regression, Random Forest, SVM) and combines them with `VotingClassifier`.
* Reports precision, recall, and F1‑score on a test split.
* **Suggested improvements** – pipeline with `StandardScaler`, cross‑validation, hyper‑parameter grid search, model persistence (`joblib.dump`), and feature‑importance visualisation.

---

## 📈 Results & Visualisations
Running the notebooks yields:
* Gradient magnitude maps for each axis.
* Harris corner response heat‑map.
* Overlay image with red corners and green edges.
* Classification report showing the benefit of the ensemble (e.g., accuracy ↑ 5‑10 %).

Export figures with `plt.savefig('figure.png')` for inclusion in reports or presentations.

---

## 🤝 Contributing
Contributions are welcome! Please:
1. Fork the repo.
2. Create a feature branch.
3. Keep notebooks runnable from start to finish.
4. Follow PEP 8 style and update this README if you add major features.
5. Open a pull request.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📚 Citation
If you use FSDD in academic work, cite it as follows:
```bibtex
@misc{fsdd2024,
  author = {Joevita Faustina},
  title  = {Feature‑Based Structural Damage Detection},
  year   = {2024},
  url    = {https://github.com/Sauce16/FSDD},
  note   = {Version 1.0}
}
```

---

### 👤 Author
**Joevita Faustina** – Researcher & Developer

---


A Python toolbox for detecting structural damage in 3‑D data using gradient‑based feature extraction and ensemble classification. Includes two notebooks: `3DSDD.ipynb` (feature extraction & visualisation) and `Ensemble_predictions.ipynb` (ensemble model demo).

## Quick start
```bash
git clone https://github.com/Sauce16/FSDD.git && cd FSDD
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt  # or: pip install numpy scipy scikit-image matplotlib jupyter
jupyter notebook
```
Open the notebooks, run the cells, and explore the visualisations.

## Highlights
- Sobel & Harris corner detection on 3‑D images
- Multi‑scale Gaussian smoothing for robust keypoints
- Simple ensemble voting classifier for damage prediction
- Fully reproducible notebooks; easy to plug in new descriptors or ML models

## Contributing & License
Contributions welcome via pull‑requests. Licensed under MIT.

## Citation
```bibtex
@misc{fsdd2024,
  author = {Sauce16},
  title  = {Feature‑Based Structural Damage Detection},
  year   = {2024},
  url    = {https://github.com/Sauce16/FSDD},
  note   = {Version 1.0}
}
```
