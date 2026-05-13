# 🌸 Iris Flower Classification

> Comparing four classic machine learning algorithms on the Iris dataset — KNN, Logistic Regression, Decision Tree, and SVM — with full EDA, visualization, and evaluation.

---

## 📌 Overview

This notebook walks through a complete machine learning classification pipeline on the iconic **Iris dataset**, covering exploratory data analysis, feature visualization, multi-model training, and performance evaluation.

Four classifiers are trained and compared side-by-side:

| Model | Type |
|---|---|
| K-Nearest Neighbors (KNN) | Instance-based |
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Support Vector Machine (SVM) | Kernel-based |

All models achieve very high accuracy — a result of the dataset's clean structure and the strong discriminatory power of petal measurements.

---

## 📂 Dataset

**Source:** `sklearn.datasets.load_iris` (built-in, no download required)  
**Samples:** 150 &nbsp;|&nbsp; **Features:** 4 &nbsp;|&nbsp; **Classes:** 3

| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |
| `target` | Species: 0 = Setosa, 1 = Versicolor, 2 = Virginica |

> **Key observation:** Petal length and petal width are the most discriminating features — they show far less overlap between species than sepal measurements, making them the primary drivers of classification accuracy.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-dataframes-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-arrays-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-plotting-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-visualization-4c72b0)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)

```python
# Data
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset
from sklearn.datasets import load_iris

# Splitting
from sklearn.model_selection import train_test_split

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Evaluation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Taniksha-shah/iris-classification.git
cd iris-classification
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the notebook
```bash
jupyter notebook iris_classification.ipynb
```

> No dataset download needed — the Iris dataset loads directly from scikit-learn.

---

## 📊 Analysis Pipeline

```
1. Load Dataset         →  load_iris() → numpy arrays + DataFrame
2. EDA                  →  shape, dtypes, describe(), null check
3. Data Visualization   →  correlation heatmap + pairplot (hue by species)
4. Train-Test Split     →  80% train / 20% test (random_state=42)
5. Model Training       →  KNN, Logistic Regression, Decision Tree, SVM
6. Model Evaluation     →  accuracy bar chart, confusion matrix, classification report
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Training set:** 120 samples
- **Test set:** 30 samples

---

## 🤖 Models

### K-Nearest Neighbors
```python
knn = KNeighborsClassifier(n_neighbors=5)
```
Classifies each sample by majority vote among its 5 nearest neighbors in feature space. Simple and interpretable, no training phase.

### Logistic Regression
```python
lr = LogisticRegression()
```
A linear probabilistic classifier. Works well on the Iris dataset due to its near-linear class boundaries (especially for Setosa).

### Decision Tree
```python
dt = DecisionTreeClassifier()
```
Recursively splits the feature space using the most informative thresholds. Naturally interpretable — the splits align closely with petal measurement boundaries.

### Support Vector Machine
```python
svm = SVC()
```
Finds the maximum-margin hyperplane separating classes in feature space. With the default RBF kernel, handles non-linear boundaries effectively.

---

## 📈 Results

All four models achieve very high accuracy on the test set. This is expected — the Iris dataset is small, clean, and its three species are largely separable, especially along petal dimensions.

### Accuracy Comparison

```
KNN                 ████████████████████  ~96–100%
Logistic Regression ████████████████████  ~96–100%
Decision Tree       ████████████████████  ~96–100%
SVM                 ████████████████████  ~96–100%
```

### Evaluation Metrics (KNN — primary model)

- **Confusion Matrix** — visualized with `ConfusionMatrixDisplay`
- **Classification Report** — precision, recall, F1-score per class

> Setosa is perfectly classified by all models due to its distinct petal dimensions. Minor confusion may occur between Versicolor and Virginica, which have slightly overlapping feature distributions.

---

## 🔍 Key Findings

- **Petal dimensions dominate** — `petal length` and `petal width` show strong positive correlation with each other and clear species separation in the pairplot.
- **Sepal features overlap** — `sepal length` and `sepal width` show more inter-species overlap, contributing less to classification power.
- **All models perform comparably** — on a clean, low-dimensional dataset like Iris, algorithm choice matters less than feature quality.
- **Dataset simplicity is a ceiling** — the near-perfect scores reflect dataset characteristics, not model superiority; a harder dataset would differentiate them more.

---

## 📁 Notebook Structure

```
iris_classification.ipynb
├── Section 1  — Importing Libraries
├── Section 2  — Loading Dataset
│   ├── load_iris() → X, y arrays
│   └── load_iris(as_frame=True) → DataFrame
├── Section 3  — Exploratory Data Analysis
│   ├── shape, head(), info(), describe()
│   └── Null check
├── Section 4  — Data Visualization
│   ├── Correlation heatmap (seaborn, coolwarm, annotated)
│   └── Pairplot (hue by target species)
├── Section 5  — Model Training
│   ├── Train-test split (80/20, random_state=42)
│   ├── KNN (k=5)
│   ├── Logistic Regression
│   ├── Decision Tree
│   └── SVM (RBF kernel)
└── Section 6  — Model Evaluation
    ├── Accuracy bar chart (all 4 models)
    ├── Confusion matrix (KNN)
    └── Classification report (KNN)
```

---

## 🔮 Future Enhancements

- [ ] **Hyperparameter Tuning** — GridSearchCV for optimal `k` in KNN, `C` and `kernel` in SVM, `max_depth` in Decision Tree
- [ ] **Cross-Validation** — Replace single train-test split with k-fold CV for more robust accuracy estimates
- [ ] **Feature Importance** — Extract and visualize Decision Tree feature importances
- [ ] **Decision Boundary Plots** — Visualize 2D decision boundaries for each classifier using petal features
- [ ] **Pipeline with Scaling** — Wrap preprocessing + model in a `sklearn.pipeline.Pipeline` (especially useful for KNN and SVM which are distance-sensitive)
- [ ] **ROC / AUC Curves** — Multi-class ROC curves using `OneVsRest` strategy
- [ ] **Ensemble Methods** — Add Random Forest and Gradient Boosting for comparison

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset: [Fisher's Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) via `sklearn.datasets`
- Environment: Jupyter Notebook / Kaggle Notebooks
