# Iris Flower Classification

A machine learning project that classifies iris flowers into three species using the K-Nearest Neighbors (KNN) algorithm.

## Overview

This project uses the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) to build and evaluate a KNN classifier. It walks through the full data science workflow: loading data, exploratory data analysis (EDA), feature selection reasoning, model training, and accuracy evaluation.

## Dataset

The Iris dataset contains 150 samples across three species:

- *Iris setosa*
- *Iris versicolor*
- *Iris virginica*

Each sample has four features: **sepal length**, **sepal width**, **petal length**, and **petal width** (all in cm), plus a target label.

## Project Structure

```
iris_classification.ipynb   # Main Jupyter notebook
README.md
```

## Workflow

1. **Data Loading** — Load the Iris dataset from `sklearn.datasets` into a pandas DataFrame.
2. **Exploratory Data Analysis** — Inspect shape, types, and summary statistics; visualize feature correlations with a heatmap and a pairplot colored by species.
3. **Feature Selection** — All four features are used. EDA reveals that petal length and petal width carry more discriminative information, as they show less overlap between species than sepal measurements.
4. **Model Training** — Split data 80/20 (train/test), then fit a `KNeighborsClassifier` with `k=5`.
5. **Evaluation** — Report accuracy on the held-out test set.

## Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Dataset, model, and evaluation |

## Usage

Launch the notebook and run all cells in order:

```bash
jupyter notebook iris_classification.ipynb
```

## Results

The KNN classifier (k=5) achieves high accuracy on the test set, consistent with the clean, well-separated structure of the Iris dataset — particularly driven by the strong discriminating power of petal measurements.
