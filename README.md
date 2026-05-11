# Iris Flower Classification using Machine Learning

A machine learning classification project built using Python and Scikit-learn to classify Iris flower species based on sepal and petal measurements. The project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), visualization, model training, evaluation, and performance comparison across multiple classification algorithms.

---

## Project Overview

The Iris dataset is one of the most well-known datasets in machine learning and pattern recognition. It contains measurements of iris flowers belonging to three different species:

- Setosa
- Versicolor
- Virginica

The goal of this project is to build machine learning models capable of accurately predicting the species of a flower using its physical characteristics.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Dataset Information

The dataset contains 150 observations with the following features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

Target Variable:
- Iris Species

---

# Machine Learning Workflow

## 1. Data Loading and Inspection

The dataset was loaded using Scikit-learn and converted into a Pandas DataFrame for easier analysis and manipulation.

Initial inspection included:
- Dataset shape
- Column names
- Statistical summaries
- Null value checks

---

## 2. Exploratory Data Analysis (EDA)

Exploratory analysis was performed to understand:
- Feature distributions
- Relationships between variables
- Correlations between measurements
- Class separability

### Key Observations

- Petal length and petal width showed strong correlation
- Setosa flowers were clearly separable from other species
- Versicolor and Virginica had overlapping feature spaces in some dimensions

---

## 3. Data Visualization

Several visualizations were created to better understand the dataset.

### Pair Plot Visualization

Pair plots were used to visualize relationships between all feature pairs while distinguishing species using color mapping.

Insights:
- Petal features provided the clearest separation between classes
- Some overlap existed between Versicolor and Virginica

### Correlation Heatmap

A heatmap was generated to analyze feature correlations.

Key findings:
- Petal length and petal width had strong positive correlation
- Sepal width showed weaker correlation with other features

---

## 4. Data Preparation

The dataset was split into:
- Training Set
- Testing Set

This allowed unbiased evaluation of model performance on unseen data.

---

## 5. Model Training

Multiple machine learning classification algorithms were trained and evaluated:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)

Using multiple models allowed comparative analysis of performance and prediction capability.

---

## 6. Model Evaluation

Models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Classification Report

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions |
| Recall | Ability to identify true positives |
| F1-Score | Balance between precision and recall |

---

# Results

All tested models achieved very high accuracy on the Iris dataset.

This is expected because:
- The dataset is clean and structured
- Classes are relatively well-separated
- The dataset contains low noise

Among all features:
- Petal Length
- Petal Width

were the strongest contributors toward classification accuracy.

---

# Visualizations Included

- Pair Plot
- Correlation Heatmap
- Confusion Matrix
- Model Accuracy Comparison

---

# Conclusion

This project successfully demonstrated a complete machine learning classification workflow using the Iris dataset.

The project covered:
- Data preprocessing
- Exploratory data analysis
- Data visualization
- Model training
- Model comparison
- Performance evaluation

The results highlighted how machine learning algorithms can effectively classify structured datasets when meaningful features are present.

Additionally, the project emphasized the importance of visualization and feature analysis in understanding dataset behavior before model training.

---

# Future Improvements

Possible future enhancements include:

- Hyperparameter tuning
- Cross-validation
- Dimensionality reduction (PCA)
- Deployment using Flask or Streamlit
- Testing on larger real-world datasets

---

# Author

Taniksha Shah
