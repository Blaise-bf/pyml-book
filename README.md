# Python Machine Learning Book Implementations

This repository contains hands-on implementations and notebook explorations of core machine learning concepts. The project follows a learn-by-building approach, combining from-scratch models with scikit-learn workflows for feature engineering, model training, and interpretability.

## 🚀 Covered Topics

### Chapter 1: Training Simple ML Algorithms
- **Perceptron**: A basic linear classifier.
- **Adaline (Adaptive Linear Neuron)**:
  - **AdalineGD** (batch gradient descent)
  - **AdalineSGD** (stochastic gradient descent)

### Chapter 2: Tour of ML Classifiers
- **Logistic Regression** experiments and classifier workflows.
- Detailed [logistic regression derivation](./ch2-tour-of-ml-classifiers/LOGISTIC_REGRESSION_DERIVATION.md).

### Chapter 3: Preparing Data
- Encoding categorical targets and features.
- Normalization, standardization, and robust scaling.
- **Regularization analysis** (L1/L2) for logistic regression.
- **Sequential Backward Selection (SBS)** feature selection.
- **Random Forest feature importance** and threshold-based selection.
- **Model interpretability with SHAP and LIME**:
  - Global SHAP feature importance
  - Local SHAP contributions for a sample
  - Local LIME explanation for the same sample
  - SHAP vs LIME side-by-side comparison

### Chapter 4: Data Compression
- **Principal Component Analysis (PCA)**:
  - Covariance matrix, eigen decomposition, explained variance analysis.
  - Projection onto lower-dimensional principal subspaces.
- **Linear Discriminant Analysis (LDA)**:
  - Step-by-step formulation with class means, within-class scatter $S_W$, and between-class scatter $S_B$.
  - Generalized eigenvalue solution of $S_W^{-1}S_B$ and supervised projection to linear discriminants.
- **Nonlinear dimensionality reduction**:
  - **t-SNE** visualization workflow on the digits dataset.

## 🛠️ Project Structure
- `ch1-training-simple-ml-algorithms/`: Perceptron and Adaline implementations + notebook demos.
- `ch2-tour-of-ml-classifiers/`: Logistic regression notes and classifier notebooks.
- `ch3-preparing-data/`: Data preprocessing, feature selection (`sbs.py`), and interpretability notebooks.
- `ch4-data-compression/`: PCA, LDA, and t-SNE notebook experiments.
- `helpers/`: Shared utility helpers.

## 📓 Notebooks
- `ch1-training-simple-ml-algorithms/demo.ipynb`
- `ch2-tour-of-ml-classifiers/demo.ipynb`
- `ch3-preparing-data/data-prep.ipynb`
- `ch4-data-compression/data-compression.ipynb`

## 📚 Credits & References

The implementations and pedagogical structure are inspired by:

**"Machine Learning with PyTorch and Scikit-Learn"** by **Sebastian Raschka**, **Vahid Mirjalili**, and **Yuxi (Hayden) Liu**.

This repository is maintained as a personal study and experimentation workspace.
