# Python Machine Learning Book Implementations

This repository contains "from-scratch" implementations of fundamental machine learning algorithms. The goal of this project is to understand the inner workings of these algorithms by implementing them using Python and NumPy, following the architectural patterns of popular libraries like scikit-learn.

## 🚀 Algorithms Implemented

### Chapter 1: Simple Machine Learning Algorithms
- **Perceptron**: A basic linear neuron model.
- **Adaline (Adaptive Linear Neuron)**:
  - **AdalineGD**: Implementation using Batch Gradient Descent.
  - **AdalineSGD**: Implementation using Stochastic Gradient Descent.

### Chapter 2: Machine Learning Classifiers
- **Logistic Regression**: A probabilistic classifier using the sigmoid activation function and log loss optimization. Includes a detailed [mathematical derivation](./ch2-tour-of-ml-classifiers/LOGISTIC_REGRESSION_DERIVATION.md) of the loss function gradients.

## 🛠️ Project Structure
- `ch1-training-simple-ml-algorithms/`: Implementations of Perceptron and Adaline.
- `ch2-tour-of-ml-classifiers/`: Implementations of Logistic Regression and other classifiers.
- `helpers/`: Shared utility functions, including decision region visualization.
- `demo.ipynb`: Interactive Jupyter notebooks for visualizing algorithm performance.

## 📚 Credits & References

The implementations and pedagogical structure of this codebase are based on the materials and ideas from:

**"Machine Learning with PyTorch and Scikit-Learn"** by **Sebastian Raschka**, Vahid Mirjalili, and Yuxi (Hayden) Liu.

This project serves as a personal study guide and reference for the concepts covered in the book.
