# Logistic Regression Loss Function Derivation

This document provides a mathematical derivation of the loss function (Log Loss) and its derivative for the Logistic Regression classifier, as implemented in `logisticreg.py`.

## 1. The Logistic Model

The Logistic Regression model uses the **Sigmoid (Logistic) activation function** to map net input to a probability $p \in [0, 1]$.

### Net Input
The net input $z$ for a single sample $x$ is:
$$ z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^m w_j x_j + b $$

### Activation Function (Sigmoid)
The activation $\sigma(z)$ is defined as:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

A useful property of the sigmoid derivative that we will use later is:
$$ \frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z)) $$

---

## 2. The Likelihood Function

For a binary classification problem ($y \in \{0, 1\}$), we assume the outcomes follow a Bernoulli distribution:
- $P(y=1|x; w) = \sigma(z)$
- $P(y=0|x; w) = 1 - \sigma(z)$

For a single observation, the probability can be written as:
$$ P(y|x; w) = (\sigma(z))^y (1 - \sigma(z))^{1-y} $$

For a dataset of $n$ independent observations, the **Likelihood Function** $L(w)$ is the product of individual probabilities:
$$ L(w) = \prod_{i=1}^{n} (\sigma(z^{(i)}))^{y^{(i)}} (1 - \sigma(z^{(i)}))^{1-y^{(i)}} $$

---

## 3. The Log-Likelihood and Loss Function

To simplify calculations (converting products to sums) and avoid numerical underflow, we take the natural logarithm to get the **Log-Likelihood** $l(w)$:

$$
 l(w) = \log L(w) = \sum_{i=1}^{n} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right] 
 $$

In machine learning, we typically minimize a **Loss Function**. We define the Log Loss $J(w)$ as the Negative Log-Likelihood:

$$
 J(w) = -l(w) = -\sum_{i=1}^{n} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right] 
 $$

---

## 4. Derivative of the Loss Function

To update the weights using Gradient Descent, we need the partial derivative of $J(w)$ with respect to each weight $w_j$. Using the **Chain Rule**:

$$
 \frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial z} \cdot \frac{\partial z}{\partial w_j} 
 $$

### Part A: $\frac{\partial J}{\partial \sigma}$
Considering a single sample:

$$
 \frac{\partial}{\partial \sigma} [-(y \log \sigma + (1-y) \log(1-\sigma))] = -\left( \frac{y}{\sigma} - \frac{1-y}{1-\sigma} \right) = \frac{\sigma - y}{\sigma(1-\sigma)} 
 $$

### Part B: $\frac{\partial \sigma}{\partial z}$
As noted earlier:

$$ \frac{\partial \sigma}{\partial z} = \sigma(1-\sigma) $$

### Part C: Partial Derivatives of Net Input
Since $z = \sum w_j x_j + b$:

1. With respect to weight $w_j$:  $\frac{\partial z}{\partial w_j} = x_j$
2. With respect to bias $b$: $\frac{\partial z}{\partial b} = 1$

### Combining Parts (Single Sample):
For the weight $w_j$:

$$
 \frac{\partial J^{(i)}}{\partial w_j} = \left( \frac{\sigma(z^{(i)}) - y^{(i)}}{\sigma(z^{(i)})(1-\sigma(z^{(i)}))} \right) \cdot \left( \sigma(z^{(i)})(1-\sigma(z^{(i)})) \right) \cdot x_j^{(i)} = (\sigma(z^{(i)}) - y^{(i)}) x_j^{(i)} 
 $$

For the bias $b$:

$$
 \frac{\partial J^{(i)}}{\partial b} = \left( \frac{\sigma(z^{(i)}) - y^{(i)}}{\sigma(z^{(i)})(1-\sigma(z^{(i)}))} \right) \cdot \left( \sigma(z^{(i)})(1-\sigma(z^{(i)})) \right) \cdot 1 = \sigma(z^{(i)}) - y^{(i)} 
$$

### Total Gradients for $n$ samples:

$$ 
\frac{\partial J}{\partial w_j} = \sum_{i=1}^n (\sigma(z^{(i)}) - y^{(i)}) x_j^{(i)} $$

$$
 \frac{\partial J}{\partial b} = \sum_{i=1}^n (\sigma(z^{(i)}) - y^{(i)}) 
 $$

In vector notation (as seen in `logisticreg.py`):

$$
 \nabla_w J = \mathbf{X}^T (\mathbf{\sigma} - \mathbf{y})
$$
$$
 \nabla_b J = \text{sum}(\mathbf{\sigma} - \mathbf{y})
$$

---

## 5. Update Rules

In Gradient Descent, we update parameters in the opposite direction of the gradient:

### Weights Update
$$ w_j := w_j - \eta \frac{\partial J}{\partial w_j} = w_j + \eta \sum_{i=1}^n (y^{(i)} - \sigma(z^{(i)})) x_j^{(i)} $$
### Bias Update
$$ b := b - \eta \frac{\partial J}{\partial b} = b + \eta \sum_{i=1}^n (y^{(i)} - \sigma(z^{(i)})) $$

This matches the implementation in the `fit` method:
```python
errors = (y - output)
self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
self.b_ += self.eta * errors.mean()
```
*(Note: `errors.mean()` is equivalent to `errors.sum() / X.shape[0]`, which computes the average gradient for the bias.)*
