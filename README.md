# Neural Networks & Data Processing Library

This project implements the basic building blocks for a Neural Networks, focusing on **matrix operations**, **vector operations**, **layers**, **activation functions**, **loss functions**, and **simple dataset handling** — without external dependencies, similar to python libraries NumPy, PyTorch and pandas.  


---


# Example Usage

```cpp
// EXAMPLE ON XOR PROBLEM

//////////////////////////////////////////////////////////////////////////////////////////
// XOR cannot be classified using perceptron, that why we need a Multi Layer Perceptron //
//////////////////////////////////////////////////////////////////////////////////////////

using DataType = float;

DataStorage df;
df.readCsv("inputs/testXOR.csv", SEMICOLON);
df.print();

int input_columns = 2;
int target_column = 2;

std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out = df.toInputsTargets(input_columns, target_column);
auto inputs = out.first;
auto targets = out.second;

// model with 1 hippen layer with 2 neurons
Sequential<DataType> model;
model.add(std::make_shared<Dense<DataType>>(2, 2));        // input to hidden
model.add(std::make_shared<ActivationSigmoid<DataType>>());
model.add(std::make_shared<Dense<DataType>>(2, 1));        // hidden to output
model.add(std::make_shared<ActivationSigmoid<DataType>>());

BinaryCrossEntropyLoss<DataType> loss;

float lr = 0.15f;
size_t epochs = 10'000;

// training
for (size_t epoch = 0; epoch < epochs; ++epoch) {
    float total_loss = 0.0f;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto prediction = model.forward(inputs[i]);
        total_loss += loss.forward(prediction, targets[i])[0];

        auto grad_loss = loss.backward(prediction, targets[i]);
        model.backward(grad_loss, lr);
    }
}

// final prediction
for (size_t i = 0; i < inputs.size(); ++i) {
    auto out = model.forward(inputs[i]);
    std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
        << "Predicted: " << out[0] << " vs. " << "Actual Value: " << targets[i][0] << NEW_LINE;
}
```
---


# Inputs

In directory `./inputs/`

For showcase there are 3 files:
-  `testXOR.csv` - file with inputs and targets for XOR
-  `small_housing.csv` - file with 13 rows containing sq. footage, number of rooms, age of the property and the price (actual target to be predicted)
-  `housing.csv` - real dataset with 1000 entries containing information as longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households,median_income, median_house_value (target to be predicted)
  

---

# Tests 

In directory `./tests/`

There are a few tests for each class testing its basic functionality 

---

# Key Components Overview

| Module            | Purpose |
|-------------------|---------|
| [Matrix](#matrix)         | A templated 2D matrix class (row-major storage) with scalar/matrix/vector operations, transpose, rank, inverse. |
| [Vector](#vector)         | A 1D vector class supporting row/column orientation and basic vector algebra (dot, outer product, scalar ops). |
| [DataStorage](#datastorage)    | Simple DataFrame-like tabular data handler: CSV reading, normalization, missing value handling, splits. |
| [Layer](#layer)          | Base class for neural network layers + Sequential and Dense layers. |
| [Activation Functions](#activation-functions) | Standard nonlinearities: Sigmoid, ReLU, Tanh, LeakyReLU, Swish, Softplus, Softmax. |
| [Loss Functions](#loss-functions) | Implementations of Binary Cross-Entropy, Mean Squared Error, Mean Absolute Error, and Categorical Cross-Entropy. |
| [Perceptron](#perceptron) | Implementation of simple perceptron algorithm

---

# Matrix

Implemented in `Matrix.h`.

### Purpose
Efficient, flexible handling of 2D arrays of numbers.  
**Supports:**
- Scalar addition, subtraction, multiplication, division
- Matrix-matrix addition, subtraction, dot product
- Matrix-vector operations
- Random initialization, zeros, ones, diagonal matrices
- Transpose
- Inverse (Gaussian elimination)
- Rank calculation

### Important Functions

| Function               | Purpose |
|-------------------------|---------|
| `random(rows, cols, min, max)` | Randomly fill a matrix with values in [min, max]. |
| `zeros(rows, cols)`     | Create a matrix filled with zeros. |
| `ones(rows, cols)`      | Create a matrix filled with ones. |
| `transpose()`           | Returns the transposed matrix. |
| `dot(other)`            | Matrix multiplication (dot product). |
| `addBiasColumn(bias=1.0)` | Adds a bias (constant 1) as a last column. |
| `inverse()`             | Inverts a square matrix. |
| `rank()`                | Calculates the matrix's rank. |
| `operator==(other)`     | Compares two matrices element-wise. |

---

# Vector

Implemented in `Vector.h`.

### Purpose
Represent mathematical **vectors**, supporting **Row** and **Column** orientation.

**Supports:**
- Scalar addition, subtraction, multiplication, division
- Vector addition, subtraction
- Dot product
- Outer product
- Reshape (transpose between row and column)

### Important Functions

| Function                  | Purpose |
|----------------------------|---------|
| `transpose()`              | Switch between Row and Column vectors. |
| `dot(other)`               | Computes dot product (row × column). |
| `outer(other)`             | Computes outer product (column × row) → Matrix. |
| `resize(new_size)`         | Resize the vector. |
| `operator+(scalar)`        | Adds a scalar to each element. |
| `operator+(other)`         | Adds two vectors element-wise. |

---

# DataStorage

Implemented in `DataStorage.h`.

### Purpose
Simple replacement for pandas-like tabular data handling: CSV reading, cleaning, normalization.

### Important Functions

| Function                  | Purpose |
|----------------------------|---------|
| `readCsv(filename, delimiter)` | Reads data from CSV into internal storage. |
| `normalizeMinMax(column_idx)`  | Rescales column to [0, 1]. |
| `normalizeZScore(column_idx)`  | Rescales column to mean 0, stddev 1. |
| `oneHotEncodeColumn(column_idx)` | One-hot encodes a categorical column. |
| `head(n)`, `tail(n)`       | Display first/last n rows. |
| `dropna()`                 | Drops rows containing NaNs. |
| `fillna(value)`            | Replaces NaNs with a specific value. |
| `drop(column_name)`        | Deletes a column by name. |
| `rename(old, new)`         | Renames a column. |
| `trainTestSplit(ratio)`    | Splits the dataset into training and testing parts. |
| `correlationMatrix()`      | Calculates correlation matrix. |
| `covarianceMatrix()`       | Calculates covariance matrix. |

---

# Layer

Implemented in `Layer.h`.

### Purpose
Abstract base class defining the basic interface for neural network layers.

- **`forward(input)`**: computes output of the layer.
- **`backward(grad_output, learning_rate)`**: computes gradients and updates parameters.

### Provided Layers

| Class                   | Purpose |
|--------------------------|---------|
| `Sequential`             | Stack of layers executed sequentially. |
| `Dense`                  | Fully connected (linear) layer: $y = W^T x + b$. |
| `Dropout`                | Randomly zeros-out activations during training. |

---

# Activation Functions

Implemented in `ActivationLayers.h`.

Each activation layer implements `forward()` and `backward()` methods for automatic differentiation.

| Activation               | Formula | Derivative |
|---------------------------|---------|------------|
| **Sigmoid**               | $\frac{1}{1 + e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| **ReLU**                  | $\max(0, x)$ | $1$ if $x > 0$ else $0$ |
| **Tanh**                  | $\tanh(x)$ | $1 - \tanh(x)^2$ |
| **LeakyReLU**             | $x$ if $x>0$, else $\alpha x$ | $1$ or $\alpha$ |
| **Swish**                 | $x \cdot \text{sigmoid}(x)$ | $\frac{(e^{-x}(x+1) + 1)}{({1+e^{-x}})^2}$ |
| **Softplus**              | $\log(1 + e^x)$ | $\text{sigmoid}(x)$ |
| **Softmax**               | Converts logits into probabilities (outputs sum to 1). |

---

# Loss Functions

Implemented in `LossFunctions.h`.

### Provided Losses

| Loss                          | Formula |
|--------------------------------|---------|
| **Binary Cross-Entropy**       |  $-[y\log(p) + (1-y)\log(1-p)]$ |
| **Mean Squared Error**         |  $\frac{1}{2}(y - p)^2$ |
| **Mean Absolute Error**        |  $\|y - p\|$ |
| **Categorical Cross-Entropy**  |  $-\sum y\log(p)$ |

Each loss provides:
- **`forward(predictions, targets)`**: compute the per-sample loss.
- **`backward(predictions, targets)`**: compute the gradient wrt predictions.

---

# Perceptron
class implements a basic perceptron model
### Purpose 
Purpose: Linear decision boundary


| Function      |       Purpose |
|---------------|---------|
|`train`        | Trains the perceptron on a set of inputs and labels. |
|`predict`      | Predicts the binary class (0 or 1) for a given input. |

