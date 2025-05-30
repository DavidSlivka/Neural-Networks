/**
 * @file LossFunctions.h
 * @brief Common loss functions for neural network training.
 *
 * Supported Losses:
 * - Binary Cross Entropy
 * - Mean Squared Error
 * - Mean Absolute Error
 * - Categorical Cross Entropy
 *
 * Each loss function provides:
 * - forward() for computing the loss value
 * - backward() for computing the gradient w.r.t the predictions
 *
 * Example usage:
 * @code
 * BinaryCrossEntropyLoss<float> loss_fn;
 * auto loss = loss_fn.forward(predictions, targets);
 * auto grad = loss_fn.backward(predictions, targets);
 * @endcode
 */

#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H


#include "Vector.h"

 // ======================= Binary Cross Entropy Loss ======================= //

 /**
  * @class BinaryCrossEntropyLoss
  * @brief Binary classification loss function.
  *
  * Loss:
  * \f$ L = - (y \log(\hat{y}) + (1-y) \log(1-\hat{y})) \f$
  */
template<typename DataType>
class BinaryCrossEntropyLoss {
public:
    /**
     * @brief Forward pass: computes binary cross entropy loss.
     * @param y_pred Predicted probabilities (0-1).
     * @param y_true True labels (0 or 1).
     * @return Per-sample loss vector.
     */
    Vector<DataType> forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);

    /**
     * @brief Backward pass: computes gradient of loss w.r.t y_pred.
     * @param y_pred Predicted probabilities.
     * @param y_true True labels.
     * @return Gradient vector.
     */
    Vector<DataType> backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
};


template<typename DataType>
Vector<DataType> BinaryCrossEntropyLoss<DataType>::forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> loss(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        loss[i] = -(y_true[i] * std::log(y_pred[i] + EPSILON) + (1 - y_true[i]) * std::log(1 - y_pred[i] + EPSILON));
    }

    return loss;
}

template<typename DataType>
Vector<DataType> BinaryCrossEntropyLoss<DataType>::backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> grad(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        grad[i] = (y_pred[i] - y_true[i]) / ((y_pred[i] + EPSILON) * (1 - y_pred[i] + EPSILON));
    }

    return grad;
}

// ======================= Mean Squared Error Loss ======================= //

/**
 * @class MeanSquaredErrorLoss
 * @brief Loss function for regression: penalizes squared differences.
 *
 * Loss:
 * \f$ L = \frac{1}{2}(y - \hat{y})^2 \f$
 */
template<typename DataType>
class MeanSquaredErrorLoss {
public:
    Vector<DataType> forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
    Vector<DataType> backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
};

template<typename DataType>
Vector<DataType> MeanSquaredErrorLoss<DataType>::forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> loss(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        loss[i] = 0.5f * std::pow(y_pred[i] - y_true[i], 2);
    }

    return loss;
}

template<typename DataType>
Vector<DataType> MeanSquaredErrorLoss<DataType>::backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> grad(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        grad[i] = y_pred[i] - y_true[i];
    }

    return grad;
}

// ======================= Mean Absolute Error Loss ======================= //

/**
 * @class MeanAbsoluteErrorLoss
 * @brief Loss function for regression: penalizes absolute differences.
 *
 * Loss:
 * \f$ L = |y - \hat{y}| \f$
 */
template<typename DataType>
class MeanAbsoluteErrorLoss {
public:
    Vector<DataType> forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
    Vector<DataType> backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
};

template<typename DataType>
Vector<DataType> MeanAbsoluteErrorLoss<DataType>::forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> loss(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        loss[i] = std::abs(y_pred[i] - y_true[i]);
    }

    return loss;
}

template<typename DataType>
Vector<DataType> MeanAbsoluteErrorLoss<DataType>::backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> grad(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        grad[i] = (y_pred[i] > y_true[i]) ? 1.0f : ((y_pred[i] < y_true[i]) ? -1.0f : 0.0f);
    }

    return grad;
}

// ======================= Categorical Cross Entropy Loss ======================= //

/**
 * @class CategoricalCrossEntropyLoss
 * @brief Loss for multi-class classification tasks (one-hot encoded targets).
 *
 * Loss:
 * \f$ L = -\sum y_i \log(\hat{y}_i) \f$
 */
template<typename DataType>
class CategoricalCrossEntropyLoss {
public:
    Vector<DataType> forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
    Vector<DataType> backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true);
};

template<typename DataType>
Vector<DataType> CategoricalCrossEntropyLoss<DataType>::forward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> loss(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
        loss[i] = -y_true[i] * std::log(y_pred[i] + EPSILON);
    }

    return loss;
}

template<typename DataType>
Vector<DataType> CategoricalCrossEntropyLoss<DataType>::backward(const Vector<DataType>& y_pred, const Vector<DataType>& y_true) {
    assert(y_pred.size() == y_true.size());
    Vector<DataType> grad(y_pred.size());
    for (int i = 0; i < y_pred.size(); ++i) {
      
        grad[i] = -y_true[i] / (y_pred[i] + EPSILON);
    }

    return grad;
}

#endif // LOSS_FUNCTIONS_H