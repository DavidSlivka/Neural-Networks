/**
 * @file Perceptron.h
 * @brief Simple single-layer perceptron implementation for binary classification.
 *
 * This class implements a basic perceptron model with:
 * - Linear decision boundary
 * - Step activation function
 * - Online learning (weight updates per sample)
 *
 * Example usage:
 * @code
 * #include "Perceptron.h"
 *
 * Perceptron p(2);
 * Vector<float> sample = {1.0f, -1.0f};
 * int output = p.predict(sample);
 * @endcode
 */

#ifndef PERCEPTRON_H
#define PERCEPTRON_H


#include "Vector.h"

using DataType = float;

/**
 * @class Perceptron
 * @brief Basic single-layer perceptron model.
 */
class Perceptron {
public:
    /**
     * @brief Constructs a perceptron with a given input size and learning rate.
     */
    Perceptron(int inputSize, DataType learningRate = 0.01);

    /**
     * @brief Predicts the binary class (0 or 1) for a given input.
     */
    int predict(const Vector<DataType>& inputs) const;

    /**
     * @brief Trains the perceptron on a set of inputs and labels.
     */
    void train(const Vector<Vector<DataType>>& inputs, const Vector<int>& labels, int epochs);

private:
    Vector<DataType> weights_;    ///< Weights associated with each input feature.
    DataType bias_;               ///< Bias term.
    DataType learningRate_;       ///< Learning rate for updates.

    /**
     * @brief Activation function (step function).
     */
    int activation(DataType x) const;
};

// ======================== Implementation ========================

/**
 * @brief Perceptron constructor: initializes weights and bias to zero.
 */
Perceptron::Perceptron(int inputSize, DataType learningRate)
    : weights_(inputSize, 0.0f), bias_(0.0f), learningRate_(learningRate) {}

/**
 * @brief Step activation function.
 */
int Perceptron::activation(DataType x) const {
    return x > 0 ? 1 : 0;
}

/**
 * @brief Predicts the output for a single input vector.
 *
 * Computes the weighted sum plus bias and applies activation function.
 */
int Perceptron::predict(const Vector<DataType>& inputs) const {
    DataType sum = bias_;
    for (size_t i = 0; i < inputs.size(); ++i)
        sum += weights_[i] * inputs[i];
    return activation(sum);
}

/**
 * @brief Trains the perceptron using the perceptron learning rule.
 *
 * Updates weights and bias after each training sample based on error.
 */
void Perceptron::train(const Vector<Vector<DataType>>& trainingData,
    const Vector<int>& labels,
    int epochs) {
    for (int e = 0; e < epochs; ++e) {
        for (size_t i = 0; i < trainingData.size(); ++i) {
            int prediction = predict(trainingData[i]);
            int error = labels[i] - prediction;
            // Weight update: w_j += learning_rate * error * x_j
            for (size_t j = 0; j < weights_.size(); ++j)
                weights_[j] += learningRate_ * error * trainingData[i][j];
            // Bias update
            bias_ += learningRate_ * error;
        }
    }
}

#endif // PERCEPTRON_H
