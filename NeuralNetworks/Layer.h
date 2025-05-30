/**
 * @file Layer.h
 * @brief Defines the basic components of a neural network architecture:
 *        abstract Layer interface, Sequential container, Dense (fully connected) layer, and Dropout regularization.
 *
 * Example usage:
 * @code
 * Sequential<float> model;
 * model.add(std::make_shared<Dense<float>>(3, 5));
 * model.add(std::make_shared<Dropout<float>>(0.5));
 * @endcode
 */

#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <cmath>
#include <vector>
#include <random>

#include "Matrix.h"
#include "Vector.h"
#include "constants.h"

 // ============================= Layer (Base Class) ============================= //

 /**
  * @class Layer
  * @brief Abstract base class for all neural network layers.
  *
  * Requires derived classes to implement:
  * - forward()
  * - backward()
  */
template<typename DataType>
class Layer {
public:
    /**
     * @brief Forward pass: computes layer output given input.
     * @param input Input vector.
     * @return Output vector after transformation.
     */
    virtual Vector<DataType> forward(const Vector<DataType>& input) = 0;

    /**
     * @brief Backward pass: computes gradients and updates parameters if needed.
     * @param grad_output Gradient w.r.t output.
     * @param learning_rate Learning rate for parameter update.
     * @return Gradient w.r.t input.
     */
    virtual Vector<DataType> backward(const Vector<DataType>& grad_output, float learning_rate) = 0;

    virtual ~Layer() = default;
};

// ============================= Sequential ============================= //

/**
 * @class Sequential
 * @brief Container for stacking multiple layers sequentially.
 */
template<typename DataType>
class Sequential {
public:
    /**
     * @brief Add a new layer to the model.
     * @param layer Shared pointer to the layer.
     */
    void add(std::shared_ptr<Layer<DataType>> layer);

    /**
     * @brief Forward pass through all layers.
     * @param input Input vector.
     * @return Output after passing through all layers.
     */
    Vector<DataType> forward(const Vector<DataType>& input);

    /**
     * @brief Backward pass through all layers in reverse order.
     * @param grad_output Gradient from next layer/loss function.
     * @param learning_rate Learning rate.
     * @return Gradient w.r.t input.
     */
    Vector<DataType> backward(const Vector<DataType>& grad_output, float learning_rate);

private:
    std::vector<std::shared_ptr<Layer<DataType>>> layers_; ///< Stack of layers
};

// --- Sequential Definitions ---

template<typename DataType>
void Sequential<DataType>::add(std::shared_ptr<Layer<DataType>> layer) {
    layers_.push_back(layer);
}

template<typename DataType>
Vector<DataType> Sequential<DataType>::forward(const Vector<DataType>& input) {
    Vector<DataType> x = input;
    for (auto& layer : layers_) {
        x = layer->forward(x);
    }

    return x;
}

template<typename DataType>
Vector<DataType> Sequential<DataType>::backward(const Vector<DataType>& grad_output, float learning_rate) {
    Vector<DataType> grad = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad, learning_rate);
    }

    return grad;
}

// ============================= Dense ============================= //

/**
 * @class Dense
 * @brief Fully connected linear layer.
 *
 * Transformation: \f$ y = W^T x + b \f$
 */
template<typename DataType>
class Dense : public Layer<DataType> {
public:
    /**
     * @brief Construct a Dense layer.
     * @param input_size Number of input features.
     * @param output_size Number of output features.
     */
    Dense(int input_size, int output_size);

    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float learning_rate) override;

private:
    Matrix<DataType> weights_; ///< Weight matrix (input_size × output_size)
    Vector<DataType> biases_;  ///< Bias vector (output_size)
    Vector<DataType> input_cache_; ///< Cached input for backpropagation
};

// --- Dense Definitions ---

template<typename DataType>
Dense<DataType>::Dense(int input_size, int output_size)
    : weights_(Matrix<DataType>::random(input_size, output_size, -1.0, 1.0)), biases_(Vector<DataType>(output_size)) {}

template<typename DataType>
Vector<DataType> Dense<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    Vector<DataType> output = weights_.transpose() * input;
    output = output + biases_;
    return output;
}

template<typename DataType>
Vector<DataType> Dense<DataType>::backward(const Vector<DataType>& grad_output, float learning_rate) {
    int input_size = input_cache_.size();
    int output_size = grad_output.size();

    Matrix<DataType> grad_weights(input_size, output_size);
    Vector<DataType> grad_input(input_size);

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            grad_weights.at(i, j) = input_cache_[i] * grad_output[j];
            grad_input[i] += weights_.at(i, j) * grad_output[j];
        }
    }

    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights_.at(i, j) -= learning_rate * grad_weights.at(i, j);
        }
    }

    for (int j = 0; j < output_size; ++j) {
        biases_[j] -= learning_rate * grad_output[j];
    }

    return grad_input;
}

// ============================= Dropout ============================= //

/**
 * @class Dropout
 * @brief Dropout layer to prevent overfitting by randomly setting activations to zero during training.
 */
template<typename DataType>
class Dropout : public Layer<DataType> {
public:
    /**
     * @brief Construct a Dropout layer.
     * @param rate Probability of dropping a unit (0-1).
     */
    Dropout(float rate);

    /**
     * @brief Set training mode (true) or inference mode (false).
     */
    void set_training(bool mode);

    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float learning_rate) override;

private:
    float dropout_rate_; ///< Dropout probability
    std::vector<bool> mask_; ///< Mask of kept/dropped neurons
    std::mt19937 rng_; ///< Random number generator
    std::bernoulli_distribution dist_; ///< Sampling distribution
    bool training_; ///< Training or inference mode
    Vector<DataType> input_cache_; ///< Cached input for backward pass
};

// --- Dropout Definitions ---

template<typename DataType>
Dropout<DataType>::Dropout(float rate)
    : dropout_rate_(rate), rng_(std::random_device{}()), dist_(1.0f - rate), training_(true) {}

template<typename DataType>
void Dropout<DataType>::set_training(bool mode) {
    training_ = mode;
}

template<typename DataType>
Vector<DataType> Dropout<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    Vector<DataType> output(input.size());
    mask_.resize(input.size());

    if (training_) {
        for (int i = 0; i < input.size(); ++i) {
            mask_[i] = dist_(rng_);
            output[i] = mask_[i] ? input[i] / (1.0f - dropout_rate_) : 0.0f;
        }
    }
    else {
        for (int i = 0; i < input.size(); ++i) {
            output[i] = input[i];
        }
    }

    return output;
}

template<typename DataType>
Vector<DataType> Dropout<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = training_ ? (mask_[i] ? grad_output[i] / (1.0f - dropout_rate_) : 0.0f) : grad_output[i];
    }

    return grad_input;
}


#endif // LAYER_H
