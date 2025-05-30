/**
 * @file Activation_Layers.h
 * @brief Implementation of common activation functions as neural network layers.
 *
 * Each activation layer inherits from the base Layer<DataType> class and implements:
 * - forward(): activation function application
 * - backward(): derivative for backpropagation
 *
 * Supported activations:
 * - Sigmoid
 * - ReLU
 * - Tanh
 * - Leaky ReLU
 * - Swish
 * - Softplus
 * - Softmax
 *
 * Example usage:
 * @code
 * ActivationReLU<float> relu;
 * Vector<float> input = { -1.0f, 2.0f, 3.0f };
 * Vector<float> output = relu.forward(input);
 * @endcode
 */

#ifndef ACTIVATION_LAYERS_H
#define ACTIVATION_LAYERS_H

#include "Layer.h" // Base Layer class

 // ============================= ActivationSigmoid ============================= //

 /**
  * @class ActivationSigmoid
  * @brief Sigmoid activation layer.
  *
  * Applies element-wise: \f$ \sigma(x) = \frac{1}{1 + e^{-x}} \f$
  */
template<typename DataType>
class ActivationSigmoid : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> output_cache_; ///< Cached output for backward pass
};

// -- Definitions --

/**
 * @brief Computes forward pass using Sigmoid.
 */
template<typename DataType>
Vector<DataType> ActivationSigmoid<DataType>::forward(const Vector<DataType>& input) {
    output_cache_ = input;
    for (size_t i = 0; i < input.size(); ++i) {
        output_cache_[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }

    return output_cache_;
}

/**
 * @brief Computes backward pass (gradient of Sigmoid).
 */
template<typename DataType>
Vector<DataType> ActivationSigmoid<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * output_cache_[i] * (1 - output_cache_[i]);
    }

    return grad_input;
}

// ============================= ActivationReLU ============================= //

/**
 * @class ActivationReLU
 * @brief ReLU (Rectified Linear Unit) activation layer.
 *
 * Applies element-wise: \f$ ReLU(x) = \max(0, x) \f$
 */
template<typename DataType>
class ActivationReLU : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> input_cache_; ///< Cached input for backward pass
};

// -- Definitions --

/**
 * @brief Computes forward pass using ReLU.
 */
template<typename DataType>
Vector<DataType> ActivationReLU<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    Vector<DataType> output = input;
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = std::max<DataType>(0, input[i]);
    }

    return output;
}

/**
 * @brief Computes backward pass (gradient of ReLU).
 */
template<typename DataType>
Vector<DataType> ActivationReLU<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input = grad_output;
    for (size_t i = 0; i < grad_input.size(); ++i) {
        grad_input[i] = input_cache_[i] > 0 ? grad_input[i] : 0;
    }

    return grad_input;
}

// ============================= ActivationTanh ============================= //

/**
 * @class ActivationTanh
 * @brief Hyperbolic tangent activation layer.
 *
 * Applies element-wise: \f$ tanh(x) \f$
 */
template<typename DataType>
class ActivationTanh : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> output_cache_; ///< Cached output for backward pass
};

// -- Definitions --

/**
 * @brief Computes forward pass using tanh.
 */
template<typename DataType>
Vector<DataType> ActivationTanh<DataType>::forward(const Vector<DataType>& input) {
    output_cache_.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output_cache_[i] = std::tanh(input[i]);
    }

    return output_cache_;
}

/**
 * @brief Computes backward pass (gradient of tanh).
 */
template<typename DataType>
Vector<DataType> ActivationTanh<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (1 - output_cache_[i] * output_cache_[i]);
    }

    return grad_input;
}

// ============================= ActivationLeakyReLU ============================= //

/**
 * @class ActivationLeakyReLU
 * @brief Leaky ReLU activation layer.
 *
 * Allows small gradient (alpha) for negative inputs.
 */
template<typename DataType>
class ActivationLeakyReLU : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> input_cache_; ///< Cached input
    const DataType alpha = 0.01;    ///< Slope for negative input
};

// -- Definitions --

/**
 * @brief Computes forward pass using Leaky ReLU.
 */
template<typename DataType>
Vector<DataType> ActivationLeakyReLU<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    Vector<DataType> output = input;
    for (int i = 0; i < output.size(); ++i) {
        output[i] = input[i] > 0 ? input[i] : alpha * input[i];
    }

    return output;
}

/**
 * @brief Computes backward pass (gradient of Leaky ReLU).
 */
template<typename DataType>
Vector<DataType> ActivationLeakyReLU<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input = grad_output;
    for (int i = 0; i < grad_input.size(); ++i) {
        grad_input[i] = input_cache_[i] > 0 ? grad_output[i] : alpha * grad_output[i];
    }

    return grad_input;
}

// ============================= ActivationSwish ============================= //

/**
 * @class ActivationSwish
 * @brief Swish activation layer.
 *
 * Applies: \f$ Swish(x) = x * sigmoid(x) \f$
 */
template<typename DataType>
class ActivationSwish : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> input_cache_;   ///< Cached input
    Vector<DataType> sigmoid_cache_; ///< Cached sigmoid(x)
};

// -- Definitions --

/**
 * @brief Computes forward pass using Swish.
 */
template<typename DataType>
Vector<DataType> ActivationSwish<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    sigmoid_cache_.resize(input.size());
    Vector<DataType> output(input.size());
    for (int i = 0; i < input.size(); ++i) {
        sigmoid_cache_[i] = 1.0f / (1.0f + std::exp(-input[i]));
        output[i] = input[i] * sigmoid_cache_[i];
    }

    return output;
}

/**
 * @brief Computes backward pass (gradient of Swish).
 */
template<typename DataType>
Vector<DataType> ActivationSwish<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (sigmoid_cache_[i] + input_cache_[i] * sigmoid_cache_[i] * (1 - sigmoid_cache_[i]));
    }

    return grad_input;
}

// ============================= ActivationSoftplus ============================= //

/**
 * @class ActivationSoftplus
 * @brief Softplus activation layer.
 *
 * Smooth version of ReLU: \f$ Softplus(x) = \log(1 + e^x) \f$
 */
template<typename DataType>
class ActivationSoftplus : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> input_cache_; ///< Cached input
};

// -- Definitions --

/**
 * @brief Computes forward pass using Softplus.
 */
template<typename DataType>
Vector<DataType> ActivationSoftplus<DataType>::forward(const Vector<DataType>& input) {
    input_cache_ = input;
    Vector<DataType> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::log(1 + std::exp(input[i]));
    }

    return output;
}

/**
 * @brief Computes backward pass (gradient of Softplus).
 */
template<typename DataType>
Vector<DataType> ActivationSoftplus<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (1.0 / (1.0 + std::exp(-input_cache_[i])));
    }

    return grad_input;
}

// ============================= ActivationSoftmax ============================= //

/**
 * @class ActivationSoftmax
 * @brief Softmax activation layer.
 *
 * Converts logits into a probability distribution.
 */
template<typename DataType>
class ActivationSoftmax : public Layer<DataType> {
public:
    Vector<DataType> forward(const Vector<DataType>& input) override;
    Vector<DataType> backward(const Vector<DataType>& grad_output, float) override;

private:
    Vector<DataType> output_cache_; ///< Cached output (softmax probabilities)
};

// -- Definitions --

/**
 * @brief Computes forward pass using Softmax.
 */
template<typename DataType>
Vector<DataType> ActivationSoftmax<DataType>::forward(const Vector<DataType>& input) {
    DataType max_val = *std::max_element(input.begin(), input.end());
    DataType sum_exp = 0;
    output_cache_.resize(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        output_cache_[i] = std::exp(input[i] - max_val);
        sum_exp += output_cache_[i];
    }

    for (size_t i = 0; i < output_cache_.size(); ++i) {
        output_cache_[i] /= sum_exp;
    }

    return output_cache_;
}

/**
 * @brief Computes backward pass (Jacobian-vector product for Softmax).
 */
template<typename DataType>
Vector<DataType> ActivationSoftmax<DataType>::backward(const Vector<DataType>& grad_output, float) {
    Vector<DataType> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = 0;
        for (size_t j = 0; j < grad_output.size(); ++j) {
            if (i == j) {
                grad_input[i] += grad_output[j] * output_cache_[i] * (1 - output_cache_[j]);
            }
            else {
                grad_input[i] -= grad_output[j] * output_cache_[i] * output_cache_[j];
            }
        }
    }
    return grad_input;
}

#endif // ACTIVATION_LAYERS_H