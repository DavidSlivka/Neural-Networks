#ifndef TEST_LAYER_H
#define TEST_LAYER_H

#include <cassert>
#include <iostream>
#include <cmath>
#include "../Layer.h"
#include "../ActivationLayer.h"
#include "../LossFunctions.h"


using DataType = float;

void test_xor() {
    using DataType = float;

    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(2, 2));        // input to hidden
    model.add(std::make_shared<ActivationSigmoid<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(2, 1));        // hidden to output
    model.add(std::make_shared<ActivationSigmoid<DataType>>());

    BinaryCrossEntropyLoss<DataType> loss;

    std::vector<Vector<DataType>> inputs = {
        Vector<DataType>({0, 0}),
        Vector<DataType>({0, 1}),
        Vector<DataType>({1, 0}),
        Vector<DataType>({1, 1})
    };

    std::vector<Vector<DataType>> targets = {
        Vector<DataType>({0}),
        Vector<DataType>({1}),
        Vector<DataType>({1}),
        Vector<DataType>({0})
    };

    float lr = 0.1f;
    size_t epochs = 10'000;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto prediction = model.forward(inputs[i]);
            total_loss += loss.forward(prediction, targets[i])[0];

            auto grad_loss = loss.backward(prediction, targets[i]);
            model.backward(grad_loss, lr);
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = model.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << COMMA << inputs[i][1] << "] -> "
            << "Predicted: " << out[0] << std::endl;
    }
}

int testDropoutXOR() {
    using DataType = float;


    std::vector<Vector<DataType>> inputs = {
        Vector<DataType>({0, 0}),
        Vector<DataType>({0, 1}),
        Vector<DataType>({1, 0}),
        Vector<DataType>({1, 1})
    };

    std::vector<Vector<DataType>> targets = {
        Vector<DataType>({0}),
        Vector<DataType>({1}),
        Vector<DataType>({1}),
        Vector<DataType>({0})
    };


    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(2, 8));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    auto dropout = std::make_shared<Dropout<DataType>>(0.5f);
    model.add(dropout);
    model.add(std::make_shared<Dense<DataType>>(8, 1));
    model.add(std::make_shared<ActivationSigmoid<DataType>>());

    BinaryCrossEntropyLoss<DataType> loss_fn;

    for (int epoch = 0; epoch < 10'000; ++epoch) {
        float total_loss = 0.0f;
        for (int i = 0; i < 4; ++i) {
            dropout->set_training(true);
            Vector<DataType> output = model.forward(inputs[i]);
            Vector<DataType> grad = loss_fn.backward(output, targets[i]);
            model.backward(grad, 0.1f);
            total_loss += loss_fn.forward(output, targets[i])[0];
        }
        if (epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / 4 << NEW_LINE;
    }

    dropout->set_training(false);
    for (int i = 0; i < 4; ++i) {
        Vector<DataType> output = model.forward(inputs[i]);
        std::cout << "Input: (" << inputs[i][0] << COMMA << inputs[i][1] << ") -> Output: " << output[0] << NEW_LINE;
    }

    std::cout << "Dropout XOR test completed." << NEW_LINE;
    return 0;
}

// sin(x) + cos(y) + x*y*z
int testDropoutFunctionApproximation() {
    using DataType = float;

    std::vector<Vector<DataType>> inputs;
    std::vector<Vector<DataType>> targets;

    //  samples
    for (float x = 0.1f; x <= 1.0f; x += 0.2f) {
        for (float y = 0.1f; y <= 1.0f; y += 0.2f) {
            for (float z = 0.1f; z <= 1.0f; z += 0.2f) {
                inputs.push_back({ x, y, z });
                float t = std::sin(x) + std::cos(y) + x * y * z;
                float norm = (t + 3.0f) / 6.0f; // normalize to [0,1]
                targets.push_back({ norm });
            }
        }
    }

    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(3, 12));
    model.add(std::make_shared<ActivationTanh<DataType>>());
    auto dropout = std::make_shared<Dropout<DataType>>(0.2f);
    model.add(dropout);
    model.add(std::make_shared<Dense<DataType>>(12, 1));
    model.add(std::make_shared<ActivationSigmoid<DataType>>());

    MeanSquaredErrorLoss<DataType> loss_fn;

    for (int epoch = 0; epoch < 4000; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            dropout->set_training(true);
            Vector<DataType> output = model.forward(inputs[i]);
            Vector<DataType> grad = loss_fn.backward(output, targets[i]);
            model.backward(grad, 0.01f);
            total_loss += loss_fn.forward(output, targets[i])[0];
        }
        if (epoch % 500 == 0)
            std::cout << "[Function Approx] Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << NEW_LINE;
    }

    dropout->set_training(false);
    for (size_t i = 0; i < 5; ++i) {
        Vector<DataType> output = model.forward(inputs[i]);
        std::cout << "Eval Input: (" << inputs[i][0] << ", " << inputs[i][1] << ", " << inputs[i][2] << ") -> Predicted: " << output[0] << ", Target: " << targets[i][0] << NEW_LINE;
    }

    std::cout << "Function Approximation Dropout test completed.\n";
    return 0;
}


int testSoftmaxCategoricalCrossEntropy() {
    using DataType = float;

    // Synthetic dataset: 3 classes (one-hot encoded targets)
    std::vector<Vector<DataType>> inputs = {
        {0.1f, 0.2f, 0.7f}, {0.9f, 0.1f, 0.0f}, {0.3f, 0.4f, 0.3f},
        {0.6f, 0.3f, 0.1f}, {0.2f, 0.8f, 0.0f}, {0.4f, 0.5f, 0.1f}
    };

    // One-hot encoded targets
    std::vector<Vector<DataType>> targets = {
        {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
    };

    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(3, 5));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(5, 3));
    model.add(std::make_shared<ActivationSoftmax<DataType>>());

    CategoricalCrossEntropyLoss<DataType> loss_fn;

    for (int epoch = 0; epoch < 3000; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Vector<DataType> output = model.forward(inputs[i]);
            Vector<DataType> grad = loss_fn.backward(output, targets[i]);
            model.backward(grad, 0.05f);
            total_loss += loss_fn.forward(output, targets[i])[0];
        }
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << NEW_LINE;
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector<DataType> output = model.forward(inputs[i]);
        std::cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1] << ", " << inputs[i][2]
            << ") -> Output (Softmax): [" << output[0] << ", " << output[1] << ", " << output[2] << "]\n";
           
        int predicted_class = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        std::cout << "Predicted class: " << predicted_class << NEW_LINE;
    }

    std::cout << "Softmax + Categorical Cross Entropy test completed." << NEW_LINE;
    return 0;
}


#endif