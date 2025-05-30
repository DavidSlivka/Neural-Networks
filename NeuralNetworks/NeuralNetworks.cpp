#include <iostream>
#include "ActivationLayer.h"
#include "constants.h"
#include "DataStorage.h"
#include "Layer.h"
#include "LossFunctions.h"
#include "Matrix.h"
#include "Vector.h"


int testXOR() {
    using DataType = float;

    DataStorage df;
    df.readCsv("inputs/testXOR.csv", SEMICOLON);
    df.print();

    int input_columns = 2;
    int target_column = 2;

    std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out = df.toInputsTargets(input_columns, target_column);
    auto inputs = out.first;
    auto targets = out.second;


    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(2, 2));        // input to hidden
    model.add(std::make_shared<ActivationSigmoid<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(2, 1));        // hidden to output
    model.add(std::make_shared<ActivationSigmoid<DataType>>());

    BinaryCrossEntropyLoss<DataType> loss;

    float lr = 0.15f;
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
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss << NEW_LINE;
        }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = model.forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] -> "
            << "Predicted: " << out[0] << " vs. " << "Actual Value: " << targets[i][0] << NEW_LINE;
    }
    return 0;
}


int testSmallHousing() {
    using DataType = float;
    DataStorage df;
    df.readCsv("inputs/small_housing.csv", SEMICOLON);
    df.head(5);
    df.tail(5);

    df.normalizeZScore(0);
    df.normalizeZScore(1);
    df.normalizeZScore(2);

    df.normalizeMinMax(3);

    float split_point = 0.615f;
    std::pair<DataStorage, DataStorage> traintest = df.trainTestSplit(split_point);
    auto training_data = traintest.first;
    auto testing_data = traintest.second;

    int input_columns = 3;
    int target_column = 3;

    std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out = training_data.toInputsTargets(input_columns, target_column);
    auto inputs = out.first;
    auto targets = out.second;

    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(3, 32));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(32, 1));
    model.add(std::make_shared<ActivationSigmoid<DataType>>());


    MeanSquaredErrorLoss<DataType> loss_fn;

    for (int epoch = 0; epoch < 100; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {

            Vector<DataType> output = model.forward(inputs[i]);

            for (auto& val : output) {
                if (std::isnan(val)) {
                    std::cout << "NaN detected in output at epoch " << epoch << ", input index " << i << NEW_LINE;
                    return -1;
                }
            }

            Vector<DataType> grad = loss_fn.backward(output, targets[i]);

            model.backward(grad, 0.01f);

            total_loss += loss_fn.forward(output, targets[i])[0];

            for (auto& val : grad) {
                if (std::isnan(val)) {
                    std::cout << "NaN detected in gradients at epoch " << epoch << ", input index " << i << NEW_LINE;
                    return -1;
                }
            }
        }

        if (epoch % 500 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << NEW_LINE;
    }


    std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out2 = testing_data.toInputsTargets(input_columns, target_column);
    auto test_inputs = out2.first;
    auto test_targets = out2.second;

    std::pair<float, float> minmax = df.getMinMaxValues();

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Vector<DataType> output = model.forward(test_inputs[i]);
        float predicted_price = output[0] * (minmax.second - minmax.first) + minmax.first;
        std::cout << "Predicted Price: " << predicted_price << " vs. Correct Price: " << test_targets[i][0] * (minmax.second - minmax.first) + minmax.first << NEW_LINE;
    }

    std::cout << "Price prediction test completed." << NEW_LINE;
    return 0;
}

int testHousing() {
    using DataType = float;
    DataStorage df;
    df.readCsv("inputs/housing.csv", COMMA);
    df.head(5);
    df.tail(5);

    df.normalizeZScore(0);
    df.normalizeZScore(1);
    df.normalizeZScore(2);
    df.normalizeZScore(3);
    df.normalizeZScore(4);
    df.normalizeZScore(5);
    df.normalizeZScore(6);
    df.normalizeZScore(7);

    df.normalizeMinMax(8);

    float split_point = 0.8f;
    std::pair<DataStorage, DataStorage> traintest = df.trainTestSplit(split_point);
    auto training_data = traintest.first;
    auto testing_data = traintest.second;

    int input_columns = 8;
    int target_column = 8;

    std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out = training_data.toInputsTargets(input_columns, target_column);
    auto input = out.first;
    auto target = out.second;

    Sequential<DataType> model;
    model.add(std::make_shared<Dense<DataType>>(8, 64));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    auto dropout1 = std::make_shared<Dropout<DataType>>(0.5f);
    model.add(dropout1);
    model.add(std::make_shared<Dense<DataType>>(64, 64));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(64, 64));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    model.add(std::make_shared<Dense<DataType>>(64, 32));
    model.add(std::make_shared<ActivationReLU<DataType>>());
    auto dropout2 = std::make_shared<Dropout<DataType>>(0.5f);
    model.add(dropout2);
    model.add(std::make_shared<Dense<DataType>>(32, 1));
    model.add(std::make_shared<ActivationSigmoid<DataType>>());

    MeanSquaredErrorLoss<DataType> loss_fn;

    float lr = 0.005f;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < input.size(); ++i) {
            dropout1->set_training(true);
            dropout2->set_training(true);

            Vector<DataType> output = model.forward(input[i]);

            for (auto& val : output) {
                if (std::isnan(val)) {
                    val = 0;
                    std::cout << "NaN detected in output at epoch " << epoch << ", input index " << i << NEW_LINE;
                    return -1;
                }
            }

            Vector<DataType> grad = loss_fn.backward(output, target[i]);

            model.backward(grad, lr); 

            total_loss += loss_fn.forward(output, target[i])[0];

            for (auto& val : grad) {
                if (std::isnan(val)) {
                    std::cout << "NaN detected in gradients at epoch " << epoch << ", input index " << i << NEW_LINE;
                    return -1;
                }
            }
        }

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / input.size() << NEW_LINE;
    }

    std::pair<std::vector<Vector<DataType>>, std::vector<Vector<DataType>>> out2 = testing_data.toInputsTargets(input_columns, target_column);
    auto test_inputs = out2.first;
    auto test_targets = out2.second;

    dropout1->set_training(false);
    dropout2->set_training(false);

    std::pair<float, float> minmax = df.getMinMaxValues();

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Vector<DataType> output = model.forward(test_inputs[i]);
        float predicted_price = output[0] * (minmax.second - minmax.first) + minmax.first;
        std::cout << "Predicted Price: " << predicted_price << " vs. Correct Price: " << test_targets[i][0] * (minmax.second - minmax.first) + minmax.first << NEW_LINE;
    }

    std::cout << "Price prediction with dropout test completed." << NEW_LINE;
    return 0;
}


int main()
{
    testXOR();
    testSmallHousing();
    //testHousing();
    return 0;
}