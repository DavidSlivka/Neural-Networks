#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>

#include "Vector.h"
#include "constants.h"
#include "Matrix.h"
/**
 * @file DataStorage.h
 * @brief Lightweight DataFrame-like handler for tabular datasets in C++.
 *
 * Provides:
 * - Reading CSV files into memory
 * - Statistical operations: mean, stddev, describe
 * - Normalization: Min-Max scaling, Z-Score standardization
 * - Data cleaning: drop missing values, fill missing values
 * - One-hot encoding for categorical columns
 * - Train-test splitting
 * - Conversion into Vector format for machine learning
 *
 * Example usage:
 * @code
 * DataStorage ds;
 * ds.readCsv("data.csv", ';');
 * ds.normalizeMinMax(0); // Normalize first column
 * auto [train, test] = ds.trainTestSplit(0.8f);
 * @endcode
 */
class DataStorage {
public:
    // -------- File Reading --------

    /**
     * @brief Reads a CSV file into memory.
     * @param filename Path to CSV file.
     * @param delimiter Field delimiter (e.g., ',' or ';').
     */
    void readCsv(const std::string& filename, char delimiter);

    // -------- Statistical Functions --------

    /**
     * @brief Computes mean for each column.
     * @return Vector of mean values.
     */
    std::vector<float> mean() const;

    /**
     * @brief Computes standard deviation for each column.
     * @return Vector of standard deviations.
     */
    std::vector<float> stddev() const;

    // -------- Normalization Functions --------

    /**
     * @brief Min-Max normalizes a specific column.
     * @param column_index Index of column to normalize.
     */
    void normalizeMinMax(int column_index);

    /**
     * @brief Z-Score standardizes a specific column.
     * @param column_index Index of column to normalize.
     */
    void normalizeZScore(int column_index);

    // -------- Data Transformation --------

    /**
     * @brief One-hot encodes a categorical column.
     * @param column_index Index of column to encode.
     */
    void oneHotEncodeColumn(int column_index);

    /**
     * @brief Converts all columns into Vectors.
     * @return Vector of Vectors (1D).
     */
    std::vector<Vector<float>> toVectors() const;

    // -------- Accessors / Getters --------

    /**
     * @brief Access full dataset.
     */
    const std::vector<std::vector<float>>& getData() const;

    /**
     * @brief Access column names.
     */
    const std::vector<std::string>& getHeaders() const;

    // -------- Viewing Data --------

    /**
     * @brief Display first n rows.
     */
    void head(int n) const;

    /**
     * @brief Display last n rows.
     */
    void tail(int n) const;

    /**
     * @brief Print entire dataset (optionally limiting max rows).
     */
    void print(std::ostream& os = std::cout, int max_rows = 5) const;

    // -------- Metadata --------

    /**
     * @brief Get (rows, columns) shape of dataset.
     */
    std::pair<int, int> shape() const;

    /**
     * @brief Get column names.
     */
    std::vector<std::string> columns() const;

    /**
     * @brief Get data types (all are "float" currently).
     */
    std::vector<std::string> dtypes() const;

    // -------- Data Summary --------

    /**
     * @brief Describe statistics (mean, std, min, max) per column.
     */
    void describe() const;

    // -------- Data Cleaning --------

    /**
     * @brief Remove rows with missing (NaN) values.
     */
    void dropna();

    /**
     * @brief Fill missing (NaN) values with a constant.
     * @param value Value to fill.
     */
    void fillna(float value);

    /**
     * @brief Drop a specific column.
     * @param column Name of the column to drop.
     */
    void drop(const std::string& column);

    /**
     * @brief Rename a column.
     * @param old_name Current name.
     * @param new_name New name.
     */
    void rename(const std::string& old_name, const std::string& new_name);

    // -------- Statistical Matrices --------

    /**
     * @brief Compute Pearson correlation matrix.
     * @return Matrix of correlations.
     */
    Matrix<float> correlationMatrix() const;

    /**
     * @brief Compute covariance matrix.
     * @return Matrix of covariances.
     */
    Matrix<float> covarianceMatrix() const;

    // -------- Dataset Splitting --------

    /**
     * @brief Split dataset into train and test sets.
     * @param train_ratio Ratio of data to use for training (0-1).
     * @return Pair (train_data, test_data).
     */
    std::pair<DataStorage, DataStorage> trainTestSplit(float train_ratio) const;

    /**
     * @brief Split dataset into Input and Target data.
     * @param input_end index of the last column that should be added to Input Data.
     * @param target_col index of column of the Target Data.
     * @return Pair (Input, Target).
     */
    std::pair<std::vector<Vector<float>>, std::vector<Vector<float>>> toInputsTargets(int input_end, int target_col) const;


    std::pair<float, float> getMinMaxValues() { return std::make_pair(min_value_, max_value_); }

private:
    static constexpr int PRECISION = 100; ///< Precision for rounding matrix outputs
    std::vector<std::string> column_names_; ///< Column names
    std::vector<std::vector<float>> data_; ///< Data stored column-wise
    float max_value_ = 0;
    float min_value_ = 0;
};


void DataStorage::readCsv(const std::string& filename, char delimeter) {
    std::ifstream file(filename);
    std::string line;
    bool header_read = false;

    if (!file.good()) {
        std::cout << "Error reading the file: " << filename << NEW_LINE;
        return;
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;
        while (std::getline(ss, cell, delimeter)) {
            row.push_back(cell);
        }
        if (!header_read) {
            column_names_ = row;
            data_.resize(column_names_.size());
            header_read = true;
        }
        else {
            for (size_t i = 0; i < column_names_.size(); ++i) {
                if (i < row.size() && !row[i].empty()) {
                    data_[i].push_back(std::stof(row[i]));
                }
                else {
                    data_[i].push_back(std::numeric_limits<float>::quiet_NaN());
                }
            }
        }
    }
}

std::vector<float> DataStorage::mean() const {
    if (data_.empty()) {
        return {};
    }

    int cols = data_[0].size();
    std::vector<float> result(cols, 0);
    for (const auto& row : data_) {
        for (int j = 0; j < cols; ++j) {
            result[j] += row[j];
        }
    }

    for (float& v : result) {
        v /= data_.size();
    }

    return result;
}

std::vector<float> DataStorage::stddev() const {
    auto m = mean();
    int cols = m.size();
    std::vector<float> result(cols, 0);
    for (const auto& row : data_) {
        for (int j = 0; j < cols; ++j) {
            result[j] += (row[j] - m[j]) * (row[j] - m[j]);
        }
    }

    for (float& v : result) {
        v = std::sqrt(v / data_.size());
    }

    return result;
}

void DataStorage::normalizeMinMax(int column_index) {
    if (column_index < 0 || column_index >= data_.size()) {
        return;
    }

    float min_val = data_[column_index][0];
    float max_val = data_[column_index][0];

    for (float val : data_[column_index]) {
        if (val < min_val) {
            min_val = val;
        }
        if (val > max_val) {
            max_val = val;
        }
    }

    float range = max_val - min_val;
    if (range == 0.0f) {
        range = 1.0f;
    }

    for (float& val : data_[column_index]) {
        val = (val - min_val) / range;
    }
    min_value_ = min_val;
    max_value_ = max_val;
}

void DataStorage::normalizeZScore(int column_index) {
    if (column_index < 0 || column_index >= data_.size()) {
        return;
    }

    float sum = 0.0f;
    for (float val : data_[column_index]) {
        sum += val;
    }
    float mean = sum / data_[column_index].size();

    float variance = 0.0f;
    for (float val : data_[column_index]) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data_[column_index].size();
    float std_dev = std::sqrt(variance);
    if (std_dev == 0.0f) {
        std_dev = 1.0f;
    }

    for (float& val : data_[column_index]) {
        val = (val - mean) / std_dev;
    }
}

typename std::vector<Vector<float>> DataStorage::toVectors() const {
    std::vector<Vector<float>> result;
    for (const auto& col : data_) {
        result.emplace_back(col);
    }

    return result;
}

void DataStorage::oneHotEncodeColumn(int column_index) {
    if (column_index < 0 || column_index >= data_.size()) {
        return;
    }

    std::unordered_map<float, int> mapping;
    for (const auto& value : data_[column_index]) {
        if (mapping.find(value) == mapping.end()) {
            mapping[value] = mapping.size();  
        }
    }
    int num_classes = mapping.size();

    std::vector<std::vector<float>> new_columns(num_classes, std::vector<float>(data_[column_index].size(), 0.0f));

    for (size_t i = 0; i < data_[column_index].size(); ++i) {
        float original_value = data_[column_index][i];
        int class_id = mapping[original_value];
        new_columns[class_id][i] = 1.0f;
    }

    data_.erase(data_.begin() + column_index);
    column_names_.erase(column_names_.begin() + column_index);

    for (int c = 0; c < num_classes; ++c) {
        data_.insert(data_.begin() + column_index + c, std::move(new_columns[c]));
        column_names_.insert(column_names_.begin() + column_index + c, "onehot_" + std::to_string(c));
    }
}


const std::vector<std::vector<float>>& DataStorage::getData() const { 
    return data_;
}

const std::vector<std::string>& DataStorage::getHeaders() const { 
    return column_names_;
}

void DataStorage::head(int n) const {
    for (const auto& name : column_names_) {
        std::cout << name << TABULATOR;
    }

    std::cout << NEW_LINE;

    n = data_.empty() ? 0 : std::min(n, (int)data_[0].size());
    for (size_t i = 0; i < n; ++i) {
        for (const auto& col : data_) {
            if (i < col.size()) {
                if (std::isnan(col[i])) {
                    std::cout << NOT_A_NUMBER << TABULATOR;
                }
                else {
                    std::cout << col[i] << TABULATOR;
                }
            }
        }
        std::cout << NEW_LINE;
    }
}

void DataStorage::tail(int n) const {
    for (const auto& name : column_names_) {
        std::cout << name << TABULATOR;
    }

    std::cout << NEW_LINE;

    size_t total = data_.empty() ? 0 : data_[0].size();
    n = std::min(n, (int)total);
    size_t start = total - n;

    for (size_t i = start; i < total; ++i) {
        std::cout << i+1 << TABULATOR;
        for (const auto& col : data_) {
            if (std::isnan(col[i])) {
                std::cout << NOT_A_NUMBER << TABULATOR;
            }
            else {
                std::cout << col[i] << TABULATOR;
            }
        }
        std::cout << NEW_LINE;
    }
}

std::pair<int, int> DataStorage::shape() const {
    if (data_.empty()) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(static_cast<int>(data_[0].size()), static_cast<int>(data_.size()));
}

std::vector<std::string> DataStorage::columns() const {
    return column_names_;
}

std::vector<std::string> DataStorage::dtypes() const {
    return std::vector<std::string>(data_.size(), "float");
}

void DataStorage::describe() const {
    for (size_t j = 0; j < data_.size(); ++j) {
        const auto& col = data_[j];
        float sum = 0, sq_sum = 0;
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        int count = 0;

        for (float val : col) {
            if (!std::isnan(val)) {
                sum += val;
                sq_sum += val * val;
                if (val < min) {
                    min = val;
                }
                if (val > max) {
                    max = val;
                }
                count++;
            }
        }
        if (count == 0) {
            continue;
        }

        float mean = sum / count;
        float stddev = std::sqrt(sq_sum / count - mean * mean);
        std::cout << column_names_[j] << ": mean=" << mean << ", std=" << stddev << ", min=" << min << ", max=" << max << NEW_LINE;
    }
}

void DataStorage::dropna() {
    if (data_.empty()) {
        return;
    }

    size_t num_rows = data_[0].size();
    std::vector<bool> keep_row(num_rows, true);

    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < data_.size(); ++j) {
            if (std::isnan(data_[j][i])) {
                keep_row[i] = false;
                break;
            }
        }
    }

    for (auto&& column : data_) {
        std::vector<float> new_col;
        for (size_t i = 0; i < num_rows; ++i) {
            if (keep_row[i]) {
                new_col.push_back(column[i]);
            }
        }
        column = std::move(new_col);
    }
}

void DataStorage::fillna(float value) {
    for (auto&& column : data_) {
        for (auto&& val : column) {
            if (std::isnan(val)) {
                val = value;
            }
        }
    }
}

void DataStorage::drop(const std::string& column) {
    auto it = std::find(column_names_.begin(), column_names_.end(), column);
    if (it != column_names_.end()) {
        int idx = it - column_names_.begin();
        column_names_.erase(it);
        data_.erase(data_.begin() + idx);
    }
}

void DataStorage::rename(const std::string& old_name, const std::string& new_name) {
    auto it = std::find(column_names_.begin(), column_names_.end(), old_name);
    if (it != column_names_.end()) {
        *it = new_name;
    }
}

Matrix<float> DataStorage::correlationMatrix() const {
    size_t n = data_.size();
    Matrix<float> corr(n, n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum_i = 0;
            float sum_j = 0;
            int count = 0;
            for (size_t k = 0; k < data_[i].size(); ++k) {
                if (!std::isnan(data_[i][k]) && !std::isnan(data_[j][k])) {
                    sum_i += data_[i][k];
                    sum_j += data_[j][k];
                    count++;
                }
            }
            float mean_i = sum_i / count;
            float mean_j = sum_j / count;

            float num = 0; 
            float den_i = 0;
            float den_j = 0;
            for (size_t k = 0; k < data_[i].size(); ++k) {
                if (!std::isnan(data_[i][k]) && !std::isnan(data_[j][k])) {
                    float di = data_[i][k] - mean_i;
                    float dj = data_[j][k] - mean_j;
                    num += di * dj;
                    den_i += di * di;
                    den_j += dj * dj;
                }
            }

            auto total = num / std::sqrt(den_i * den_j);
            total = round(total * PRECISION) / PRECISION;
            corr.setValue(i, j, total);
        }
    }
    return corr;
}

Matrix<float> DataStorage::covarianceMatrix() const {
    size_t n = data_.size();
    Matrix<float> cov(n, n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum_i = 0;
            float sum_j = 0;
            int count = 0;
            for (size_t k = 0; k < data_[i].size(); ++k) {
                if (!std::isnan(data_[i][k]) && !std::isnan(data_[j][k])) {
                    sum_i += data_[i][k];
                    sum_j += data_[j][k];
                    count++;
                }
            }
            float mean_i = sum_i / count;
            float mean_j = sum_j / count;

            float sum = 0;
            for (size_t k = 0; k < data_[i].size(); ++k) {
                if (!std::isnan(data_[i][k]) && !std::isnan(data_[j][k])) {
                    sum += (data_[i][k] - mean_i) * (data_[j][k] - mean_j);
                }
            }
            auto total = sum / (count - 1);
            total = round(total * PRECISION) / PRECISION;
            cov.setValue(i, j, total);
        }
    }
    return cov;
}

std::pair<DataStorage, DataStorage> DataStorage::trainTestSplit(float train_ratio) const {
    DataStorage train, test;
    if (data_.empty()) {
        return std::make_pair(train, test);
    }

    size_t total = data_[0].size();
    size_t split = static_cast<size_t>(total * train_ratio);
    train.column_names_ = column_names_;
    test.column_names_ = column_names_;
    train.data_.resize(data_.size());
    test.data_.resize(data_.size());

    for (size_t j = 0; j < data_.size(); ++j) {
        train.data_[j].assign(data_[j].begin(), data_[j].begin() + split);
        test.data_[j].assign(data_[j].begin() + split, data_[j].end());
    }

    return std::make_pair(train, test);
}

std::pair<std::vector<Vector<float>>, std::vector<Vector<float>>> DataStorage::toInputsTargets(int input_end, int target_index) const {
    std::vector<Vector<float>> inputs, targets;
    int rows = data_.empty() ? 0 : data_[0].size();
    for (int i = 0; i < rows; ++i) {
        Vector<float> input(input_end);
        for (int j = 0; j < input_end; ++j) {
            input[j] = data_[j][i];
        }

        inputs.push_back(input);
        Vector<float> target(1);
        target[0] = data_[target_index][i];
        targets.push_back(target);
    }
    return std::make_pair(inputs, targets);
}

void DataStorage::print(std::ostream& os, int max_rows) const {
    for (const auto& name : column_names_) {
        std::cout << name << TABULATOR;

    }
    std::cout << NEW_LINE;
    if (data_.empty()) {
        return;
    }

    size_t num_rows = data_[0].size();
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < data_.size(); ++j) {
            float value = data_[j][i];
            if (std::isnan(value)) {
                std::cout << NOT_A_NUMBER <<TABULATOR;
            }
            else {
                std::cout << value << TABULATOR;
            }
        }
        std::cout << NEW_LINE;
    }
}

/**
 * @brief Overload operator<< for pretty printing DataStorage.
 */
std::ostream& operator<<(std::ostream& os, const DataStorage& data_storage) {
    data_storage.print(os);
    return os;
}


#endif // !DATA_STORAGE_H

