#ifndef MATRIX_HPP
#define MATRIX_HPP

/**
 * @file Matrix.h
 * @brief Defines the Matrix class for linear algebra operations.
 */

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <cassert>
#include <stdexcept>
#include <type_traits>

#include "constants.h"
#include "Vector.h"


template<typename T> class Vector;

/**
 * @class Matrix
 * @brief Represents a 2D matrix stored in row-major order.
 *
 * Supports scalar operations, matrix addition/subtraction, dot products,
 * transposition, inversion, rank computation, and application of custom functions.
 *
 * @tparam DataType Type of the data (e.g., float, double)
 */
template<typename DataType>
class Matrix {
public:
    // --- Constructors ---
    /**
     * @brief Default constructor (creates empty matrix).
     */
    Matrix();

    /**
     * @brief Create a matrix of size (rows x cols) filled with specified value.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param val  Value to fill
     */
    Matrix(size_t rows, size_t cols, DataType val = 0.0);

    /**
     * @brief Create a matrix initialized with a 2D std::vector.
     * @param values 2D vector representing the matrix.
     */
    Matrix(const std::vector<std::vector<DataType>>& values);

    // --- Static Initializers ---
    /**
     * @brief Create a matrix with random values between min and max.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param min Minimum random value
     * @param max Maximum random value
     */
    static Matrix random(size_t rows, size_t cols, DataType min = -1.0, DataType max = 1.0);

    /**
     * @brief Create a matrix filled with zeros.
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * @brief Create a matrix filled with ones.
     */
    static Matrix ones(size_t rows, size_t cols);

    /**
     * @brief Create a diagonal matrix of specified size and value.
     */
    static Matrix diagonal(size_t size, DataType value);

    /**
     * @brief Create a diagonal matrix with ones on the diagonal.
     */
    static Matrix diagonalOnes(size_t size);

    /**
     * @brief Fill all elements with the same value.
     */
    void fill(DataType val);

    // --- Scalar Operations ---
    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator+(const U& scalar) const;

    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator-(const U& scalar) const;

    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator*(const U& scalar) const;

    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator/(const U& scalar) const;

    // --- Matrix Operations ---
    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator+(const Matrix<U>& other) const;

    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator-(const Matrix<U>& other) const;

    /**
     * @brief Perform matrix multiplication (dot product).
     */
    template<typename U>
    Matrix<std::common_type_t<DataType, U>> dot(const Matrix<U>& other) const;

    // --- Vector Operations ---
    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator+(const Vector<U>& vec) const;

    template<typename U>
    Matrix<std::common_type_t<DataType, U>> operator-(const Vector<U>& vec) const;

    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator*(const Vector<U>& vec) const;

    // --- Comparison ---
    bool operator==(const Matrix& other) const;
    bool isEqual(const Matrix& other) const;

    // --- Transformations ---
    /**
     * @brief Apply a function element-wise to the matrix.
     */
    Matrix apply(const std::function<double(double)>& func) const;

    /**
     * @brief Return the transpose of the matrix.
     */
    Matrix transpose() const;

    /**
     * @brief Add a bias column (constant value) to the matrix.
     */
    Matrix addBiasColumn(DataType biasValue = 1.0) const;

    // --- Properties ---
    bool isSquare() const;
    bool isSymmetric() const;

    // --- Accessors ---
    DataType& at(size_t row, size_t col);
    const DataType& at(size_t row, size_t col) const;

    /**
     * @brief Set a specific value at given (row, column).
     */
    void setValue(size_t row, size_t col, DataType value);

    /**
     * @brief Get the number of rows.
     */
    size_t getRows() const { return rows_; }

    /**
     * @brief Get the number of columns.
     */
    size_t getCols() const { return cols_; }

    // --- Copy / Move ---
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;
    Matrix(const Matrix& other);

    /**
     * @brief Cast matrix to a different data type.
     */
    template<typename U>
    Matrix(const Matrix<U>& other);

    /**
     * @brief Print the matrix (pretty format).
     */
    void print(std::ostream& os = std::cout) const;

    // --- Advanced ---
    /**
     * @brief Compute the rank of the matrix.
     */
    int rank() const;

    /**
     * @brief Compute the inverse of the matrix (if square and non-singular).
     */
    Matrix<std::common_type_t<DataType, double>> inverse() const;

protected:
    size_t rows_; ///< Number of rows
    size_t cols_; ///< Number of columns
    std::vector<DataType> data_; ///< Data stored row-major order

    size_t index(size_t row, size_t col) const { return row * cols_ + col; }
};

template<typename DataType>
inline Matrix<DataType>::Matrix() : rows_(0), cols_(0) {}

template<typename DataType>
inline Matrix<DataType>::Matrix(size_t rows, size_t cols, DataType val) : rows_(rows), cols_(cols) {
    data_.resize(rows * cols, val);
}

template<typename DataType>
inline Matrix<DataType>::Matrix(const std::vector<std::vector<DataType>>& values) {
    rows_ = values.size();
    cols_ = values[0].size();
    data_.reserve(rows_ * cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            data_.push_back(values[i][j]);
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::random(size_t rows, size_t cols, DataType min, DataType max) {
    Matrix<DataType> m(rows, cols);
    std::mt19937 gen(44);
    std::uniform_real_distribution<> dis(min, max);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m.data_[m.index(i, j)] = dis(gen);
        }
    }
    return m;
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::diagonal(size_t size, DataType value) {
    Matrix<DataType> result = Matrix::zeros(size, size);
    for (size_t i = 0; i < size; ++i) {
        result.setValue(i, i, value);
    }
    return result;
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::diagonalOnes(size_t size) {
    return diagonal(size, 1);
}

template<typename DataType>
inline void Matrix<DataType>::fill(DataType val) {
    std::fill(data_.begin(), data_.end(), val);
}

#pragma region MATRIX_OPERATIONS

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator+(const Matrix<U>& other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    using ResultType = std::common_type_t<DataType, U>;
    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<ResultType>(data_[index(i, j)]) + static_cast<ResultType>(other.data_[other.index(i, j)]);
        }
    }
    return result;
}

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator-(const Matrix<U>& other) const {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    using ResultType = std::common_type_t<DataType, U>;
    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<ResultType>(data_[index(i, j)]) - static_cast<ResultType>(other.data_[other.index(i, j)]);
        }
    }
    return result;
}


template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::dot(const Matrix<U>& other) const {
    assert(cols_ == other.rows_);
    using ResultType = std::common_type_t<DataType, U>;
    Matrix<ResultType> result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            for (size_t k = 0; k < cols_; ++k) {
                result.data_[result.index(i, j)] += static_cast<ResultType>(data_[index(i, k)]) * static_cast<ResultType>(other.data_[other.index(k, j)]);
            }
        }
    }
    return result;
}

#pragma endregion MATRIX_OPERATIONS

template<typename DataType>
inline bool Matrix<DataType>::operator==(const Matrix<DataType>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }

    for (int i = 0; i < rows_ * cols_; ++i) {
        if (std::abs(data_[i] - other.data_[i]) > EPSILON) {
            return false;
        }
    }

    return true;
}

template<typename DataType>
inline bool Matrix<DataType>::isEqual(const Matrix<DataType>& other) const {
    return *this == other;
}


template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::apply(const std::function<double(double)>& func) const {
    Matrix<DataType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = func(data_[index(i, j)]);
        }
    }
    return result;
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::transpose() const {
    Matrix<DataType> result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(j, i)] = data_[index(i, j)];
        }
    }
    return result;
}

template<typename DataType>
inline Matrix<DataType> Matrix<DataType>::addBiasColumn(DataType biasValue) const {
    Matrix<DataType> result(rows_, cols_ + 1);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = data_[index(i, j)];
        }
        result.data_[result.index(i, cols_)] = biasValue;
    }
    return result;
}

template<typename DataType>
inline bool Matrix<DataType>::isSquare() const {
    return rows_ == cols_;
}

template<typename DataType>
inline bool Matrix<DataType>::isSymmetric() const {
    if (!isSquare()) { 
        return false; 
    }

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = i + 1; j < cols_; ++j) {
            if (data_[index(i, j)] != data_[index(j, i)]) {
                return false;
            }
        }
    }
    return true;
}

template<typename DataType>
inline DataType& Matrix<DataType>::at(size_t row, size_t col) {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[index(row, col)];
}

template<typename DataType>
inline const DataType& Matrix<DataType>::at(size_t row, size_t col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[index(row, col)];
}

template<typename DataType>
inline void Matrix<DataType>::setValue(size_t row, size_t col, DataType value) {
    data_[index(row, col)] = value;
}

template<typename DataType>
Matrix<DataType>::Matrix(Matrix<DataType>&& other) noexcept : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_)) {
    other.rows_ = 0;
    other.cols_ = 0;
}

template<typename DataType>
inline Matrix<DataType>::Matrix(const Matrix<DataType>& other) : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {}


template<typename DataType>
Matrix<DataType>& Matrix<DataType>::operator=(Matrix<DataType>&& other) noexcept {
    if (this != &other) {
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = std::move(other.data_);
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

// for converting Matrix DataType to another
template<typename DataType>
template<typename U>
inline Matrix<DataType>::Matrix(const Matrix<U>& other) {
    rows_ = other.getRows();
    cols_ = other.getCols();
    data_.resize(rows_ * cols_);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[index(i, j)] = static_cast<DataType>(other.at(i, j));
        }
    }
}

template<typename DataType>
void Matrix<DataType>::print(std::ostream& os) const {
    os << SQUARE_OPENING_BRACKET << SPACE;
    for (size_t i = 0; i < rows_; ++i) {
        os << SQUARE_OPENING_BRACKET;
        for (size_t j = 0; j < cols_; ++j) {
            os << at(i, j);
            if (j < cols_ - 1) os << COMMA << SPACE;
        }
        os << SQUARE_CLOSEING_BRACKET;
        if (i < rows_ - 1) os << COMMA << NEW_LINE << SPACE << SPACE;
    }
    os << SPACE << SQUARE_CLOSEING_BRACKET;
}

template<typename DataType>
inline std::ostream& operator<<(std::ostream& os, const Matrix<DataType>& matrix) {
    matrix.print(os);
    return os;
}

template<typename DataType>
inline int Matrix<DataType>::rank() const {
    using FloatType = std::common_type_t<DataType, double>;
    Matrix<FloatType> temp(*this);
    int rank = 0;
    std::vector<bool> row_selected(rows_, false);

    for (size_t i = 0; i < cols_; ++i) {
        size_t j;
        for (j = 0; j < rows_; ++j) {
            if (!row_selected[j] && std::abs(temp.at(j, i)) > EPSILON) {
                break;
            }
        }
        if (j != rows_) {
            ++rank;
            row_selected[j] = true;
            for (size_t p = 0; p < rows_; ++p) {
                if (p != j) {
                    auto factor = temp.at(p, i) / temp.at(j, i);
                    for (size_t k = 0; k < cols_; ++k) {
                        temp.at(p, k) -= factor * temp.at(j, k);
                    }
                }
            }
        }
    }
    return rank;
}

template<typename DataType>
inline Matrix<std::common_type_t<DataType, double>> Matrix<DataType>::inverse() const {
    using FloatType = std::common_type_t<DataType, double>;

    if (!isSquare()) {
        throw std::runtime_error("Only square matrices can be inverted");
    }

    size_t n = rows_;
    Matrix<FloatType> A(*this);
    Matrix<FloatType> I = Matrix<FloatType>::diagonalOnes(n);

    for (size_t i = 0; i < n; ++i) {
        auto pivot = A.at(i, i);
        if (std::abs(pivot) < EPSILON) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }

        for (size_t j = 0; j < n; ++j) {
            A.at(i, j) /= pivot;
            I.at(i, j) /= pivot;
        }

        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                auto factor = A.at(k, i);
                for (size_t j = 0; j < n; ++j) {
                    A.at(k, j) -= factor * A.at(i, j);
                    I.at(k, j) -= factor * I.at(i, j);
                }
            }
        }
    }

    return I;
}


#pragma region SCALAR_OPERATORS

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator+(const U& scalar) const {
    Matrix<std::common_type_t<DataType, U>> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<std::common_type_t<DataType, U>>(data_[index(i, j)]) + scalar;
        }
    }

    return result;
}

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator-(const U& scalar) const {
    Matrix<std::common_type_t<DataType, U>> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<std::common_type_t<DataType, U>>(data_[index(i, j)]) - scalar;
        }
    }

    return result;
}

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator*(const U& scalar) const {
    Matrix<std::common_type_t<DataType, U>> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<std::common_type_t<DataType, U>>(data_[index(i, j)])* scalar;
        }
    }

    return result;
}

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator/(const U& scalar) const {
    Matrix<std::common_type_t<DataType, U>> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[result.index(i, j)] = static_cast<std::common_type_t<DataType, U>>(data_[index(i, j)]) / scalar;
        }
    }


    return result;
}

template<typename U, typename DataType>
Matrix<std::common_type_t<U, DataType>> operator+(const U& scalar, const Matrix<DataType>& matrix) {
    return matrix + scalar;
}

template<typename U, typename DataType>
Matrix<std::common_type_t<U, DataType>> operator-(const U& scalar, const Matrix<DataType>& matrix) {
    Matrix<std::common_type_t<U, DataType>> result(matrix.rows_, matrix.cols_);
    for (size_t i = 0; i < matrix.rows_; ++i) {
        for (size_t j = 0; j < matrix.cols_; ++j) {
            result(i, j) = scalar - static_cast<std::common_type_t<U, DataType>>(matrix(i, j));
        }
    }

    return result;
}

template<typename U, typename DataType>
Matrix<std::common_type_t<U, DataType>> operator*(const U& scalar, const Matrix<DataType>& matrix) {
    return matrix * scalar;
}

#pragma endregion SCALAR_OPERATIONS


#pragma region VECTOR_OPERATIONS
template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator+(const Vector<U>& vec) const {
    using ResultType = std::common_type_t<DataType, U>;

    if (cols_ != vec.size()) {
        throw std::invalid_argument("Vector size must match number of columns for row-wise addition.");
    }

    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(i, j) = static_cast<ResultType>(at(i, j)) + static_cast<ResultType>(vec[j]);
        }
    }

    return result;
}

template<typename DataType, typename U>
Matrix<std::common_type_t<DataType, U>> operator+(const Vector<U>& vec, const Matrix<DataType>& matrix) {
    return matrix + vec;
}

template<typename DataType>
template<typename U>
inline Matrix<std::common_type_t<DataType, U>> Matrix<DataType>::operator-(const Vector<U>& vec) const {
    using ResultType = std::common_type_t<DataType, U>;

    if (cols_ != vec.size()) {
        throw std::invalid_argument("Vector size must match number of columns for row-wise subtraction.");
    }

    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(i, j) = static_cast<ResultType>(at(i, j)) - static_cast<ResultType>(vec[j]);
        }
    }

    return result;
}


template<typename DataType, typename U>
Matrix<std::common_type_t<DataType, U>> operator-(const Vector<U>& vec, const Matrix<DataType>& matrix) {
    using ResultType = std::common_type_t<DataType, U>;

    if (matrix.getCols() != vec.size()) {
        throw std::invalid_argument("Vector size must match number of columns for row-wise subtraction.");
    }

    Matrix<ResultType> result(matrix.getRows(), matrix.getCols());
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result.at(i, j) = static_cast<ResultType>(vec[j]) - static_cast<ResultType>(matrix.at(i, j));
        }
    }

    return result;
}

template<typename DataType>
template<typename U>
inline Vector<std::common_type_t<DataType, U>> Matrix<DataType>::operator*(const Vector<U>& vec) const {
    using ResultType = std::common_type_t<DataType, U>;

    if (cols_ != vec.size()) {
        throw std::invalid_argument("Vector size must match number of columns for row-wise multiplication.");
    }

    Vector<ResultType> result(rows_, ResultType{});
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(i) += static_cast<ResultType>(at(i, j)) * static_cast<ResultType>(vec[j]);
        }
    }

    return result;

}

#pragma endregion VECTOR_OPERATIONS


#endif // MATRIX_HPP