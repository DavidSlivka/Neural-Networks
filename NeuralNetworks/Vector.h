#ifndef VECTOR_H
#define VECTOR_H

/**
 * @file Vector.h
 * @brief Defines the Vector class for mathematical vector operations.
 */

#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <variant>


#include "constants.h"
#include "Matrix.h"


template<typename T> class Matrix;

/**
 * @enum Orientation
 * @brief Specifies the orientation of a vector (Row or Column).
 */
enum class Orientation {
    Row,    ///< Row vector
    Column  ///< Column vector
};

/**
 * @class Vector
 * @brief Represents a mathematical vector with optional orientation.
 *
 * Supports scalar operations, vector addition/subtraction, dot product, outer product,
 * resizing, transposition, and type conversion.
 *
 * @tparam DataType Type of elements (e.g., float, double)
 */
template<typename DataType>
class Vector {
public:
    // --- Constructors ---
    /**
     * @brief Default constructor (empty vector, column-oriented).
     */
    Vector();

    /**
     * @brief Constructs a vector of given size and value.
     * @param size Number of elements
     * @param val Initial value
     * @param orient Orientation (Row or Column)
     */
    Vector(size_t size, DataType val = 0.0, Orientation orient = Orientation::Column);

    /**
     * @brief Constructs a vector of given size.
     */
    Vector(size_t size, Orientation orient);

    /**
     * @brief Constructs a vector from a std::vector.
     */
    Vector(const std::vector<DataType>& values, Orientation orient = Orientation::Column);

    /**
     * @brief Move constructor.
     */
    Vector(Vector&& other) noexcept;

    /**
     * @brief Move assignment operator.
     */
    Vector& operator=(Vector&& other) noexcept;

    /**
     * @brief Copy constructor.
     */
    Vector(const Vector& other);

    /**
     * @brief Copy assignment operator.
     */
    Vector& operator=(const Vector& other);

    /**
     * @brief Constructor from an initializer list.
     */
    Vector(std::initializer_list<DataType> list, Orientation orient = Orientation::Column);

    /**
     * @brief Constructor from nested initializer list (for nested vectors).
     */
    template<typename T = DataType>
    Vector(std::initializer_list<std::initializer_list<typename T::value_type>> list, Orientation orient = Orientation::Column);

    /**
     * @brief Type-conversion constructor from another Vector type.
     */
    template<typename U>
    Vector(const Vector<U>& other);

    // --- Element Access ---
    DataType& operator[](size_t index);
    const DataType& operator[](size_t index) const;

    /**
     * @brief Access element at index with bounds checking.
     */
    DataType& at(size_t i);

    /**
     * @brief Const version of at().
     */
    const DataType& at(size_t i) const;

    /**
     * @brief Get the size (number of elements).
     */
    size_t size() const;

    /**
     * @brief Get the orientation (Row or Column).
     */
    Orientation orientation() const;

    /**
     * @brief Transpose the orientation (Row -> Column, Column -> Row).
     */
    void transpose();

    /**
     * @brief Resize the vector.
     */
    void resize(size_t new_size);

    // --- Iterators ---
    typename std::vector<DataType>::iterator begin();
    typename std::vector<DataType>::iterator end();
    typename std::vector<DataType>::const_iterator begin() const;
    typename std::vector<DataType>::const_iterator end() const;

    // --- Scalar Operations ---
    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator+(const U& scalar) const;

    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator-(const U& scalar) const;

    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator*(const U& scalar) const;

    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator/(const U& scalar) const;

    // --- Vector Operations ---
    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator+(const Vector<U>& other) const;

    template<typename U>
    Vector<std::common_type_t<DataType, U>> operator-(const Vector<U>& other) const;

    /**
     * @brief Computes the dot product (Row . Column).
     */
    template<typename U>
    std::common_type_t<DataType, U> dot(const Vector<U>& other) const;

    /**
     * @brief Computes the outer product (Column . Row).
     */
    template<typename U>
    Matrix<std::common_type_t<DataType, U>> outer(const Vector<U>& other) const;

    // --- Other Utilities ---
    /**
     * @brief Check for equality (element-wise comparison with EPSILON).
     */
    bool operator==(const Vector& other) const;

    /**
     * @brief Pretty-print the vector.
     */
    void print(std::ostream& os = std::cout);

protected:
    std::vector<DataType> data_; ///< Underlying data storage
    Orientation orientation_;   ///< Row or Column orientation
};

template<typename DataType>
Vector<DataType>::Vector() : data_(), orientation_(Orientation::Column) {}

template<typename DataType>
Vector<DataType>::Vector(size_t size, Orientation orient) : data_(size), orientation_(orient) {}

template<typename DataType>
Vector<DataType>::Vector(size_t size, DataType val, Orientation orient): data_(size, val), orientation_(orient) {}

template<typename DataType>
Vector<DataType>::Vector(const std::vector<DataType>& values, Orientation orient): data_(values), orientation_(orient) {}

template<typename DataType>
Vector<DataType>::Vector(Vector<DataType>&& other) noexcept : data_(std::move(other.data_)), orientation_(other.orientation_) {}

template<typename DataType>
Vector<DataType>& Vector<DataType>::operator=(Vector<DataType>&& other) noexcept {
    if (this != &other) {
        orientation_ = (other.orientation());
        data_ = std::move(other.data_);
    }
    return *this;
}

template<typename DataType>
Vector<DataType>::Vector(const Vector<DataType>& other) : data_(other.data_), orientation_(other.orientation_) {}

template<typename DataType>
Vector<DataType>& Vector<DataType>::operator=(const Vector<DataType>& other) {
    if (this != &other) {
        data_ = other.data_;
        orientation_ = other.orientation_;
    }
    return *this;
}

template<typename DataType>
template<typename U>
Vector<DataType>::Vector(const Vector<U>& other) {
    orientation_ = other.orientation();

    data_.resize(other.size());
    for (size_t i = 0; i < other.size(); ++i) {
        data_[i] = static_cast<DataType>(other.at(i));
    }
}

template<typename DataType>
Vector<DataType>::Vector(std::initializer_list<DataType> list, Orientation orient)
    : data_(list), orientation_(orient) {}


template<typename DataType>
template<typename T>
Vector<DataType>::Vector(std::initializer_list<std::initializer_list<typename T::value_type>> list, Orientation orient) {
    static_assert(std::is_same<DataType, Vector<typename T::value_type>>::value,
        "Nested initializer list constructor is only for Vector<Vector<T>>");

    orientation_ = orient;
    data_.reserve(list.size());
    for (const auto& sublist : list) {
        data_.emplace_back(Vector<typename T::value_type>(sublist, orient));
    }
}

template<typename DataType>
size_t Vector<DataType>::size() const { return static_cast<size_t>(data_.size()); }

template<typename DataType>
DataType& Vector<DataType>::at(size_t i) {
    if (i < 0 || i >= size()) throw std::out_of_range("Vector index out of bounds");
    return data_[i];
}

template<typename DataType>
const DataType& Vector<DataType>::at(size_t i) const {
    if (i < 0 || i >= size()) throw std::out_of_range("Vector index out of bounds");
    return data_[i];
}

template<typename DataType>
typename Orientation Vector<DataType>::orientation() const {
    return orientation_;
}

template<typename DataType>
void Vector<DataType>::transpose() {
    orientation_ = (orientation_ == Orientation::Row) ? Orientation::Column : Orientation::Row;
}

template<typename DataType>
void Vector<DataType>::resize(size_t new_size) {
    if (size() < new_size) {
        for (size_t i = size(); i < new_size; ++i) {
            data_.push_back(DataType{});
        }
    }
    else {
        while (size() != new_size) {
            data_.pop_back();
        }
    }
}

template<typename DataType>
typename std::vector<DataType>::iterator Vector<DataType>::begin() {
    return data_.begin();
}

template<typename DataType>
typename std::vector<DataType>::iterator Vector<DataType>::end() {
    return data_.end();
}

template<typename DataType>
typename std::vector<DataType>::const_iterator Vector<DataType>::begin() const {
    return data_.cbegin();
}

template<typename DataType>
typename std::vector<DataType>::const_iterator Vector<DataType>::end() const {
    return data_.cend();
}

#pragma region SCALAR_OPERATIONS

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator+(const U& scalar) const {
    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(size(), ResultType{}, orientation());

    for (size_t i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(data_[i]) + static_cast<ResultType>(scalar);
    }

    return result;
}

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator-(const U& scalar) const {
    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(size(), ResultType{}, orientation());

    for (size_t i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(data_[i]) - static_cast<ResultType>(scalar);
    }

    return result;
}

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator*(const U& scalar) const {
    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(size(), ResultType{}, orientation());

    for (size_t i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(data_[i])* static_cast<ResultType>(scalar);
    }

    return result;
}

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator/(const U& scalar) const {
    using ResultType = std::common_type_t<DataType, U>;

    if (std::abs(static_cast<double>(scalar)) < EPSILON) {
        throw std::invalid_argument("Scalar cannot be 0.");
    }

    Vector<ResultType> result(size(), ResultType{}, orientation());

    for (size_t i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(data_[i]) / static_cast<ResultType>(scalar);
    }

    return result;
}

template<typename DataType, typename U>
Vector<std::common_type_t<DataType, U>> operator+(const U& scalar, const Vector<DataType>& vector) {
    return vector + scalar;
}

template<typename DataType, typename U>
Vector<std::common_type_t<DataType, U>> operator-(const U& scalar, const Vector<DataType>& vector) {
    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(vector.size(), ResultType{}, vector.orientation());

    for (size_t i = 0; i < vector.size(); ++i) {
        result.at(i) = static_cast<ResultType>(scalar) - static_cast<ResultType>(vector.at(i));
    }

    return result;
}

template<typename DataType, typename U>
Vector<std::common_type_t<DataType, U>> operator*(const U& scalar, const Vector<DataType>& vector) {
    return vector * scalar;
}

#pragma endregion SCALAR_OPERATIONS

#pragma region VECTOR_OPERATIONS

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator+(const Vector<U>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Size mismatch");
    }

    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(size(), ResultType{}, orientation_);

    for (int i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(at(i)) + static_cast<ResultType>(other.at(i));
    }

    return result;
}

template<typename DataType>
template<typename U>
Vector<std::common_type_t<DataType, U>> Vector<DataType>::operator-(const Vector<U>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Size mismatch");
    }

    using ResultType = std::common_type_t<DataType, U>;
    Vector<ResultType> result(size(), ResultType{}, orientation_);

    for (int i = 0; i < size(); ++i) {
        result.at(i) = static_cast<ResultType>(at(i)) - static_cast<ResultType>(other.at(i));
    }

    return result;
}

template<typename DataType>
template<typename U>
std::common_type_t<DataType, U> Vector<DataType>::dot(const Vector<U>& other) const {
    if (orientation_ != Orientation::Row || other.orientation() != Orientation::Column) {
        throw std::invalid_argument("dot: Expecting Row . Column");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Size mismatch.");
    }

    using ResultType = std::common_type_t<DataType, U>;
    ResultType sum = 0;
    for (size_t i = 0; i < size(); ++i) {
        sum += static_cast<ResultType>(at(i)) * static_cast<ResultType>(other.at(i));
    }
    return sum;
}

template<typename DataType>
template<typename U>
Matrix<std::common_type_t<DataType, U>> Vector<DataType>::outer(const Vector<U>& other) const {
    if (orientation_ != Orientation::Column || other.orientation() != Orientation::Row) {
        throw std::invalid_argument("outer: Expecting Column . Row");
    }

    using ResultType = std::common_type_t<DataType, U>;
    Matrix<ResultType> result(size(), other.size(), DataType{});
    for (size_t i = 0; i < size(); ++i) {
        for (size_t j = 0; j < other.size(); ++j) {
            result.at(i, j) = static_cast<ResultType>(at(i)) * static_cast<ResultType>(other.at(j));
        }
    }
    return result;
}

#pragma endregion VECTOR_OPERATIONS

template<typename DataType>
bool Vector<DataType>::operator==(const Vector& other) const {
    if (size() != other.size() || orientation_ != other.orientation_) return false;
    for (size_t i = 0; i < size(); ++i) {
        if (std::abs(at(i) - other.at(i)) > EPSILON) {
            return false;
        }
    }
    return true;
}

template<typename DataType>
DataType& Vector<DataType>::operator[](size_t index) {
    return at(index);
}

template<typename DataType>
const DataType& Vector<DataType>::operator[](size_t index) const {
    return at(index);
}

template<typename DataType>
void Vector<DataType>::print(std::ostream& os) {
    os << (orientation_ == Orientation::Row ? SQUARE_OPENING_BRACKET << SPACE : SQUARE_OPENING_BRACKET << NEW_LINE << SPACE);
    for (size_t i = 0; i < size(); ++i) {
        os << data_[i];
        if (i < size() - 1) {
            os << (orientation_ == Orientation::Row ? COMMA << SPACE : COMMA << NEW_LINE << SPACE);
        }
    }
    os << (orientation_ == Orientation::Row ? SPACE << SQUARE_CLOSEING_BRACKET : NEW_LINE << SQUARE_CLOSEING_BRACKET);
}

template<typename DataType>
std::ostream& operator<<(std::ostream& os, const Vector<DataType>& v) {
    v.print(os);
    return os;
}

#endif // !VECTOR_H

