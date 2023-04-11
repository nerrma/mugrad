#pragma once
#include "node/node.hpp"
#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mugrad {

template<typename T>
class Tensor {
public:
    Tensor() = default;
    ~Tensor() = default;

    explicit Tensor(std::pair<int, int> const& dims)
        : vals_ { std::vector<std::vector<T>>(dims.first, std::vector<double>(dims.second)) }
        , dims_ { dims }
    {
    }

    explicit Tensor(std::vector<std::vector<T>> const& init)
        : vals_ { init }
        , dims_ { init.size(), init[0].size() }
    {
    }

    auto operator[](size_t i) const -> std::vector<T>
    {
        return vals_[i];
    }

    auto operator[](size_t i) -> std::vector<T>&
    {
        return vals_[i];
    }

    auto operator+(Tensor const& other) -> Tensor
    {
        if (this->dims_ != other.dims_) {
            throw std::runtime_error("Matrix sum dim mismatch!");
        }

        auto result = Tensor(this->dims_);
        for (int i = 0; i < dims_.first; i++) {
            for (int j = 0; j < dims_.second; j++) {
                result[i][j] = this->vals_[i][j] + other.vals_[i][j];
            }
        }

        return result;
    }

    auto operator*(Tensor const& other) -> Tensor
    {
        if (this->dims_.second != other.dims_.first) {
            throw std::runtime_error("Matrix mulitplication dim mismatch!");
        }

        auto result = Tensor({ this->dims_.first, other.dims_.second });

        for (int i = 0; i < this->dims_.first; i++) {
            for (int j = 0; j < other.dims_.second; j++) {
                for (int k = 0; k < this->dims_.second; k++) {
                    result[i][j] += this->vals_[i][k] * other.vals_[k][j];
                }
            }
        }

        return result;
    }

    auto transpose() const -> Tensor
    {
        auto result = Tensor({ dims_.second, dims_.first });
        for (int i = 0; i < dims_.first; i++) {
            for (int j = 0; j < dims_.second; j++) {
                result[j][i] = vals_[i][j];
            }
        }

        return result;
    }

private:
    std::vector<std::vector<T>> vals_;
    std::pair<int, int> dims_;
};

} // namespace mugrad
