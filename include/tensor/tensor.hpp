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

        auto result = Tensor();
        for (int i = 0; i < dims_.first; i++) {
            for (int j = 0; j < dims_.second; j++) {
                result[i][j] = this->vals_[i][j] + other.vals_[i][j];
            }
        }

        return result;
    }

    auto operator*(Tensor const& other) -> Tensor
    {
        if (this->dims_.first != other.dims_.second) {
            throw std::runtime_error("Matrix mulitplication dim mismatch!");
        }
        // TODO: Matrix multiplication
    }

private:
    std::vector<std::vector<T>> vals_;
    std::pair<int, int> dims_;
};

} // namespace mugrad
