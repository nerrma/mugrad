#pragma once

#include "node/node.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <iostream>
#include <memory>

namespace mugrad {

// all expressions
// template<typename T>
// class SubExpr { };
//
// template<typename T>
// class MulExpr { };
//
// template<typename T>
// class ExpExpr { };

template<typename T>
class AddExpr : public mugrad::Node<T> {
    using Node<T>::Node;

public:
    void backward() override
    {
        this->get_l()->add_grad(this->get_grad());
        this->get_r()->add_grad(this->get_grad());
    }

    void update() override
    {
        this->data = (this->get_l()->data + this->get_r()->data);
    }
};

template<typename T>
class SubExpr : public mugrad::Node<T> {
    using Node<T>::Node;

public:
    void backward() override
    {
        this->get_l()->add_grad(this->get_grad());
        this->get_r()->add_grad(-this->get_grad());
    }

    void update() override
    {
        this->data = (this->get_l()->data - this->get_r()->data);
    }
};

template<typename T>
class MulExpr : public mugrad::Node<T> {
    using Node<T>::Node;

public:
    void backward() override
    {
        this->get_l()->add_grad(this->get_r()->data * this->get_grad());
        this->get_r()->add_grad(this->get_l()->data * this->get_grad());
    }

    void update() override
    {
        this->data = (this->get_l()->data * this->get_r()->data);
    }
};

template<typename T>
class ExpExpr : public mugrad::Node<T> {
    using Node<T>::Node;

public:
    ExpExpr() = delete;
    ExpExpr(T, std::string) = delete;
    ExpExpr(T, std::string, std::shared_ptr<Node<T>>, std::shared_ptr<Node<T>>) = delete;

    explicit ExpExpr(T data, std::string label, std::shared_ptr<Node<T>> l,
        std::shared_ptr<Node<T>> r, unsigned int N)
        : Node<T>(data, label, l, r)
        , N_ { N }
    {
    }

    void backward() override
    {
        this->get_l()->add_grad(N_ * std::pow(this->get_l()->data, N_ - 1) * this->get_grad());
    }

    void update() override { this->data = (std::pow(this->get_l()->data, N_)); }

private:
    unsigned int N_;
};

template<typename T>
auto backward(Node<T>* head) -> void
{
    auto topo = head->gen_topo();

    head->set_grad(1);

    for (auto const& c : topo)
        c->backward();
}

template<>
auto backward(Node<Tensor>* head) -> void
{
    auto topo = head->gen_topo();
    head->zero_grad();

    std::cout << head->data.shape().first << std::endl;
    head->set_grad(Tensor::ones(head->data.shape()));

    for (auto const& c : topo)
        c->backward();
}

} // namespace mugrad
