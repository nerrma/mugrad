#pragma once

#include "node/node.hpp"
#include <cmath>
#include <iostream>
#include <memory>

namespace mugrad {

// all expressions
template<typename T>
class AddExpr { };

template<typename T>
class SubExpr { };

template<typename T>
class MulExpr { };

template<typename T>
class ExpExpr { };

template<>
class AddExpr<double> : public mugrad::Node<double> {
    using Node::Node;

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

template<>
class SubExpr<double> : public mugrad::Node<double> {
    using Node::Node;

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

template<>
class MulExpr<double> : public mugrad::Node<double> {
    using Node::Node;

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

template<>
class ExpExpr<double> : public mugrad::Node<double> {
    using Node::Node;

public:
    ExpExpr() = delete;
    ExpExpr(double, std::string) = delete;
    ExpExpr(double, std::string, std::shared_ptr<Node<double>>, std::shared_ptr<Node<double>>) = delete;

    explicit ExpExpr(double data, std::string label, std::shared_ptr<Node<double>> l,
        std::shared_ptr<Node<double>> r, unsigned int N)
        : Node(data, label, l, r)
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

} // namespace mugrad
