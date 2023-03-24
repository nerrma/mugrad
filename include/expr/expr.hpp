#pragma once

#include "node/node.hpp"
#include <cmath>
#include <iostream>

namespace mugrad {

class AddExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(this->get_grad());
    this->get_r()->add_grad(this->get_grad());
  }

  void update() override {
    this->data = (this->get_l()->data + this->get_r()->data);
  }
};

class SubExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(this->get_grad());
    this->get_r()->add_grad(-this->get_grad());
  }

  void update() override {
    this->data = (this->get_l()->data - this->get_r()->data);
  }
};

class MulExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(this->get_r()->data * this->get_grad());
    this->get_r()->add_grad(this->get_l()->data * this->get_grad());
  }

  void update() override {
    this->data = (this->get_l()->data * this->get_r()->data);
  }
};

template <unsigned int N> class ExpExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(N * std::pow(this->get_l()->data, N - 1) *
                            this->get_grad());
  }

  void update() override { this->data = (std::pow(this->get_l()->data, N)); }
};

auto backward(std::shared_ptr<Node> &head) -> void {
  auto topo = head->gen_topo();

  head->set_grad(1);

  for (auto const &c : topo)
    c->backward();
}

} // namespace mugrad
