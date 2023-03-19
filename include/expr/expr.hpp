#ifndef EXPR_H_
#define EXPR_H_

#include "node/node.hpp"
#include <iostream>

namespace mugrad {
class AddExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(this->get_grad());
    this->get_r()->add_grad(this->get_grad());
  }
};

class MulExpr : public mugrad::Node {
  using Node::Node;

public:
  void backward() override {
    this->get_l()->add_grad(this->get_r()->get_data() * this->get_grad());
    this->get_r()->add_grad(this->get_l()->get_data() * this->get_grad());
  }
};

auto backward(std::shared_ptr<Node> &head) -> void {
  auto topo = head->gen_topo();

  head->set_grad(1);

  for (auto const &c : topo)
    c->backward();
}

} // namespace mugrad

#endif // EXPR_H_
