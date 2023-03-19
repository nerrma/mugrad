#ifndef EXPR_H_
#define EXPR_H_

#include "node/node.hpp"

namespace mugrad {
class AddExpr : public mugrad::Node {
  using Node::Node;

public:
  virtual void backward() {
    for (auto &child : this->get_children())
      child->add_grad(this->get_grad());
  }
};

class MulExpr : public mugrad::Node {
  using Node::Node;

public:
  virtual void backward() {
    for (auto &child : this->get_children())
      child->add_grad(child->get_grad() * this->get_grad());
  }
};
} // namespace mugrad

#endif // EXPR_H_
