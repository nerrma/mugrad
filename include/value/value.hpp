#ifndef VALUE_H_
#define VALUE_H_

#include "expr/expr.hpp"
#include "node/node.hpp"
#include <functional>
#include <memory>
#include <set>
#include <string>

namespace mugrad {
class Value : public mugrad::Node {
  using Node::Node;

public:
  void backward() override { return; }
};

auto operator+(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  AddExpr result(a->get_data() + b->get_data(), "+", a, b);

  return std::make_shared<AddExpr>(result);
}

auto operator*(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  MulExpr result(a->get_data() * b->get_data(), "*", a, b);

  return std::make_shared<MulExpr>(result);
}

} // namespace mugrad

#endif // VALUE_H_
