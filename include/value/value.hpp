#pragma once

#include "expr/expr.hpp"
#include "node/node.hpp"

namespace mugrad {
class Value : public mugrad::Node {
  using Node::Node;

public:
  void backward() override { return; }
};
} // namespace mugrad
