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
  auto operator+(Node &o) -> AddExpr {
    AddExpr result(o.get_data() + this->get_data(), "+",
                   {std::make_shared<Node>(this), std::make_shared<Node>(o)});

    return AddExpr();
  }

  auto operator*(Node &o) -> MulExpr {
    MulExpr result(o.get_data() * this->get_data(), "+",
                   {std::make_shared<Node>(this), std::make_shared<Node>(o)});

    return MulExpr();
  }

  virtual void backward() { return; }
};
} // namespace mugrad

#endif // VALUE_H_
