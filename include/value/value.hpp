#ifndef VALUE_H_
#define VALUE_H_

#include "expr/expr.hpp"
#include "node/node.hpp"
#include <cmath>
#include <concepts>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

namespace mugrad {
class Value : public mugrad::Node {
  using Node::Node;

public:
  void backward() override { return; }
};

// Pointer operators
auto operator+(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  AddExpr result(a->get_data() + b->get_data(), "+", a, b);
  auto ptr = std::make_shared<AddExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

auto operator-(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  SubExpr result(a->get_data() - b->get_data(), "+", a, b);
  auto ptr = std::make_shared<SubExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

auto operator*(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  MulExpr result(a->get_data() * b->get_data(), "*", a, b);
  auto ptr = std::make_shared<MulExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

// Constant type operators
template <typename T>
auto operator*(T const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node>
  requires std::integral<T> || std::floating_point<T>
{
  auto av = std::make_shared<Value>(Value(a));
  MulExpr result(av->get_data() * b->get_data(), "*", av, b);
  auto ptr = std::make_shared<MulExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

template <typename T>
auto operator+(T const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node>
  requires std::integral<T> || std::floating_point<T>
{
  auto av = std::make_shared<Value>(Value(a));
  AddExpr result(av->get_data() + b->get_data(), "*", av, b);
  auto ptr = std::make_shared<AddExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

template <typename T>
auto operator-(T const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node>
  requires std::integral<T> || std::floating_point<T>
{
  auto av = std::make_shared<Value>(Value(a));
  SubExpr result(a - b->get_data(), "*", av, b);
  auto ptr = std::make_shared<SubExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

template <unsigned int N>
auto Exp(std::shared_ptr<Node> const &a) -> std::shared_ptr<Node> {
  ExpExpr<N> result(std::pow(a->get_data(), N), "*", a, nullptr);
  auto ptr = std::make_shared<ExpExpr<N>>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

} // namespace mugrad

#endif // VALUE_H_
