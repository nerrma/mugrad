#pragma once

#include "expr/expr.hpp"
#include "node/node.hpp"
#include "value/value.hpp"

#include <cmath>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

namespace mugrad {

// Pointer operators
auto operator+(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  AddExpr result(a->data + b->data, "+", a, b);
  auto ptr = std::make_shared<AddExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

auto operator-(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  SubExpr result(a->data - b->data, "+", a, b);
  auto ptr = std::make_shared<SubExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

auto operator*(std::shared_ptr<Node> const &a, std::shared_ptr<Node> const &b)
    -> std::shared_ptr<Node> {
  MulExpr result(a->data * b->data, "*", a, b);
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
  MulExpr result(av->data * b->data, "*", av, b);
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
  AddExpr result(av->data + b->data, "*", av, b);
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
  SubExpr result(a - b->data, "*", av, b);
  auto ptr = std::make_shared<SubExpr>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

template <unsigned int N>
auto Exp(std::shared_ptr<Node> const &a) -> std::shared_ptr<Node> {
  ExpExpr<N> result(std::pow(a->data, N), "*", a, nullptr);
  auto ptr = std::make_shared<ExpExpr<N>>(result);

  ptr->set_ptr(ptr);

  return ptr;
}

} // namespace mugrad
