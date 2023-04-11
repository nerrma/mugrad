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
template<typename T>
auto operator+(std::shared_ptr<Node<T>> const& a, std::shared_ptr<Node<T>> const& b)
    -> std::shared_ptr<Node<T>>
{
    AddExpr<T> result(a->data + b->data, "+", a, b);
    auto ptr = std::make_shared<AddExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

template<typename T>
auto operator-(std::shared_ptr<Node<T>> const& a, std::shared_ptr<Node<T>> const& b)
    -> std::shared_ptr<Node<T>>
{
    SubExpr<T> result(a->data - b->data, "+", a, b);
    auto ptr = std::make_shared<SubExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

template<typename T>
auto operator*(std::shared_ptr<Node<T>> const& a, std::shared_ptr<Node<T>> const& b)
    -> std::shared_ptr<Node<T>>
{
    MulExpr<T> result(a->data * b->data, "*", a, b);
    auto ptr = std::make_shared<MulExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

// Constant type operators
template<typename T, typename T2>
auto operator*(T const& a, std::shared_ptr<Node<T2>> const& b)
    -> std::shared_ptr<Node<T>>
requires std::integral<T> || std::floating_point<T>
{
    auto av = std::make_shared<Value>(Value(a));
    MulExpr<T> result(av->data * b->data, "*", av, b);
    auto ptr = std::make_shared<MulExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

template<typename T>
auto operator+(T const& a, std::shared_ptr<Node<T>> const& b)
    -> std::shared_ptr<Node<T>>
requires std::integral<T> || std::floating_point<T>
{
    auto av = std::make_shared<Value>(Value(a));
    AddExpr result(av->data + b->data, "*", av, b);
    auto ptr = std::make_shared<AddExpr>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

template<typename T>
auto operator-(T const& a, std::shared_ptr<Node<T>> const& b)
    -> std::shared_ptr<Node<T>>
requires std::integral<T> || std::floating_point<T>
{
    auto av = std::make_shared<Value>(Value(a));
    SubExpr<T> result(a - b->data, "*", av, b);
    auto ptr = std::make_shared<SubExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

template<typename T>
auto Exp(std::shared_ptr<Node<T>> const& a, unsigned int N) -> std::shared_ptr<Node<T>>
{
    ExpExpr<T> result(std::pow(a->data, N), "*", a, nullptr, N);
    auto ptr = std::make_shared<ExpExpr<T>>(result);

    ptr->set_ptr(ptr);

    return ptr;
}

} // namespace mugrad
