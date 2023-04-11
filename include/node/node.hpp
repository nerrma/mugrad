#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mugrad {
template<typename T>
class Node {
    /* definition of an expression tree node
     *
     * this acts as an abstract base class which
     * values and operations inherit from
     * */
public:
    Node() = default;
    virtual ~Node() = default;

    Node(T data)
        : data { data }
        , label_ {}
        , l_ {}
        , r_ {}
    {
    }

    Node(T data, std::string label)
        : data { data }
        , label_ { label }
        , l_ {}
        , r_ {}
    {
    }

    Node(T data, std::string label, std::shared_ptr<Node<T>> l,
        std::shared_ptr<Node<T>> r)
        : data { data }
        , label_ { label }
        , l_ { l }
        , r_ { r }
    {
    }

    double data;

    auto set_grad(T const& grad) -> void { grad_ = grad; }
    [[nodiscard]] auto get_grad() const -> double { return grad_; }
    auto add_grad(T const& addend) -> void { this->grad_ += addend; }
    auto set_ptr(std::shared_ptr<Node<T>> const& ptr) -> void { this->ptr_ = ptr; }
    auto zero_grad() -> void
    {
        this->grad_ = 0;
        if (l_ != nullptr)
            l_->zero_grad();

        if (r_ != nullptr)
            r_->zero_grad();
    }

    virtual void backward() { }
    virtual void update()
    {
        if (l_ != nullptr)
            l_->update();

        if (r_ != nullptr)
            r_->update();
    }

    [[nodiscard]] auto get_l() -> std::shared_ptr<Node<T>> { return l_; }
    [[nodiscard]] auto get_r() -> std::shared_ptr<Node<T>> { return r_; }

    [[nodiscard]] auto get_label() -> std::string { return label_; }

    [[nodiscard]] auto gen_topo() -> std::vector<Node<T>*>
    {
        if (this->ptr_ == nullptr) {
            return {};
        }

        std::vector<Node<T>*> res;
        std::set<Node<T>*> seen;

        dfs(this, res, seen);
        std::reverse(res.begin(), res.end());
        return res;
    }

private:
    auto dfs(Node<T>* node,
        std::vector<Node<T>*>& topo,
        std::set<Node<T>*>& seen) -> void
    {
        if (node == nullptr) {
            return;
        }

        seen.insert(node);

        if (!seen.contains(node->l_.get())) {
            dfs(node->l_.get(), topo, seen);
        }

        if (!seen.contains(node->r_.get())) {
            dfs(node->r_.get(), topo, seen);
        }

        topo.push_back(node);
    }

    T grad_;
    std::string label_;
    std::shared_ptr<Node<T>> l_;
    std::shared_ptr<Node<T>> r_;
    std::shared_ptr<Node<T>> ptr_;
};

} // namespace mugrad
