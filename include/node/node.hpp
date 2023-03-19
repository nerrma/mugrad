#ifndef NODE_H_
#define NODE_H_

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace mugrad {
class Node {
  /* definition of an expression tree node
   *
   * this acts as an abstract base class which
   * values and operations inherit from
   * */
public:
  Node() = default;
  virtual ~Node() = default;

  Node(double data, std::string label)
      : data_{data}, label_{label}, l_{}, r_{} {}

  Node(double data, std::string label, std::shared_ptr<Node> l,
       std::shared_ptr<Node> r)
      : data_{data}, label_{label}, l_{l}, r_{r} {}

  [[nodiscard]] auto get_data() const -> double { return data_; }
  auto set_data(double const &d) -> void { data_ = d; }

  [[nodiscard]] auto get_grad() const -> double { return grad_; }
  auto add_grad(double const &addend) -> void { this->grad_ += addend; }
  auto set_grad(double const &val) -> void { this->grad_ = val; }
  auto zero_grad() -> void { this->grad_ = 0; }

  virtual void backward(){};

  [[nodiscard]] auto get_l() -> std::shared_ptr<Node> { return l_; }

  [[nodiscard]] auto get_r() -> std::shared_ptr<Node> { return r_; }

  [[nodiscard]] auto get_label() -> std::string { return label_; }

  [[nodiscard]] auto gen_topo(std::shared_ptr<Node> const &ptr)
      -> std::vector<std::shared_ptr<Node>> {
    std::vector<std::shared_ptr<Node>> res;
    std::set<std::shared_ptr<Node>> seen;

    dfs(ptr, res, seen);
    std::reverse(res.begin(), res.end());
    return res;
  }

private:
  auto dfs(std::shared_ptr<Node> const &node,
           std::vector<std::shared_ptr<Node>> &topo,
           std::set<std::shared_ptr<Node>> &seen) -> void {
    if (node == nullptr) {
      return;
    }

    seen.insert(node);

    if (!seen.contains(node->l_)) {
      dfs(node->l_, topo, seen);
    }

    if (!seen.contains(node->r_)) {
      dfs(node->r_, topo, seen);
    }

    topo.push_back(node);
  }

  double data_;
  double grad_;
  std::string label_;
  // std::vector<std::shared_ptr<Node>> children_;
  std::shared_ptr<Node> l_;
  std::shared_ptr<Node> r_;
};

} // namespace mugrad

#endif // NODE_H_
