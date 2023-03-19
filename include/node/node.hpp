#ifndef NODE_H_
#define NODE_H_

#include <memory>
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
  Node(double data, std::string label)
      : data_{data}, label_{label}, children_{} {}

  Node(double data, std::string label,
       std::vector<std::shared_ptr<Node>> children)
      : data_{data}, label_{label}, children_{children} {}

  [[nodiscard]] auto get_data() const -> double { return data_; }
  auto set_data(double const &d) -> void { data_ = d; }

  [[nodiscard]] auto get_grad() const -> double { return grad_; }
  auto add_grad(double const &addend) -> void { this->grad_ += addend; }
  auto set_grad(double const &val) -> void { this->grad_ = val; }
  auto zero_grad() -> void { this->grad_ = 0; }

  virtual auto backward() -> void = 0;

  [[nodiscard]] auto get_children() -> std::vector<std::shared_ptr<Node>> {
    return children_;
  }

private:
  double data_;
  double grad_;
  std::string label_;
  std::vector<std::shared_ptr<Node>> children_;
};
} // namespace mugrad

#endif // NODE_H_
