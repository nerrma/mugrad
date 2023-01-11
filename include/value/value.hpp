#ifndef VALUE_H_
#define VALUE_H_

#include <functional>
#include <memory>
#include <set>
#include <string>

namespace mugrad {
class value {
public:
  value(double data, std::string label)
      : data_{data}, label_{label}, backward_{[]() {}}, children_{{nullptr,
                                                                   nullptr}} {}

  value(double data, std::string label, std::array<value *, 2> children)
      : data_{data}, label_{label}, backward_{[]() {}}, children_{children} {}

  ~value() {}

  [[nodiscard]] auto get_data() const -> double { return data_; }
  auto set_data(double const &d) -> void { data_ = d; }

  [[nodiscard]] auto get_grad() const -> double { return grad_; }
  auto add_grad(double const &addend) -> void { this->grad_ += addend; }
  auto set_grad(double const &val) -> void { this->grad_ = val; }

  auto set_backward(std::function<void(void)> const &f) -> void {
    backward_ = f;
  }

  auto backward() -> void {
    this->backward_();

    if (children_[0] != nullptr)
      children_[0]->backward();
    if (children_[1] != nullptr && children_[0] != children_[1])
      children_[1]->backward();
  }

  auto operator+(value &o) -> value {
    value result(o.data_ + this->data_, "+", {this, &o});

    auto backward = [&]() {
      this->add_grad(result.grad_);
      o.add_grad(result.grad_);
    };

    result.set_backward(backward);
    return result;
  }

private:
  double data_;
  double grad_;
  std::function<void(void)> backward_;
  std::string label_;
  std::array<value *, 2> children_;
};
} // namespace mugrad

#endif // VALUE_H_
