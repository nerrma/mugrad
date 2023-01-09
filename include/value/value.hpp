#ifndef VALUE_H_
#define VALUE_H_

namespace mugrad {
class value {
public:
  value(int s) : s_{s} {}

  auto get_s() -> int { return s_; }

private:
  int s_;
};
} // namespace mugrad

#endif // VALUE_H_
