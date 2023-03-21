#include "expr/expr.hpp"
#include "value/value.hpp"
#include <iostream>
#include <vector>

auto main() -> int {
  auto x1 = std::make_shared<mugrad::Node>(mugrad::Value(1.5, "x1"));
  auto x2 = std::make_shared<mugrad::Node>(mugrad::Value(1.5, "x2"));

  auto x = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  auto y = std::vector<int>{0, 3, 3, 2, 8, 4, 6, 9};

  double eta = 0.001;
  double tol = 0.0001;
  for (int epoch = 0; epoch < 10; epoch++) {
    for (int i = 0; i < x.size(); i++) {
      auto loss = mugrad::Exp<2>(y[i] - x[i] * x2 - x1) + mugrad::Exp<2>(x2) +
                  mugrad::Exp<2>(x1);
      std::cout << loss->get_data() << std::endl;

      mugrad::backward(loss);
      if (x1->get_grad() < tol && x2->get_grad() < tol) {
        std::cout << "reached thresh with - "
                  << "x1: " << x1->get_data() << ", x2: " << x2->get_data()
                  << std::endl;
        return 0;
      }
      x1->set_data(x1->get_data() - x1->get_grad() * eta);
      x2->set_data(x2->get_data() - x2->get_grad() * eta);

      loss->update();
      std::cout << "x1: " << x1->get_data() << ", x2: " << x2->get_data()
                << "\t grads -> "
                << "x1: " << x1->get_grad() << ", x2: " << x2->get_grad()
                << std::endl;

      loss->zero_grad();
    }
  }
  std::cout << "terminated with - "
            << "x1: " << x1->get_data() << ", x2: " << x2->get_data()
            << std::endl;
}
