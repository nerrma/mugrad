#include "expr/expr.hpp"
#include "value/value.hpp"
#include "var/var.hpp"
#include <iostream>
#include <vector>

auto main() -> int
{
    auto x1 = std::make_shared<mugrad::Node<double>>(mugrad::Value(1.5, "x1"));
    auto x2 = std::make_shared<mugrad::Node<double>>(mugrad::Value(1.5, "x2"));

    auto x = std::vector<double> { 1, 2, 3, 4, 5, 6, 7, 8 };
    auto y = std::vector<double> { 0, 3, 3, 2, 8, 4, 6, 9 };

    double eta = 0.001;
    double tol = 0.0001;
    for (int epoch = 0; epoch < 10; epoch++) {
        for (int i = 0; i < x.size(); i++) {
            auto loss = mugrad::Exp(y[i] - x[i] * x2 - x1, 2) + mugrad::Exp(x2, 2) + mugrad::Exp(x1, 2);
            std::cout << loss->data << std::endl;

            mugrad::backward(loss.get());
            if (x1->get_grad() < tol && x2->get_grad() < tol) {
                std::cout << "reached thresh with - "
                          << "x1: " << x1->data << ", x2: " << x2->data << std::endl;
                return 0;
            }
            x1->data = (x1->data - x1->get_grad() * eta);
            x2->data = (x2->data - x2->get_grad() * eta);

            loss->update();
            std::cout << "x1: " << x1->data << ", x2: " << x2->data << "\t grads -> "
                      << "x1: " << x1->get_grad() << ", x2: " << x2->get_grad()
                      << std::endl;

            loss->zero_grad();
        }
    }
    std::cout << "terminated with - "
              << "x1: " << x1->data << ", x2: " << x2->data << std::endl;
}
