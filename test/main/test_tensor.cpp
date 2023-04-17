#include "expr/expr.hpp"
#include "tensor/tensor.hpp"
#include "var/var.hpp"
#include <catch2/catch.hpp>
#include <memory>

TEST_CASE("basic constructor test")
{
    auto v = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

    CHECK(v[0] == std::vector<double> { 1, 2, 3 });
    CHECK(v[1] == std::vector<double> { 4, 5, 6 });
}

TEST_CASE("basic addition test")
{
    auto v1 = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });
    auto v2 = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

    auto v3 = v1 + v2;
    CHECK(v3[0] == std::vector<double> { 2, 4, 6 });
    CHECK(v3[1] == std::vector<double> { 8, 10, 12 });
}

TEST_CASE("basic mul test")
{
    auto v1 = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });
    auto v2 = mugrad::Tensor({ { 10, 11 }, { 20, 21 }, { 30, 31 } });

    auto v3 = v1 * v2;
    CHECK(v3[0] == std::vector<double> { 140, 146 });
    CHECK(v3[1] == std::vector<double> { 320, 335 });
}

TEST_CASE("basic transpose test")
{
    auto v1 = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });

    auto v2 = v1.transpose();
    CHECK(v2[0] == std::vector<double> { { 1, 4 } });
    CHECK(v2[1] == std::vector<double> { { 2, 5 } });
    CHECK(v2[2] == std::vector<double> { { 3, 6 } });
}

TEST_CASE("backward test")
{
    auto tensor = mugrad::Tensor({ { 1, 2, 3 }, { 4, 5, 6 } });
    auto v1 = std::make_shared<mugrad::Node<mugrad::Tensor>>(tensor);
    auto v2 = std::make_shared<mugrad::Node<mugrad::Tensor>>(tensor.transpose());

    // v1->set_grad(mugrad::Tensor::ones(tensor.shape()));
    // v2->set_grad(mugrad::Tensor::ones(tensor.transpose().shape()));

    auto v3 = v1 * v2;
    mugrad::backward(v3.get());

    std::cout << v3.use_count() << " " << v1.use_count() << " " << v2.use_count() << std::endl;

    // auto grad = v3.get()->get_grad();
    //  CHECK(grad == tensor);
}
