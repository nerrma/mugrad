#include "tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <memory>

TEST_CASE("basic constructor test")
{
    auto v = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });

    CHECK(v[0] == std::vector<double> { 1, 2, 3 });
    CHECK(v[1] == std::vector<double> { 4, 5, 6 });
}

TEST_CASE("basic addition test")
{
    auto v1 = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });
    auto v2 = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });

    auto v3 = v1 + v2;
    CHECK(v3[0] == std::vector<double> { 2, 4, 6 });
    CHECK(v3[1] == std::vector<double> { 8, 10, 12 });
}

TEST_CASE("basic mul test")
{
    auto v1 = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });
    auto v2 = mugrad::Tensor<double>({ { 10, 11 }, { 20, 21 }, { 30, 31 } });

    auto v3 = v1 * v2;
    CHECK(v3[0] == std::vector<double> { 140, 146 });
    CHECK(v3[1] == std::vector<double> { 320, 335 });
}

TEST_CASE("basic transpose test")
{
    auto v1 = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });

    auto v2 = v1.transpose();
    CHECK(v2[0] == std::vector<double> { { 1, 4 } });
    CHECK(v2[1] == std::vector<double> { { 2, 5 } });
    CHECK(v2[2] == std::vector<double> { { 3, 6 } });
}
