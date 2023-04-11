#include "tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <memory>

TEST_CASE("basic constructor test")
{
    auto v = mugrad::Tensor<double>({ { 1, 2, 3 }, { 4, 5, 6 } });

    CHECK(v[0] == std::vector<double> { 1, 2, 3 });
    CHECK(v[1] == std::vector<double> { 4, 5, 6 });
}
