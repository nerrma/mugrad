#include "value/value.hpp"
#include <catch2/catch.hpp>

TEST_CASE("basic constructor test") {
  auto v = mugrad::value(2.0, "v1");
  CHECK(v.get_data() == 2.0);
}

TEST_CASE("basic addition test") {
  auto v1 = mugrad::value(2.0, "v1");
  auto v2 = mugrad::value(3.0, "v2");
  auto res = v1 + v2;
  CHECK(res.get_data() == 5.0);
}

TEST_CASE("basic addtion backward test") {
  auto v1 = mugrad::value(2.0, "v1");
  auto v2 = mugrad::value(3.0, "v2");
  auto res = v1 + v2;
  res.backward();

  CHECK(res.get_grad() == 1.0);
}
