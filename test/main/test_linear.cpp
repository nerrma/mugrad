#include "value/value.hpp"
#include <catch2/catch.hpp>

TEST_CASE("basic constructor test") {
  auto v = mugrad::Value(2.0, "v1");
  CHECK(v.get_data() == 2.0);
}

TEST_CASE("basic addition test") {
  auto v1 = mugrad::Value(2.0, "v1");
  auto v2 = mugrad::Value(3.0, "v2");
  auto res = v1 + v2;
  CHECK(res.get_data() == 5.0);
}

TEST_CASE("basic mult test") {
  auto v1 = mugrad::Value(2.0, "v1");
  auto v2 = mugrad::Value(3.0, "v2");
  auto res = v1 * v2;
  CHECK(res.get_data() == 6.0);
}

TEST_CASE("backward?") {
  auto v1 = mugrad::Value(2.0, "v1");
  auto v2 = mugrad::Value(3.0, "v2");
  auto res = v1 * v2;
  CHECK(res.get_data() == 6.0);
  res.backward();
  CHECK(v1.get_grad() == 3.0);
  CHECK(v2.get_grad() == 2.0);
}
