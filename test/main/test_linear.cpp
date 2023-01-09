#include "value/value.hpp"
#include <catch2/catch.hpp>

TEST_CASE("basic test") {
  auto v = mugrad::value(2.0);
  CHECK(v.get_s() == 2.0);
}
