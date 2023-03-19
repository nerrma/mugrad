#include "value/value.hpp"
#include <catch2/catch.hpp>

TEST_CASE("basic constructor test") {
  auto v = mugrad::Value(2.0, "v1");
  CHECK(v.get_data() == 2.0);
}

TEST_CASE("basic addition test") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(2.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v2"));
  auto res = v1 + v2;
  CHECK(res->get_data() == 5.0);
}

TEST_CASE("basic mult test") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(2.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v2"));
  auto res = v1 * v2;
  CHECK(res->get_data() == 6.0);
}

TEST_CASE("backward?") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(2.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v2"));
  auto res = v1 * v2;
  CHECK(res->get_data() == 6.0);
  res->set_grad(1);
  res->backward();
  CHECK(v1->get_grad() == 3.0);
  CHECK(v2->get_grad() == 2.0);
}

TEST_CASE("multiple layer backward") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(1.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(2.0, "v2"));
  auto v3 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v3"));
  auto v4 = std::make_shared<mugrad::Node>(mugrad::Value(4.0, "v4"));
  // auto res = v1 * v2 + v3 * v2 + v4 * v2;
  auto res = v1 * v2 * v3 * v4;
  CHECK(res->get_data() == 24.0);

  auto topo = res->gen_topo(res);

  res->set_grad(1);

  for (auto const &c : topo)
    c->backward();

  CHECK(v2->get_grad() == 12.0);
}
