#include "expr/expr.hpp"
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
  auto res = v1 * v2 * v3 * v4;
  CHECK(res->get_data() == 24.0);

  mugrad::backward(res);
  CHECK(v2->get_grad() == 12.0);
}

TEST_CASE("multiple operation backward") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(1.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(2.0, "v2"));
  auto v3 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v3"));
  auto v4 = std::make_shared<mugrad::Node>(mugrad::Value(4.0, "v4"));
  auto res = v1 * v2 + v3 * v2 + v4 * v2;
  CHECK(res->get_data() == 16.0);

  mugrad::backward(res);
  CHECK(v2->get_grad() == 8.0);
}

TEST_CASE("layered backward") {
  auto v1 = std::make_shared<mugrad::Node>(mugrad::Value(3.0, "v1"));
  auto v2 = std::make_shared<mugrad::Node>(mugrad::Value(4.0, "v2"));

  auto res = v1 * v2 + v1;
  CHECK(res->get_data() == 15.0);

  auto res2 = res * v2 + res;
  CHECK(res2->get_data() == 75.0);

  /*
    res2 = v2 * (v1 * v2 + v1) + v1 * v2 + v1
    dres2/dv2 = 2 * v1 * v2 + v1 + v1
   */
  mugrad::backward(res2);
  CHECK(v2->get_grad() == 30.0);
}
