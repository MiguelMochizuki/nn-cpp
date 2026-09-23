/**
 * test_tensor.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the Tensor class
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "nn/tensor.hpp"

using namespace nn;

TEST_CASE("zero-initialized construction") {
	Tensor t({2, 3});
	CHECK(t.shape() == std::vector<size_t>{2, 3});
	CHECK(t.ndim() == 2);
	CHECK(t.size() == 6);
	for (float v : t.data()) CHECK(v == 0.0f);
}

TEST_CASE("construction from explicit values") {
	Tensor t({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
	CHECK(t.get({0, 0}) == 1.0f);
	CHECK(t.get({0, 1}) == 2.0f);
	CHECK(t.get({1, 0}) == 3.0f);
	CHECK(t.get({1, 1}) == 4.0f);
}

TEST_CASE("construction from explicit values with wrong size throws") {
	CHECK_THROWS_AS(Tensor({2, 2}, std::vector<float>{1.0f}), TensorShapeError);
}

TEST_CASE("fill construction") {
	Tensor t({2, 2}, 5.0f);
	for (float v : t.data()) CHECK(v == 5.0f);
}

TEST_CASE("row-major strides") {
	Tensor t({2, 3, 4});
	CHECK(t.strides() == std::vector<size_t>{12, 4, 1});
}

TEST_CASE("at() writes are visible through get()") {
	Tensor t({2, 2});
	t.at({1, 1}) = 9.0f;
	CHECK(t.get({1, 1}) == 9.0f);
}

TEST_CASE("get() out of bounds throws") {
	Tensor t({2, 2});
	CHECK_THROWS_AS(t.get({2, 0}), TensorIndexError);
}

TEST_CASE("get() with wrong number of indices throws") {
	Tensor t({2, 2});
	CHECK_THROWS_AS(t.get({0}), TensorIndexError);
}

TEST_CASE("zeros and ones factories") {
	Tensor z = Tensor::zeros({2, 2});
	Tensor o = Tensor::ones({2, 2});
	for (float v : z.data()) CHECK(v == 0.0f);
	for (float v : o.data()) CHECK(v == 1.0f);
}

TEST_CASE("full factory") {
	Tensor t = Tensor::full({3}, 7.0f);
	for (float v : t.data()) CHECK(v == 7.0f);
}

TEST_CASE("random factory produces values in range") {
	Tensor t = Tensor::random({100}, -1.0f, 1.0f);
	for (float v : t.data()) {
		CHECK(v >= -1.0f);
		CHECK(v <= 1.0f);
	}
}

TEST_CASE("reshape preserves data in row-major order") {
	Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor r = t.reshape({3, 2});
	CHECK(r.shape() == std::vector<size_t>{3, 2});
	CHECK(r.get({0, 0}) == 1.0f);
	CHECK(r.get({0, 1}) == 2.0f);
	CHECK(r.get({2, 1}) == 6.0f);
}

TEST_CASE("reshape with mismatched size throws") {
	Tensor t({2, 3});
	CHECK_THROWS_AS(t.reshape({4, 4}), TensorShapeError);
}

TEST_CASE("transpose permutes dimensions") {
	Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor r = t.transpose({1, 0});
	CHECK(r.shape() == std::vector<size_t>{3, 2});
	CHECK(r.get({0, 0}) == 1.0f);
	CHECK(r.get({1, 0}) == 2.0f);
	CHECK(r.get({0, 1}) == 4.0f);
}

TEST_CASE("sum along axis drops that axis") {
	Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor s = t.sum(1);
	CHECK(s.shape() == std::vector<size_t>{2});
	CHECK(s.get({0}) == 6.0f);
	CHECK(s.get({1}) == 15.0f);
}

TEST_CASE("sum along axis 0") {
	Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor s = t.sum(0);
	CHECK(s.shape() == std::vector<size_t>{3});
	CHECK(s.get({0}) == 5.0f);
	CHECK(s.get({1}) == 7.0f);
	CHECK(s.get({2}) == 9.0f);
}

TEST_CASE("mean along axis") {
	Tensor t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor m = t.mean(1);
	CHECK(m.get({0}) == doctest::Approx(2.0f));
	CHECK(m.get({1}) == doctest::Approx(5.0f));
}

TEST_CASE("max along axis") {
	Tensor t({2, 3}, std::vector<float>{1, 5, 3, 4, 2, 6});
	Tensor m = t.max(1);
	CHECK(m.get({0}) == 5.0f);
	CHECK(m.get({1}) == 6.0f);
}

TEST_CASE("sum with negative or out-of-range axis throws") {
	Tensor t({2, 3});
	CHECK_THROWS_AS(t.sum(-1), TensorIndexError);
	CHECK_THROWS_AS(t.sum(2), TensorIndexError);
}

TEST_CASE("max with negative or out-of-range axis throws") {
	Tensor t({2, 3});
	CHECK_THROWS_AS(t.max(-1), TensorIndexError);
	CHECK_THROWS_AS(t.max(5), TensorIndexError);
}

TEST_CASE("sum over a zero-sized dimension returns zeros instead of crashing") {
	Tensor t({0, 3});
	Tensor s = t.sum(0);
	CHECK(s.shape() == std::vector<size_t>{3});
	CHECK(s.get({0}) == 0.0f);
}

TEST_CASE("add same shape") {
	Tensor a({2}, std::vector<float>{1, 2});
	Tensor b({2}, std::vector<float>{3, 4});
	Tensor c = add(a, b);
	CHECK(c.get({0}) == 4.0f);
	CHECK(c.get({1}) == 6.0f);
}

TEST_CASE("add broadcasts a smaller trailing dimension") {
	Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor b({3}, std::vector<float>{10, 20, 30});
	Tensor c = add(a, b);
	CHECK(c.shape() == std::vector<size_t>{2, 3});
	CHECK(c.get({0, 0}) == 11.0f);
	CHECK(c.get({1, 2}) == 36.0f);
}

TEST_CASE("add with incompatible shapes throws") {
	Tensor a({3});
	Tensor b({4});
	CHECK_THROWS_AS(add(a, b), TensorShapeError);
}

TEST_CASE("add with incompatible non-trailing shapes throws") {
	Tensor a({2, 3});
	Tensor b({2, 4});
	CHECK_THROWS_AS(add(a, b), TensorShapeError);
}

TEST_CASE("sub, mul, div elementwise") {
	Tensor a({2}, std::vector<float>{6, 8});
	Tensor b({2}, std::vector<float>{2, 4});
	CHECK(sub(a, b).get({0}) == 4.0f);
	CHECK(mul(a, b).get({1}) == 32.0f);
	CHECK(div(a, b).get({0}) == 3.0f);
}

TEST_CASE("scale multiplies every element") {
	Tensor a({2}, std::vector<float>{1, 2});
	Tensor c = scale(a, 3.0f);
	CHECK(c.get({0}) == 3.0f);
	CHECK(c.get({1}) == 6.0f);
}

TEST_CASE("matmul plain 2D") {
	Tensor a({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
	Tensor b({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
	Tensor c = matmul(a, b);
	CHECK(c.shape() == std::vector<size_t>{2, 2});
	CHECK(c.get({0, 0}) == doctest::Approx(58.0f));
	CHECK(c.get({0, 1}) == doctest::Approx(64.0f));
	CHECK(c.get({1, 0}) == doctest::Approx(139.0f));
	CHECK(c.get({1, 1}) == doctest::Approx(154.0f));
}

TEST_CASE("matmul batched with matching batch dim") {
	Tensor a({2, 2, 2}, std::vector<float>{1, 0, 0, 1, 2, 0, 0, 2});
	Tensor b({2, 2, 2}, std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1});
	Tensor c = matmul(a, b);
	CHECK(c.shape() == std::vector<size_t>{2, 2, 2});
	CHECK(c.get({0, 0, 0}) == doctest::Approx(1.0f));
	CHECK(c.get({1, 0, 0}) == doctest::Approx(2.0f));
}

TEST_CASE("matmul inner dimension mismatch throws") {
	Tensor a({2, 3});
	Tensor b({4, 2});
	CHECK_THROWS_AS(matmul(a, b), TensorShapeError);
}
