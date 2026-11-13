/**
 * test_tensor.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the Tensor class
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "nn/tensor.hpp"
#include "nn/kernels/parallel.hpp"

#include <cmath>

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

TEST_CASE("reshape of a temporary keeps data and shape, and leaves the source intact when copied") {
	for (size_t n : {size_t{1}, size_t{7}, size_t{33}, size_t{257}}) {
		Tensor t({n});
		for (size_t i = 0; i < n; i++) t.data()[i] = static_cast<float>(i);
		Tensor copy = t.reshape({1, n});
		Tensor moved = Tensor(t).reshape({n, 1});
		CHECK(copy.shape() == std::vector<size_t>{1, n});
		CHECK(moved.shape() == std::vector<size_t>{n, 1});
		CHECK(copy.data() == t.data());
		CHECK(moved.data() == t.data());
	}
	CHECK_THROWS_AS(Tensor({2, 3}).reshape({4, 4}), TensorShapeError);
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

TEST_CASE("broadcasting keeps zero-size dimensions") {
	CHECK(add(Tensor({0, 3}, 1.0f), Tensor({1, 3}, 2.0f)).shape() == std::vector<size_t>{0, 3});
	CHECK(add(Tensor({1, 3}, 2.0f), Tensor({0, 3}, 1.0f)).shape() == std::vector<size_t>{0, 3});
	CHECK(add(Tensor({0}, 1.0f), Tensor({1}, 2.0f)).shape() == std::vector<size_t>{0});
	CHECK(mul(Tensor({2, 0}, 1.0f), Tensor({2, 1}, 2.0f)).shape() == std::vector<size_t>{2, 0});
	CHECK_THROWS_AS(add(Tensor({0, 3}, 1.0f), Tensor({2, 3}, 2.0f)), TensorShapeError);
}

namespace {
// Reference broadcast: walks every output index and reads each operand through its own dimensions
Tensor reference_broadcast(const Tensor& a, const Tensor& b, float (*op)(float, float), const std::vector<size_t>& out_shape) {
	Tensor out(out_shape);
	size_t rank = out_shape.size();
	for (size_t flat = 0; flat < out.size(); flat++) {
		std::vector<size_t> idx(rank);
		size_t rem = flat;
		for (size_t d = rank; d-- > 0;) {
			idx[d] = rem % out_shape[d];
			rem /= out_shape[d];
		}
		auto pick = [&](const Tensor& t) {
			std::vector<size_t> ti(t.ndim());
			for (size_t d = 0; d < t.ndim(); d++) ti[d] = t.shape()[d] == 1 ? 0 : idx[d + rank - t.ndim()];
			return t.get(ti);
		};
		out.data()[flat] = op(pick(a), pick(b));
	}
	return out;
}
}  // namespace

TEST_CASE("elementwise broadcast fast paths match the index-by-index reference") {
	auto fill = [](Tensor& t, float seed) {
		for (size_t i = 0; i < t.size(); i++) t.data()[i] = seed + static_cast<float>((i * 7) % 11);
	};
	for (size_t n : {size_t{1}, size_t{7}, size_t{33}, size_t{257}}) {
		using Shapes = std::pair<std::vector<size_t>, std::vector<size_t>>;
		std::vector<Shapes> cases = {
			{{3, n}, {n}},          // row
			{{3, n}, {1, n}},       // row with leading 1
			{{3, n}, {1}},          // scalar
			{{3, n}, {}},           // rank-0 scalar
			{{3, n}, {3, 1}},       // column
			{{2, 3, n}, {3, n}},    // batch broadcast
			{{2, 3, n}, {2, 3, 1}}, // column on 3D
			{{2, 3, n}, {1, 1}},    // all-ones lower rank
			{{3, 1}, {1, n}},       // both expand (general path)
			{{n}, {n}},             // same shape
		};
		for (const auto& c : cases) {
			Tensor a(c.first), b(c.second);
			fill(a, 1.0f);
			fill(b, 0.25f);
			std::vector<size_t> out_shape = (a.ndim() >= b.ndim() ? a : b).shape();
			if (c.first == std::vector<size_t>{3, 1}) out_shape = {3, n};
			CHECK(sub(a, b).shape() == out_shape);
			CHECK(sub(a, b).data() == reference_broadcast(a, b, +[](float x, float y) { return x - y; }, out_shape).data());
			CHECK(sub(b, a).data() == reference_broadcast(b, a, +[](float x, float y) { return x - y; }, out_shape).data());
		}
	}
}

namespace {
using ShapePair = std::pair<std::vector<size_t>, std::vector<size_t>>;

// Fills a tensor with reproducible, non-zero values of mixed sign and magnitude
void fill_random(Tensor& t, unsigned seed) {
	unsigned s = seed * 2654435761u + 12345u;
	for (size_t i = 0; i < t.size(); i++) {
		s = s * 1664525u + 1013904223u;
		float v = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 20.0f;
		t.data()[i] = v == 0.0f ? 0.5f : v;
	}
}

// Checks add, sub, mul and div of a and b (both orders) bitwise against the reference
void check_against_reference(const Tensor& a, const Tensor& b) {
	std::vector<size_t> out_shape(std::max(a.ndim(), b.ndim()), 1);
	for (const Tensor* t : {&a, &b}) {
		for (size_t d = 0; d < t->ndim(); d++) {
			size_t& o = out_shape[d + out_shape.size() - t->ndim()];
			o = std::max(o, t->shape()[d]);
		}
	}
	using Fn = Tensor (*)(const Tensor&, const Tensor&);
	using Ref = float (*)(float, float);
	Fn fns[] = {add, sub, mul, div};
	Ref refs[] = {+[](float x, float y) { return x + y; }, +[](float x, float y) { return x - y; },
	              +[](float x, float y) { return x * y; }, +[](float x, float y) { return x / y; }};
	for (int k = 0; k < 4; k++) {
		Tensor ab = fns[k](a, b), ba = fns[k](b, a);
		CHECK(ab.shape() == out_shape);
		CHECK(ba.shape() == out_shape);
		CHECK(ab.data() == reference_broadcast(a, b, refs[k], out_shape).data());
		CHECK(ba.data() == reference_broadcast(b, a, refs[k], out_shape).data());
	}
}
}  // namespace

TEST_CASE("elementwise broadcast is bitwise equal to the reference for many shape pairs") {
	unsigned seed = 1;
	for (size_t n : {size_t{1}, size_t{7}, size_t{33}, size_t{257}}) {
		size_t c = n > 33 ? 5 : n, h = n > 33 ? 9 : 3;
		std::vector<ShapePair> cases = {
			{{2, c, h, n}, {1, c, 1, 1}},   // BatchNorm statistics
			{{2, c, h, n}, {c, 1, 1}},      // rank mismatch, channel
			{{2, c, h, n}, {1, 1, h, n}},   // trailing block with leading ones
			{{2, c, h, n}, {2, 1, 1, 1}},   // per-sample
			{{2, c, h, n}, {1}},            // scalar
			{{2, c, h, n}, {}},             // rank-0
			{{2, c, h, n}, {2, c, h, n}},   // same shape
			{{2, c, h, n}, {n}},            // trailing 1-D
			{{2, c, h, n}, {2, 1, h, 1}},   // interleaved broadcast dims
			{{2, c, h, n}, {1, c, h, 1}},
			{{c, 1, n}, {1, h, 1}},         // both expand
			{{c, 1}, {1, n}},               // outer product
			{{n, 1, 1}, {1, c, h}},
			{{1, 1, 1}, {c, h, n}},         // size-1 dims everywhere
			{{1, 1, 1, 1}, {1, 1, 1, 1}},
			{{n}, {n}},                     // 1-D
			{{n}, {1}},
			{{1}, {1}},
			{{3, 1, 5, 1, n}, {1, c, 1, h, 1}},
		};
		for (const auto& cs : cases) {
			Tensor a(cs.first), b(cs.second);
			fill_random(a, seed++);
			fill_random(b, seed++);
			check_against_reference(a, b);
		}
	}
}

TEST_CASE("elementwise broadcast over a large tensor is identical for any thread count") {
	Tensor a({8, 16, 48, 48}), b({1, 16, 1, 1}), c({16, 1, 1}), d({1, 1, 48, 48});
	fill_random(a, 11);
	fill_random(b, 12);
	fill_random(c, 13);
	fill_random(d, 14);
	size_t saved = nn::kernels::num_threads();
	nn::kernels::set_num_threads(1);
	Tensor r1[] = {add(a, b), sub(c, a), mul(a, b), div(a, d), mul(b, c)};
	for (size_t threads : {size_t{3}, size_t{8}}) {
		nn::kernels::set_num_threads(threads);
		Tensor r[] = {add(a, b), sub(c, a), mul(a, b), div(a, d), mul(b, c)};
		for (int k = 0; k < 5; k++) CHECK(r[k].data() == r1[k].data());
	}
	nn::kernels::set_num_threads(saved);
	CHECK(r1[0].data() == reference_broadcast(a, b, +[](float x, float y) { return x + y; }, a.shape()).data());
	CHECK(r1[3].data() == reference_broadcast(a, d, +[](float x, float y) { return x / y; }, a.shape()).data());
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

TEST_CASE("matmul with empty matrices") {
	Tensor a = matmul(Tensor({0, 3}), Tensor({3, 2}));
	CHECK(a.shape() == std::vector<size_t>{0, 2});
	Tensor b = matmul(Tensor({2, 0}), Tensor({0, 3}));
	CHECK(b.shape() == std::vector<size_t>{2, 3});
	for (float v : b.data()) CHECK(v == 0.0f);
	Tensor c = matmul(Tensor({2, 3}), Tensor({3, 0}));
	CHECK(c.shape() == std::vector<size_t>{2, 0});
}

TEST_CASE("matmul inner dimension mismatch throws") {
	Tensor a({2, 3});
	Tensor b({4, 2});
	CHECK_THROWS_AS(matmul(a, b), TensorShapeError);
}

namespace {

Tensor ramp(std::vector<size_t> shape, float phase) {
	size_t n = 1;
	for (size_t d : shape) n *= d;
	std::vector<float> v(n);
	for (size_t i = 0; i < n; i++) v[i] = std::sin(0.37f * static_cast<float>(i) + phase);
	return Tensor(shape, v);
}

// Swap the last two dims of t
Tensor swap_last(const Tensor& t) {
	std::vector<size_t> perm(t.ndim());
	for (size_t i = 0; i < perm.size(); i++) perm[i] = i;
	std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
	return t.transpose(perm);
}

// Check matmul(a, b, ta, tb) against matmul of explicitly transposed operands. a_shape
// and b_shape are the shapes of op(A) and op(B); the stored operands are transposed back
void check_flags(std::vector<size_t> a_shape, std::vector<size_t> b_shape) {
	for (int f = 0; f < 4; f++) {
		bool ta = f & 1, tb = f & 2;
		Tensor a = ramp(a_shape, 0.1f), b = ramp(b_shape, 0.7f);
		Tensor ref = matmul(a, b);
		Tensor sa = ta ? swap_last(a) : a, sb = tb ? swap_last(b) : b;
		Tensor got = matmul(sa, sb, ta, tb);
		REQUIRE(got.shape() == ref.shape());
		for (size_t i = 0; i < ref.size(); i++) {
			CHECK(got.data()[i] == doctest::Approx(ref.data()[i]).epsilon(1e-4).scale(1.0));
		}
	}
}

}  // namespace

TEST_CASE("matmul with transpose flags matches explicit transpose") {
	check_flags({7, 33}, {33, 5});
	check_flags({4, 7, 33}, {4, 33, 5});
	check_flags({1, 7, 33}, {4, 33, 5});
	check_flags({4, 7, 33}, {33, 5});
}

TEST_CASE("matmul with transpose flags shape errors") {
	Tensor a({3, 4}), b({3, 5});
	CHECK_THROWS_AS(matmul(a, b), TensorShapeError);
	CHECK_THROWS_AS(matmul(a, b, false, false), TensorShapeError);
	Tensor c = matmul(a, b, true, false);
	CHECK(c.shape() == std::vector<size_t>{4, 5});
	CHECK_THROWS_AS(matmul(a, b, false, true), TensorShapeError);
	CHECK_THROWS_AS(matmul(Tensor({3}), b, false, false), TensorShapeError);
	CHECK_THROWS_AS(matmul(Tensor({2, 3, 4}), Tensor({3, 4, 5}), false, false), TensorShapeError);
}

TEST_CASE("operator+ - * / match free functions") {
	Tensor a({2}, std::vector<float>{6, 8});
	Tensor b({2}, std::vector<float>{2, 4});
	CHECK((a + b).get({0}) == 8.0f);
	CHECK((a - b).get({0}) == 4.0f);
	CHECK((a * b).get({1}) == 32.0f);
	CHECK((a / b).get({0}) == 3.0f);
}

TEST_CASE("operator* with scalar, both orders") {
	Tensor a({2}, std::vector<float>{1, 2});
	CHECK((a * 3.0f).get({1}) == 6.0f);
	CHECK((3.0f * a).get({1}) == 6.0f);
}

TEST_CASE("unary operator- negates every element") {
	Tensor a({2}, std::vector<float>{1, -2});
	Tensor n = -a;
	CHECK(n.get({0}) == -1.0f);
	CHECK(n.get({1}) == 2.0f);
}

TEST_CASE("compound assignment operators mutate in place") {
	Tensor a({2}, std::vector<float>{6, 8});
	Tensor b({2}, std::vector<float>{2, 4});
	a += b;
	CHECK(a.get({0}) == 8.0f);
	a -= b;
	CHECK(a.get({0}) == 6.0f);
	a *= b;
	CHECK(a.get({0}) == 12.0f);
	a /= b;
	CHECK(a.get({0}) == doctest::Approx(6.0f));
	a *= 2.0f;
	CHECK(a.get({0}) == doctest::Approx(12.0f));
}

TEST_CASE("compound assignment with different element count throws") {
	Tensor a({2});
	Tensor b({3});
	CHECK_THROWS_AS(a += b, TensorShapeError);
	CHECK_THROWS_AS(a -= b, TensorShapeError);
	CHECK_THROWS_AS(a *= b, TensorShapeError);
	CHECK_THROWS_AS(a /= b, TensorShapeError);
}

TEST_CASE("compound assignment with same element count but different shape throws") {
	Tensor a({2, 3});
	Tensor b({3, 2});
	CHECK_THROWS_AS(a += b, TensorShapeError);
	Tensor c({6});
	CHECK_THROWS_AS(a += c, TensorShapeError);
}

TEST_CASE("zero_ sets every element to zero in place") {
	Tensor a({3}, 5.0f);
	a.zero_();
	for (float v : a.data()) CHECK(v == 0.0f);
}

TEST_CASE("in-place add/sub on large tensors is identical for any thread count") {
	size_t n = 100003;
	Tensor a({n}), b({n});
	for (size_t i = 0; i < n; i++) {
		a.data()[i] = static_cast<float>(i % 13) * 0.37f;
		b.data()[i] = static_cast<float>(i % 7) * 0.11f;
	}
	std::vector<float> expect_add(n), expect_sub(n);
	for (size_t i = 0; i < n; i++) {
		expect_add[i] = a.data()[i] + b.data()[i];
		expect_sub[i] = a.data()[i] - b.data()[i];
	}
	size_t saved = nn::kernels::num_threads();
	for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
		nn::kernels::set_num_threads(threads);
		Tensor x = a, y = a;
		x += b;
		y -= b;
		CHECK(x.data() == expect_add);
		CHECK(y.data() == expect_sub);
	}
	nn::kernels::set_num_threads(saved);
}

TEST_CASE("transpose_into copies, adds and stores like transpose") {
	Tensor t({2, 3, 4});
	for (size_t i = 0; i < t.size(); i++) t.data()[i] = static_cast<float>(i) - 5.0f;
	t.data()[0] = -0.0f;
	const std::vector<size_t> perm{2, 0, 1};
	Tensor want = t.transpose(perm);
	std::vector<float> copy(t.size(), 99.0f), added(t.size(), 1.0f), stored(t.size(), 99.0f);
	t.transpose_into(perm, copy.data(), Tensor::Combine::kCopy);
	t.transpose_into(perm, added.data(), Tensor::Combine::kAdd);
	t.transpose_into(perm, stored.data(), Tensor::Combine::kAddToZero);
	for (size_t i = 0; i < t.size(); i++) {
		CHECK(copy[i] == want.data()[i]);
		CHECK(added[i] == want.data()[i] + 1.0f);
		CHECK(stored[i] == want.data()[i]);
	}
	// -0 is kept by a copy and becomes +0 when stored as a sum with zero
	CHECK(std::signbit(copy[0]));
	CHECK(!std::signbit(stored[0]));
	CHECK_THROWS_AS(t.transpose_into({1, 0}, copy.data(), Tensor::Combine::kCopy), TensorShapeError);
}
