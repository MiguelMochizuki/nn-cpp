/**
 * test_autograd.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the autograd Value/NodeImpl core and ops
 */

#include "doctest/doctest.h"
#include "nn/autograd/value.hpp"
#include "nn/autograd/ops.hpp"
#include "gradcheck.hpp"
#include "nn/kernels/parallel.hpp"
#include "nn/random.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace nn;

TEST_CASE("leaf construction exposes data and requires_grad") {
	Value v = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	CHECK(v.data().get({0}) == 1.0f);
	CHECK(v.requires_grad());
}

TEST_CASE("leaf grad starts at zero") {
	Value v = Value::leaf(Tensor({2}, 5.0f), true);
	CHECK(v.grad().get({0}) == 0.0f);
	CHECK(v.grad().get({1}) == 0.0f);
}

TEST_CASE("zero_grad resets grad to zero") {
	Value v = Value::leaf(Tensor({1}, 1.0f), true);
	v.grad().at({0}) = 3.0f;
	v.zero_grad();
	CHECK(v.grad().get({0}) == 0.0f);
}

TEST_CASE("backward on a non-scalar Value throws") {
	Value v = Value::leaf(Tensor({2}, 1.0f), true);
	CHECK_THROWS_AS(v.backward(), AutogradShapeError);
}

TEST_CASE("backward on a leaf scalar seeds its own grad to one") {
	Value v = Value::leaf(Tensor({1}, 5.0f), true);
	v.backward();
	CHECK(v.grad().get({0}) == 1.0f);
}

TEST_CASE("node backward runs its backward_fn with the seeded grad") {
	Value parent = Value::leaf(Tensor({1}, 2.0f), true);
	Value child = Value::node(Tensor({1}, 4.0f), {parent}, [parent](const Tensor& grad_out) mutable {
		parent.grad() = add(parent.grad(), scale(grad_out, 3.0f));
	});
	child.backward();
	CHECK(parent.grad().get({0}) == doctest::Approx(3.0f));
}

TEST_CASE("live_node_count returns to zero once all Values go out of scope") {
	size_t before = Value::live_node_count();
	{
		Value a = Value::leaf(Tensor({1}, 1.0f), true);
		Value b = Value::node(Tensor({1}, 2.0f), {a}, [](const Tensor&) {});
		CHECK(Value::live_node_count() == before + 2);
	}
	CHECK(Value::live_node_count() == before);
}

TEST_CASE("sum reduces to a scalar and backward gives grad 1 everywhere") {
	Value x = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	Value s = sum(x);
	CHECK(s.data().get({0}) == 6.0f);
	s.backward();
	CHECK(x.grad().get({0}) == 1.0f);
	CHECK(x.grad().get({2}) == 1.0f);
}

TEST_CASE("add gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(add(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("mul gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(mul(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("sub gradcheck") {
	Value b = Value::leaf(Tensor({3}, std::vector<float>{4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(sub(a, b)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("scale gradcheck") {
	auto f = [](Value a) { return sum(scale(a, 2.5f)); };
	Value a = Value::leaf(Tensor({3}, std::vector<float>{1, 2, 3}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("broadcasted add sums incoming grad over broadcast axes into the smaller parent") {
	Value bias = Value::leaf(Tensor({3}, 0.0f), true);
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 1, 1, 1, 1, 1}), false);
	Value y = sum(add(x, bias));
	y.backward();
	CHECK(bias.grad().get({0}) == doctest::Approx(2.0f));
	CHECK(bias.grad().get({1}) == doctest::Approx(2.0f));
}

TEST_CASE("broadcasted mul gradcheck") {
	Value b = Value::leaf(Tensor({1, 3}, std::vector<float>{2, -1, 0.5f}), false);
	auto f = [b](Value a) mutable { return sum(mul(a, b)); };
	Value a = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}), true);
	CHECK(gradcheck(f, a));
}

TEST_CASE("diamond: one leaf feeding two ops accumulates both gradients") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value branch_a = scale(w, 2.0f);
	Value branch_b = scale(w, 5.0f);
	Value total = sum(add(branch_a, branch_b));
	total.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(7.0f));
}

TEST_CASE("calling backward twice without zero_grad accumulates") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value out1 = scale(w, 2.0f);
	out1.backward();
	Value out2 = scale(w, 2.0f);
	out2.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(4.0f));
}

TEST_CASE("calling backward twice on the SAME graph accumulates onto the leaf without compounding intermediate grads") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value a = scale(w, 2.0f);
	Value b = scale(a, 5.0f);

	b.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(10.0f));

	b.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(20.0f));
}

TEST_CASE("backward, zero_grad, backward again on the same graph gives one pass worth of gradient") {
	Value w = Value::leaf(Tensor({1}, 3.0f), true);
	Value a = scale(w, 2.0f);
	Value b = scale(a, 5.0f);

	b.backward();
	w.zero_grad();
	b.backward();
	CHECK(w.grad().get({0}) == doctest::Approx(10.0f));
}

TEST_CASE("graph nodes are freed once all Values referencing them go out of scope") {
	size_t before = Value::live_node_count();
	{
		Value w = Value::leaf(Tensor({1}, 1.0f), true);
		Value branch_a = scale(w, 2.0f);
		Value branch_b = scale(w, 3.0f);
		Value total = sum(add(branch_a, branch_b));
		total.backward();
	}
	CHECK(Value::live_node_count() == before);
}

TEST_CASE("matmul gradcheck") {
	Value b = Value::leaf(Tensor({3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}), false);
	auto f = [b](Value a) mutable { return sum(matmul(a, b)); };
	Value a = Value::leaf(Tensor({2, 3}, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}), true);
	CHECK(gradcheck(f, a));
}

namespace {

/**
 * Builds a leaf Tensor of the given shape filled with a deterministic sine pattern
 *
 * Parameters:
 * vector<size_t> shape: Shape of the tensor
 * float phase: Offset into the pattern, so different tensors differ
 *
 * Returns Tensor: The filled tensor
 */
Tensor sine_tensor(const std::vector<size_t>& shape, float phase) {
	Tensor t(shape);
	for (size_t i = 0; i < t.size(); i++) t.data()[i] = std::sin(0.7f * i + phase);
	return t;
}

/**
 * Scalar loss sum(out * w) with a fixed non-uniform w, so an all-ones upstream
 * gradient cannot hide an orientation bug
 *
 * Parameters:
 * Value out: The op output
 *
 * Returns Value: Scalar loss
 */
Value weighted_sum(const Value& out) {
	Tensor w(out.data().shape());
	for (size_t i = 0; i < w.size(); i++) w.data()[i] = 0.3f + 0.1f * i;
	return sum(mul(out, Value::leaf(w, false)));
}

}  // namespace

TEST_CASE("matmul gradcheck wrt b") {
	Value a = Value::leaf(sine_tensor({2, 3}, 0.1f), false);
	Value b = Value::leaf(sine_tensor({3, 2}, 0.9f), true);
	auto f = [a](Value x) mutable { return weighted_sum(matmul(a, x)); };
	CHECK(gradcheck(f, b));
}

TEST_CASE("matmul batched gradcheck with 2D b reduces the b gradient") {
	Value a = Value::leaf(sine_tensor({2, 7, 3}, 0.1f), true);
	Value b = Value::leaf(sine_tensor({3, 5}, 0.9f), true);
	auto fa = [b](Value x) mutable { return weighted_sum(matmul(x, b)); };
	CHECK(gradcheck(fa, a));
	auto fb = [a](Value x) mutable { return weighted_sum(matmul(a, x)); };
	CHECK(gradcheck(fb, b));
}

TEST_CASE("matmul_bt batched gradcheck with 2D b reduces the b gradient") {
	Value a = Value::leaf(sine_tensor({2, 7, 3}, 0.1f), true);
	Value b = Value::leaf(sine_tensor({5, 3}, 0.9f), true);
	auto fa = [b](Value x) mutable { return weighted_sum(matmul_bt(x, b)); };
	CHECK(gradcheck(fa, a));
	auto fb = [a](Value x) mutable { return weighted_sum(matmul_bt(a, x)); };
	CHECK(gradcheck(fb, b));
}

TEST_CASE("matmul_bt gradcheck") {
	Value b = Value::leaf(Tensor({4, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1}), true);
	Value a = Value::leaf(Tensor({2, 3}, std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}), true);
	auto fa = [b](Value x) mutable { return sum(matmul_bt(x, b)); };
	CHECK(gradcheck(fa, a));
	auto fb = [a](Value x) mutable { return sum(matmul_bt(a, x)); };
	CHECK(gradcheck(fb, b));
}

TEST_CASE("matmul_bt forward equals matmul with explicit transpose") {
	Value a = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}), false);
	Value b = Value::leaf(Tensor({4, 3}, std::vector<float>{1, 0, 2, 3, 1, 0, -1, 2, 2, 0.5f, 0.5f, 0.5f}), false);
	Tensor got = matmul_bt(a, b).data();
	Tensor want = matmul(a.data(), b.data().transpose({1, 0}));
	CHECK(got.shape() == want.shape());
	for (size_t i = 0; i < want.size(); i++) CHECK(got.data()[i] == doctest::Approx(want.data()[i]).epsilon(1e-4));
}

TEST_CASE("matmul forward matches Tensor matmul") {
	Value a = Value::leaf(Tensor({2, 2}, std::vector<float>{1, 2, 3, 4}), false);
	Value b = Value::leaf(Tensor({2, 2}, std::vector<float>{5, 6, 7, 8}), false);
	Value c = matmul(a, b);
	CHECK(c.data().get({0, 0}) == doctest::Approx(19.0f));
	CHECK(c.data().get({1, 1}) == doctest::Approx(50.0f));
}

TEST_CASE("relu zeroes negative inputs and passes positive ones through") {
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1, 0, 2}), true);
	Value y = relu(x);
	CHECK(y.data().get({0}) == 0.0f);
	CHECK(y.data().get({1}) == 0.0f);
	CHECK(y.data().get({2}) == 2.0f);
}

TEST_CASE("relu gradcheck away from the kink") {
	auto f = [](Value x) { return sum(relu(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-2.0f, 1.0f, 3.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("relu backward passes the gradient only where the input is positive and accumulates") {
	for (size_t n : {size_t{1}, size_t{7}, size_t{33}, size_t{257}}) {
		Tensor v({n});
		for (size_t i = 0; i < n; i++) v.data()[i] = (i % 3 == 0) ? -1.0f : (i % 3 == 1 ? 0.0f : 2.0f);
		Value x = Value::leaf(v, true);
		x.grad().data().assign(n, 5.0f);  // pre-existing gradient must be kept, not overwritten
		sum(scale(relu(x), 3.0f)).backward();
		for (size_t i = 0; i < n; i++) {
			CHECK(relu(x).data().data()[i] == (v.data()[i] > 0.0f ? v.data()[i] : 0.0f));
			CHECK(x.grad().data()[i] == doctest::Approx(v.data()[i] > 0.0f ? 8.0f : 5.0f));
		}
	}
}

TEST_CASE("relu forward and backward on a large tensor are identical for any thread count") {
	size_t n = 100003;
	Tensor v({n});
	for (size_t i = 0; i < n; i++) v.data()[i] = static_cast<float>(static_cast<int>(i % 11) - 5);
	size_t saved = nn::kernels::num_threads();
	for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
		nn::kernels::set_num_threads(threads);
		Value x = Value::leaf(v, true);
		sum(relu(x)).backward();
		for (size_t i = 0; i < n; i++) {
			CHECK(x.grad().data()[i] == (v.data()[i] > 0.0f ? 1.0f : 0.0f));
		}
	}
	nn::kernels::set_num_threads(saved);
}

TEST_CASE("sigmoid gradcheck") {
	auto f = [](Value x) { return sum(sigmoid(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 1.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("tanh gradcheck") {
	auto f = [](Value x) { return sum(tanh(x)); };
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 1.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("sigmoid and tanh backward use the stored forward value") {
	Value x = Value::leaf(Tensor({3}, std::vector<float>{-1.0f, 0.0f, 2.0f}), true);
	Value s = sigmoid(x);
	Value t = tanh(x);
	sum(s + t).backward();
	for (size_t i = 0; i < 3; i++) {
		float v = x.data().data()[i];
		float sv = 1.0f / (1.0f + std::exp(-v));
		float tv = std::tanh(v);
		CHECK(s.data().data()[i] == doctest::Approx(sv));
		CHECK(t.data().data()[i] == doctest::Approx(tv));
		CHECK(x.grad().data()[i] == doctest::Approx(sv * (1.0f - sv) + 1.0f - tv * tv));
	}
}

TEST_CASE("add, sub and mul backward match the expected gradient for same-shape and broadcast operands") {
	for (size_t n : {size_t{1}, size_t{7}, size_t{33}, size_t{257}}) {
		Tensor av({3, n}), bv({n});
		for (size_t i = 0; i < av.size(); i++) av.data()[i] = static_cast<float>(i % 5) - 2.0f;
		for (size_t i = 0; i < n; i++) bv.data()[i] = static_cast<float>(i % 3) + 0.5f;
		Value a = Value::leaf(av, true), same = Value::leaf(av, true), b = Value::leaf(bv, true);
		sum(add(a, b) - same).backward();  // d/da = 1, d/db = sum over rows = 3, d/dsame = -1
		for (float g : a.grad().data()) CHECK(g == 1.0f);
		for (float g : same.grad().data()) CHECK(g == -1.0f);
		for (float g : b.grad().data()) CHECK(g == 3.0f);
		Value a2 = Value::leaf(av, true), b2 = Value::leaf(bv, true);
		sum(mul(a2, b2)).backward();  // d/da = b broadcast, d/db = column sums of a
		for (size_t i = 0; i < av.size(); i++) CHECK(a2.grad().data()[i] == bv.data()[i % n]);
		for (size_t j = 0; j < n; j++) {
			float col = av.data()[j] + av.data()[n + j] + av.data()[2 * n + j];
			CHECK(b2.grad().data()[j] == doctest::Approx(col));
		}
	}
}

TEST_CASE("softmax rows sum to one") {
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 1, 1, 1}), true);
	Value y = softmax(x, 1);
	float row0 = y.data().get({0, 0}) + y.data().get({0, 1}) + y.data().get({0, 2});
	CHECK(row0 == doctest::Approx(1.0f));
}

TEST_CASE("softmax gradcheck") {
	auto f = [](Value x) { return sum(softmax(x, 1)); };
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 0.5f, -1.0f, 2.0f}), true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("conv2d forward shape and value") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{1, 0, 0, 1}), false);
	Value b = Value::leaf(Tensor({1}, 0.0f), false);
	Value y = conv2d(x, w, b, 1, 0);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == doctest::Approx(1.0f + 5.0f));
	CHECK(y.data().get({0, 0, 1, 1}) == doctest::Approx(5.0f + 9.0f));
}

TEST_CASE("conv2d gradcheck wrt input, with padding") {
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), false);
	Value b = Value::leaf(Tensor({1}, 0.1f), false);
	auto f = [w, b](Value x) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), true);
	CHECK(gradcheck(f, x, 1e-3f, 0.1f));
}

TEST_CASE("conv2d gradcheck wrt weight") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value b = Value::leaf(Tensor({1}, 0.1f), false);
	auto f = [x, b](Value w) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), true);
	CHECK(gradcheck(f, w, 1e-3f, 0.1f));
}

TEST_CASE("conv2d gradcheck wrt bias") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), false);
	Value w = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{0.5f, -0.5f, 1.0f, 0.2f}), false);
	auto f = [x, w](Value b) mutable { return sum(conv2d(x, w, b, 1, 1)); };
	Value b = Value::leaf(Tensor({1}, 0.1f), true);
	CHECK(gradcheck(f, b, 1e-3f, 0.1f));
}

TEST_CASE("conv2d gradcheck with uneven output size, stride 2, multiple channels and a batch") {
	Value w = Value::leaf(Tensor::random({2, 3, 3, 3}, -0.3f, 0.3f), false);
	Value b = Value::leaf(Tensor::random({2}, -0.1f, 0.1f), false);
	auto f = [w, b](Value x) mutable { return sum(conv2d(x, w, b, 2, 1)); };
	Value x = Value::leaf(Tensor::random({2, 3, 7, 6}, -1.0f, 1.0f), true);
	Value y = conv2d(x, w, b, 2, 1);
	CHECK(y.data().shape() == std::vector<size_t>{2, 2, 4, 3});
	CHECK(gradcheck(f, x, 1e-3f, 0.15f));
}

namespace {
// Direct convolution, window by window, as the reference for the im2col path
Tensor reference_conv(const Tensor& x, const Tensor& w, const Tensor& b, size_t stride, size_t pad) {
	size_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
	size_t Cout = w.shape()[0], kh = w.shape()[2], kw = w.shape()[3];
	size_t Ho = (H + 2 * pad - kh) / stride + 1, Wo = (W + 2 * pad - kw) / stride + 1;
	Tensor out({N, Cout, Ho, Wo});
	for (size_t n = 0; n < N; n++)
		for (size_t co = 0; co < Cout; co++)
			for (size_t oh = 0; oh < Ho; oh++)
				for (size_t ow = 0; ow < Wo; ow++) {
					float acc = b.data()[co];
					for (size_t c = 0; c < Cin; c++)
						for (size_t i = 0; i < kh; i++)
							for (size_t j = 0; j < kw; j++) {
								long ih = static_cast<long>(oh * stride + i) - static_cast<long>(pad);
								long iw = static_cast<long>(ow * stride + j) - static_cast<long>(pad);
								if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) || iw >= static_cast<long>(W)) continue;
								acc += x.get({n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)}) * w.get({co, c, i, j});
							}
					out.at({n, co, oh, ow}) = acc;
				}
	return out;
}

Tensor patterned(std::vector<size_t> shape, float scale_by) {
	Tensor t(std::move(shape));
	for (size_t i = 0; i < t.size(); i++) t.data()[i] = (static_cast<float>((i * 37) % 19) - 9.0f) * scale_by;
	return t;
}
}  // namespace

TEST_CASE("conv2d matches a direct convolution on odd sizes and its results do not depend on the thread count") {
	struct Cfg { size_t N, Cin, H, W, Cout, k, stride, pad; };
	std::vector<Cfg> cfgs = {
		{1, 1, 1, 1, 1, 1, 1, 0},   // 1x1 input and kernel
		{7, 2, 5, 7, 3, 1, 1, 0},   // 1x1 kernel
		{1, 1, 2, 2, 2, 5, 1, 2},   // kernel larger than the input, padded
		{7, 2, 9, 6, 3, 3, 2, 1},   // batch 7, stride 2
		{7, 3, 16, 16, 4, 3, 1, 1}, // large enough to split across threads
		{3, 2, 33, 5, 2, 4, 3, 3},  // padding wider than the stride
	};
	size_t saved = nn::kernels::num_threads();
	for (const Cfg& c : cfgs) {
		Tensor xv = patterned({c.N, c.Cin, c.H, c.W}, 0.07f);
		Tensor wv = patterned({c.Cout, c.Cin, c.k, c.k}, 0.05f);
		Tensor bv = patterned({c.Cout}, 0.1f);
		Tensor expect = reference_conv(xv, wv, bv, c.stride, c.pad);
		Storage out1, gx1, gw1, gb1;
		for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
			nn::kernels::set_num_threads(threads);
			Value x = Value::leaf(xv, true), w = Value::leaf(wv, true), b = Value::leaf(bv, true);
			Value y = conv2d(x, w, b, static_cast<int>(c.stride), static_cast<int>(c.pad));
			REQUIRE(y.data().shape() == expect.shape());
			for (size_t i = 0; i < expect.size(); i++) CHECK(y.data().data()[i] == doctest::Approx(expect.data()[i]).epsilon(1e-4));
			sum(y).backward();
			if (threads == 1) {
				out1 = y.data().data();
				gx1 = x.grad().data();
				gw1 = w.grad().data();
				gb1 = b.grad().data();
			} else {
				CHECK(y.data().data() == out1);
				CHECK(x.grad().data() == gx1);
				CHECK(w.grad().data() == gw1);
				CHECK(b.grad().data() == gb1);
			}
		}
	}
	nn::kernels::set_num_threads(saved);
}

TEST_CASE("maxpool2d forward and backward are identical for any thread count and route gradient to the max") {
	size_t saved = nn::kernels::num_threads();
	for (size_t k : {size_t{1}, size_t{2}, size_t{3}}) {
		for (size_t stride : {size_t{1}, size_t{2}}) {
			Tensor xv = patterned({7, 3, 12, 9}, 0.31f);
			Storage out1, g1;
			for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
				nn::kernels::set_num_threads(threads);
				Value x = Value::leaf(xv, true);
				Value y = maxpool2d(x, k, stride);
				sum(y).backward();
				// total gradient mass equals the number of windows, each routed to one input cell
				float total = 0.0f;
				for (float g : x.grad().data()) total += g;
				CHECK(total == doctest::Approx(static_cast<float>(y.data().size())));
				for (size_t i = 0; i < y.data().size(); i++) {
					size_t ow_n = y.data().shape()[3], oh_n = y.data().shape()[2];
					size_t pl = i / (oh_n * ow_n), r = i % (oh_n * ow_n), oh = r / ow_n, ow = r % ow_n;
					float best = -1e30f;
					for (size_t a = 0; a < k; a++)
						for (size_t bb = 0; bb < k; bb++)
							best = std::max(best, xv.data()[pl * 12 * 9 + (oh * stride + a) * 9 + ow * stride + bb]);
					CHECK(y.data().data()[i] == best);
				}
				if (threads == 1) {
					out1 = y.data().data();
					g1 = x.grad().data();
				} else {
					CHECK(y.data().data() == out1);
					CHECK(x.grad().data() == g1);
				}
			}
		}
	}
	nn::kernels::set_num_threads(saved);
}

TEST_CASE("conv2d throws when bias size does not match out_channels") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, 1.0f), false);
	Value w = Value::leaf(Tensor({2, 1, 3, 3}, 1.0f), false);
	Value b = Value::leaf(Tensor({3}, 0.0f), false);
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0), TensorShapeError);
}

TEST_CASE("conv2d throws when kernel is larger than the padded input") {
	Value x = Value::leaf(Tensor({1, 1, 2, 2}, 1.0f), false);
	Value w = Value::leaf(Tensor({1, 1, 5, 5}, 1.0f), false);
	Value b = Value::leaf(Tensor({1}, 0.0f), false);
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0), TensorShapeError);
}

TEST_CASE("conv2d throws on zero stride") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, 1.0f), false);
	Value w = Value::leaf(Tensor({1, 1, 3, 3}, 1.0f), false);
	Value b = Value::leaf(Tensor({1}, 0.0f), false);
	CHECK_THROWS_AS(conv2d(x, w, b, 0, 0), TensorShapeError);
}

TEST_CASE("conv2d throws when input channels do not match weight") {
	Value x = Value::leaf(Tensor({1, 2, 4, 4}, 1.0f), false);
	Value w = Value::leaf(Tensor({1, 3, 3, 3}, 1.0f), false);
	Value b = Value::leaf(Tensor({1}, 0.0f), false);
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0), TensorShapeError);
}

TEST_CASE("maxpool2d picks the max of each window") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, std::vector<float>{
	                           1, 2, 5, 6,
	                           3, 4, 7, 8,
	                           9, 10, 13, 14,
	                           11, 12, 15, 16}),
	                       true);
	Value y = maxpool2d(x, 2, 2);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == 4.0f);
	CHECK(y.data().get({0, 0, 0, 1}) == 8.0f);
	CHECK(y.data().get({0, 0, 1, 0}) == 12.0f);
	CHECK(y.data().get({0, 0, 1, 1}) == 16.0f);
}

TEST_CASE("maxpool2d routes gradient only to the max element of each window") {
	Value x = Value::leaf(Tensor({1, 1, 2, 2}, std::vector<float>{1, 5, 3, 2}), true);
	Value y = maxpool2d(x, 2, 2);
	y.backward();
	CHECK(x.grad().get({0, 0, 0, 0}) == 0.0f);
	CHECK(x.grad().get({0, 0, 0, 1}) == 1.0f);
	CHECK(x.grad().get({0, 0, 1, 0}) == 0.0f);
	CHECK(x.grad().get({0, 0, 1, 1}) == 0.0f);
}

TEST_CASE("maxpool2d gradcheck") {
	auto f = [](Value x) { return sum(maxpool2d(x, 2, 2)); };
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, std::vector<float>{
	                           1, 2, 5, 6,
	                           3, 4, 7, 8,
	                           9, 10, 13, 14,
	                           11, 12, 15, 16}),
	                       true);
	CHECK(gradcheck(f, x));
}

TEST_CASE("maxpool2d floors a non-integer output size, ignoring the trailing row and column") {
	Value x = Value::leaf(Tensor({1, 1, 5, 5}, std::vector<float>{
	                           1, 2, 3, 4, 100,
	                           5, 6, 7, 8, 100,
	                           9, 10, 11, 12, 100,
	                           13, 14, 15, 16, 100,
	                           100, 100, 100, 100, 100}),
	                       false);
	Value y = maxpool2d(x, 2, 2);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().get({0, 0, 0, 0}) == 6.0f);
	CHECK(y.data().get({0, 0, 1, 1}) == 16.0f);
}

TEST_CASE("maxpool2d throws when kernel is larger than the input") {
	Value x = Value::leaf(Tensor({1, 1, 2, 2}, 1.0f), false);
	CHECK_THROWS_AS(maxpool2d(x, 5, 1), TensorShapeError);
}

TEST_CASE("maxpool2d throws on zero stride") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, 1.0f), false);
	CHECK_THROWS_AS(maxpool2d(x, 2, 0), TensorShapeError);
}

TEST_CASE("cross_entropy_loss matches hand-computed value for a 2-class batch") {
	Value logits = Value::leaf(Tensor({2, 2}, std::vector<float>{1.0f, 1.0f, 2.0f, 0.0f}), true);
	Tensor targets({2}, std::vector<float>{0.0f, 0.0f});
	Value loss = cross_entropy_loss(logits, targets);
	float expected = (0.693147f + 0.126928f) / 2.0f;
	CHECK(loss.data().get({0}) == doctest::Approx(expected).epsilon(0.01));
}

TEST_CASE("cross_entropy_loss gradcheck") {
	Tensor targets({2}, std::vector<float>{1.0f, 0.0f});
	auto f = [targets](Value logits) mutable { return cross_entropy_loss(logits, targets); };
	Value logits = Value::leaf(Tensor({2, 3}, std::vector<float>{0.2f, 1.5f, -0.3f, 1.0f, -1.0f, 0.5f}), true);
	CHECK(gradcheck(f, logits));
}

TEST_CASE("cross_entropy_loss throws when logits are not 2-D") {
	Value logits = Value::leaf(Tensor({6}, 0.0f), true);
	Tensor targets({2}, 0.0f);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets), TensorShapeError);
}

TEST_CASE("cross_entropy_loss throws when targets count does not match the batch size") {
	Value logits = Value::leaf(Tensor({2, 3}, 0.0f), true);
	Tensor targets({3}, 0.0f);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets), TensorShapeError);
}

TEST_CASE("mse_loss matches hand-computed value") {
	Value pred = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	Value target = Value::leaf(Tensor({2}, std::vector<float>{0.0f, 0.0f}), false);
	Value loss = mse_loss(pred, target);
	CHECK(loss.data().get({0}) == doctest::Approx((1.0f + 4.0f) / 2.0f));
}

TEST_CASE("mse_loss gradcheck") {
	Value target = Value::leaf(Tensor({3}, std::vector<float>{0.5f, -1.0f, 2.0f}), false);
	auto f = [target](Value pred) mutable { return mse_loss(pred, target); };
	Value pred = Value::leaf(Tensor({3}, std::vector<float>{1.0f, 0.0f, 1.5f}), true);
	CHECK(gradcheck(f, pred));
}

TEST_CASE("Value operators match free-function equivalents") {
	Value a = Value::leaf(Tensor({2}, std::vector<float>{6.0f, 8.0f}), true);
	Value b = Value::leaf(Tensor({2}, std::vector<float>{2.0f, 4.0f}), true);
	CHECK((a + b).data().get({0}) == 8.0f);
	CHECK((a - b).data().get({0}) == 4.0f);
	CHECK((a * b).data().get({1}) == 32.0f);
	CHECK((a * 3.0f).data().get({0}) == 18.0f);
	CHECK((3.0f * a).data().get({0}) == 18.0f);
	CHECK((-a).data().get({0}) == -6.0f);
}

TEST_CASE("Value operators are differentiable through backward, including sign") {
	Value a = Value::leaf(Tensor({1}, 2.0f), true);
	Value b = Value::leaf(Tensor({1}, 3.0f), true);
	Value y = a * b + a;
	y.backward();
	CHECK(a.grad().get({0}) == doctest::Approx(4.0f));
	CHECK(b.grad().get({0}) == doctest::Approx(2.0f));
}

TEST_CASE("unary negate and negative-scalar multiply carry correct sign through backward") {
	Value x = Value::leaf(Tensor({1}, 5.0f), true);
	Value neg = -x;
	neg.backward();
	CHECK(x.grad().get({0}) == doctest::Approx(-1.0f));

	Value x2 = Value::leaf(Tensor({1}, 5.0f), true);
	Value scaled = x2 * -3.0f;
	scaled.backward();
	CHECK(x2.grad().get({0}) == doctest::Approx(-3.0f));
}

TEST_CASE("data() and grad() on a temporary Value return copies, on an lvalue return references") {
	static_assert(std::is_same<decltype(std::declval<Value&>().data()), Tensor&>::value, "lvalue data() is a reference");
	static_assert(std::is_same<decltype(std::declval<Value>().data()), Tensor>::value, "temporary data() is a copy");
	static_assert(std::is_same<decltype(std::declval<Value&>().grad()), Tensor&>::value, "lvalue grad() is a reference");
	static_assert(std::is_same<decltype(std::declval<Value>().grad()), Tensor>::value, "temporary grad() is a copy");
	Tensor kept = relu(Value::leaf(Tensor({1}, -2.0f), false)).data();
	CHECK(kept.get({0}) == 0.0f);
}

TEST_CASE("NoGradGuard makes ops return graph-free leaves with identical values") {
	Value w = Value::leaf(Tensor({2, 2}, std::vector<float>{1, -1, 2, 3}), true);
	Value x = Value::leaf(Tensor({1, 2}, std::vector<float>{1, 2}), false);
	Value tracked = relu(matmul(x, w));
	REQUIRE(tracked.requires_grad());
	REQUIRE(!tracked.impl()->parents.empty());

	Value untracked = tracked;
	{
		NoGradGuard guard;
		untracked = relu(matmul(x, w));
	}
	CHECK(!untracked.requires_grad());
	CHECK(untracked.impl()->parents.empty());
	CHECK(untracked.data().data() == tracked.data().data());
}

TEST_CASE("NoGradGuard restores tracking when it goes out of scope, including when nested") {
	Value w = Value::leaf(Tensor({1}, 1.0f), true);
	{
		NoGradGuard outer;
		{
			NoGradGuard inner;
		}
		CHECK(!relu(w).requires_grad());
	}
	CHECK(relu(w).requires_grad());
}

TEST_CASE("relu and maxpool2d backward reject a leaf whose data was reshaped after the forward pass") {
	Value x = Value::leaf(Tensor({3}, 1.0f), true);
	Value y = sum(relu(x));
	x.data() = Tensor({5000}, 1.0f);
	CHECK_THROWS_AS(y.backward(), TensorShapeError);

	Value p = Value::leaf(Tensor({1, 1, 2, 2}, 1.0f), true);
	Value q = sum(maxpool2d(p, 2, 2));
	p.data() = Tensor({1, 1, 4, 4}, 1.0f);
	CHECK_THROWS_AS(q.backward(), TensorShapeError);
}

TEST_CASE("has_grad is false for a non-leaf before backward and true afterwards") {
	Value x = Value::leaf(Tensor({3}, 2.0f), true);
	Value h = relu(x);
	Value y = sum(h);
	CHECK(x.has_grad());
	CHECK_FALSE(h.has_grad());
	y.backward();
	CHECK(h.has_grad());
	CHECK(x.has_grad());
}

TEST_CASE("backward orders a node after all its consumers when paths have different depths") {
	// c feeds both mul(c, c) (a deeper path) and the final mul directly
	auto shared = [](Value x) {
		Value c = mul(x, x);
		return sum(mul(c, mul(c, c)));
	};
	// residual style: h feeds tanh(mul(h, h)) and the final add directly
	auto residual = [](Value x) {
		Value w = Value::leaf(Tensor(x.data().shape(), 0.7f), false);
		Value h = mul(x, w);
		return sum(add(h, tanh(mul(h, h))));
	};
	for (const std::vector<size_t>& shape : {std::vector<size_t>{1}, std::vector<size_t>{7, 3}, std::vector<size_t>{33}}) {
		Tensor t(shape, 0.0f);
		for (size_t i = 0; i < t.size(); i++) t.data()[i] = 0.9f - 0.05f * static_cast<float>(i % 31) * (i % 2 ? 1.0f : -1.0f);
		Value x = Value::leaf(t, true);
		CHECK(gradcheck(shared, x, 1e-3f, 5e-2f));
		CHECK(gradcheck(residual, x, 1e-3f, 5e-2f));
	}
}

TEST_CASE("backward gives the exact gradient for a node consumed at different depths") {
	// f(x) = (x^2)^3 = x^6, df/dx = 6 x^5 = 192 at x = 2
	Value x = Value::leaf(Tensor({1}, 2.0f), true);
	Value c = mul(x, x);
	sum(mul(c, mul(c, c))).backward();
	CHECK(x.grad().data()[0] == doctest::Approx(192.0f));
}

TEST_CASE("backward of mul(x, x) is 2x") {
	Value x = Value::leaf(Tensor({1}, 3.0f), true);
	sum(mul(x, x)).backward();
	CHECK(x.grad().data()[0] == doctest::Approx(6.0f));
}

TEST_CASE("backward on a 4000 node chain does not overflow the stack") {
	Value x = Value::leaf(Tensor({1}, 1.0f), true);
	{
		Value y = x;
		for (int i = 0; i < 4000; i++) y = scale(y, 1.0f);
		y.backward();
	}
	CHECK(x.grad().data()[0] == doctest::Approx(1.0f));
}

namespace {

// Weighted so gradients differ per element and a wrong broadcast cannot cancel out
Value weighted_loss(const Value& y) {
	std::vector<float> w(y.data().size());
	for (size_t i = 0; i < w.size(); i++) w[i] = 0.5f + 0.37f * static_cast<float>(i % 7);
	return sum(mul(y, Value::leaf(Tensor(y.data().shape(), w))));
}

// Positive values in [1, 2) keep float cancellation small in finite differences
Value spread_leaf(std::vector<size_t> shape) {
	size_t n = 1;
	for (size_t d : shape) n *= d;
	std::vector<float> v(n);
	for (size_t i = 0; i < n; i++) v[i] = 1.0f + 0.1f * static_cast<float>((i * 7) % 10);
	return Value::leaf(Tensor(shape, v), true);
}

Value iota_leaf(std::vector<size_t> shape, bool grad = false) {
	size_t n = 1;
	for (size_t d : shape) n *= d;
	std::vector<float> v(n);
	for (size_t i = 0; i < n; i++) v[i] = static_cast<float>(i) + 1.0f;
	return Value::leaf(Tensor(shape, v), grad);
}

}  // namespace

TEST_CASE("axis sum and mean values on a {2,3,4} tensor") {
	Value x = iota_leaf({2, 3, 4});
	// axis 0: x[0,j,k] + x[1,j,k] = (1+13), ...
	Value s0 = sum(x, 0);
	CHECK(s0.data().shape() == std::vector<size_t>{3, 4});
	CHECK(s0.data().get({0, 0}) == 14.0f);
	CHECK(s0.data().get({2, 3}) == 12.0f + 24.0f);
	// axis 1: 1+5+9, ...
	Value s1 = sum(x, 1);
	CHECK(s1.data().shape() == std::vector<size_t>{2, 4});
	CHECK(s1.data().get({0, 0}) == 15.0f);
	CHECK(s1.data().get({1, 3}) == 60.0f);
	// axis 2 and -1 agree: 1+2+3+4 = 10
	Value s2 = sum(x, 2);
	Value sm1 = sum(x, -1);
	CHECK(s2.data().shape() == std::vector<size_t>{2, 3});
	CHECK(s2.data().get({0, 0}) == 10.0f);
	CHECK(s2.data().data() == sm1.data().data());
	Value m2 = mean(x, 2);
	CHECK(m2.data().get({1, 2}) == doctest::Approx(22.5f));
	// keepdim keeps a size-1 axis
	CHECK(sum(x, 0, true).data().shape() == std::vector<size_t>{1, 3, 4});
	CHECK(sum(x, 1, true).data().shape() == std::vector<size_t>{2, 1, 4});
	CHECK(mean(x, -1, true).data().shape() == std::vector<size_t>{2, 3, 1});
	CHECK(sum(x, -3).data().data() == s0.data().data());
	CHECK_THROWS_AS(sum(x, 3), TensorIndexError);
	CHECK_THROWS_AS(mean(x, -4), TensorIndexError);
}

TEST_CASE("axis reductions on 1-D tensors and size-1 axes") {
	Value v = iota_leaf({5});
	CHECK(sum(v, 0).data().shape() == std::vector<size_t>{1});
	CHECK(sum(v, 0).data().get({0}) == 15.0f);
	CHECK(mean(v, -1).data().get({0}) == doctest::Approx(3.0f));
	CHECK(sum(v, 0, true).data().shape() == std::vector<size_t>{1});
	Value one = iota_leaf({3, 1, 2});
	Value s = sum(one, 1);
	CHECK(s.data().shape() == std::vector<size_t>{3, 2});
	CHECK(s.data().data() == std::vector<float>{1, 2, 3, 4, 5, 6});
	CHECK(mean(one, 1, true).data().data() == one.data().data());
}

TEST_CASE("axis sum, axis mean and rsqrt gradchecks") {
	for (std::vector<size_t> shape : {std::vector<size_t>{1}, std::vector<size_t>{7, 3}, std::vector<size_t>{33, 5}, std::vector<size_t>{257, 2}}) {
		for (int axis = 0; axis < static_cast<int>(shape.size()); axis++) {
			for (bool keep : {false, true}) {
				Value x = spread_leaf(shape);
				CHECK(gradcheck([&](Value a) { return weighted_loss(sum(a, axis, keep)); }, x, 1e-2f, 5e-2f));
				x = spread_leaf(shape);
				CHECK(gradcheck([&](Value a) { return weighted_loss(mean(a, axis, keep)); }, x, 1e-2f, 5e-2f));
			}
		}
		Value p = spread_leaf(shape);
		CHECK(gradcheck([](Value a) { return weighted_loss(rsqrt(a)); }, p, 1e-2f, 5e-2f));
	}
	Value r = Value::leaf(Tensor({2}, std::vector<float>{4.0f, 0.25f}));
	CHECK(rsqrt(r).data().data() == std::vector<float>{0.5f, 2.0f});
}

TEST_CASE("axis reductions are bitwise identical at 1 and 8 threads") {
	std::vector<float> v(300 * 70 * 9);
	for (size_t i = 0; i < v.size(); i++) v[i] = std::sin(0.37f * static_cast<float>(i));
	auto run = [&](size_t threads) {
		nn::kernels::set_num_threads(threads);
		std::vector<float> out;
		for (int axis : {0, 1, 2}) {
			Value x = Value::leaf(Tensor({300, 70, 9}, v), true);
			Value y = mean(x, axis);
			weighted_loss(y).backward();
			out.insert(out.end(), y.data().data().begin(), y.data().data().end());
			out.insert(out.end(), x.grad().data().begin(), x.grad().data().end());
		}
		return out;
	};
	size_t saved = nn::kernels::num_threads();
	std::vector<float> serial = run(1);
	std::vector<float> parallel = run(8);
	nn::kernels::set_num_threads(saved);
	CHECK(serial == parallel);
}

TEST_CASE("backward orders a node after all its consumers when paths have different depths") {
	// c feeds both mul(c, c) -> mean (a deeper path) and the final mul directly
	auto f = [](Value x) {
		Value c = mul(x, x);
		return sum(mul(c, mean(mul(c, c), 0, true)));
	};
	Value x = Value::leaf(Tensor({7, 3}, std::vector<float>{0.3f, -0.8f, 0.5f, 0.9f, -0.2f, 0.7f, -0.6f, 0.1f, 0.4f, -0.9f, 0.8f,
	                                                       -0.3f, 0.2f, -0.5f, 0.6f, 0.7f, -0.1f, -0.4f, 0.35f, 0.45f, -0.75f}),
	                       true);
	CHECK(gradcheck(f, x, 1e-3f, 5e-2f));
}

TEST_CASE("avgpool2d averages each window, with and without overlap") {
	Value x = Value::leaf(Tensor({1, 1, 4, 4}, std::vector<float>{
	                           1, 2, 5, 6,
	                           3, 4, 7, 8,
	                           9, 10, 13, 14,
	                           11, 12, 15, 16}),
	                       true);
	Value y = avgpool2d(x, 2, 2);
	CHECK(y.data().shape() == std::vector<size_t>{1, 1, 2, 2});
	CHECK(y.data().data() == std::vector<float>{2.5f, 6.5f, 10.5f, 14.5f});
	Value o = avgpool2d(x, 2, 1);
	CHECK(o.data().shape() == std::vector<size_t>{1, 1, 3, 3});
	CHECK(o.data().data() == std::vector<float>{2.5f, 4.5f, 6.5f, 6.5f, 8.5f, 10.5f, 10.5f, 12.5f, 14.5f});
	sum(o).backward();
	// corners belong to one window, centre to four: g/4 accumulated
	CHECK(x.grad().get({0, 0, 0, 0}) == doctest::Approx(0.25f));
	CHECK(x.grad().get({0, 0, 1, 1}) == doctest::Approx(1.0f));
	CHECK(x.grad().get({0, 0, 0, 1}) == doctest::Approx(0.5f));
}

TEST_CASE("avgpool2d gradcheck at several shapes") {
	for (auto shape : {std::vector<size_t>{1, 1, 2, 2}, std::vector<size_t>{2, 3, 7, 7}, std::vector<size_t>{1, 2, 33, 33}}) {
		size_t k = shape[2] == 2 ? 2 : 3;
		size_t s = shape[2] == 7 ? 2 : 1;
		Tensor w = patterned(avgpool2d(Value::leaf(Tensor(shape, 0.0f)), k, s).data().shape(), 0.77f);
		auto f = [&](Value x) { return sum(mul(avgpool2d(x, k, s), Value::leaf(w))); };
		CHECK(gradcheck(f, Value::leaf(patterned(shape, 0.31f), true)));
	}
}

TEST_CASE("avgpool2d gradcheck with 257 batch entries and 257 channels") {
	for (auto shape : {std::vector<size_t>{257, 1, 4, 4}, std::vector<size_t>{1, 257, 4, 4}}) {
		Tensor w = patterned(avgpool2d(Value::leaf(Tensor(shape, 0.0f)), 2, 2).data().shape(), 0.77f);
		auto f = [&](Value x) { return sum(mul(avgpool2d(x, 2, 2), Value::leaf(w))); };
		CHECK(gradcheck(f, Value::leaf(patterned(shape, 0.31f), true)));
	}
}

TEST_CASE("avgpool2d rejects a kernel larger than the input and bad arguments") {
	Value x = Value::leaf(Tensor({1, 1, 3, 3}, 1.0f));
	CHECK_THROWS_AS(avgpool2d(x, 4, 1), TensorShapeError);
	CHECK_THROWS_AS(avgpool2d(x, 0, 1), TensorShapeError);
	CHECK_THROWS_AS(avgpool2d(x, 2, 0), TensorShapeError);
	CHECK_THROWS_AS(avgpool2d(Value::leaf(Tensor({3, 3}, 1.0f)), 2, 1), TensorShapeError);
}

TEST_CASE("avgpool2d forward and backward are bitwise identical for any thread count") {
	size_t saved = nn::kernels::num_threads();
	for (size_t k : {size_t{1}, size_t{2}, size_t{3}}) {
		for (size_t stride : {size_t{1}, size_t{2}}) {
			Tensor xv = patterned({7, 3, 12, 9}, 0.31f);
			Storage out1, g1;
			for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
				nn::kernels::set_num_threads(threads);
				Value x = Value::leaf(xv, true);
				Value y = avgpool2d(x, k, stride);
				sum(y).backward();
				if (threads == 1) {
					out1 = y.data().data();
					g1 = x.grad().data();
				} else {
					CHECK(y.data().data() == out1);
					CHECK(x.grad().data() == g1);
				}
			}
		}
	}
	nn::kernels::set_num_threads(saved);
}

namespace {
// Direct grouped, dilated convolution (rectangular kernels allowed), the reference for the general path
Tensor reference_conv_general(const Tensor& x, const Tensor& w, const Tensor& b, size_t stride, size_t pad, size_t dil,
	size_t groups) {
	size_t N = x.shape()[0], Cin = x.shape()[1], H = x.shape()[2], W = x.shape()[3];
	size_t Cout = w.shape()[0], Cg = w.shape()[1], kh = w.shape()[2], kw = w.shape()[3];
	size_t Ho = (H + 2 * pad - dil * (kh - 1) - 1) / stride + 1, Wo = (W + 2 * pad - dil * (kw - 1) - 1) / stride + 1;
	size_t Cog = Cout / groups;
	(void)Cin;
	Tensor out({N, Cout, Ho, Wo});
	for (size_t n = 0; n < N; n++)
		for (size_t co = 0; co < Cout; co++)
			for (size_t oh = 0; oh < Ho; oh++)
				for (size_t ow = 0; ow < Wo; ow++) {
					float acc = b.data()[co];
					size_t g = co / Cog;
					for (size_t c = 0; c < Cg; c++)
						for (size_t i = 0; i < kh; i++)
							for (size_t j = 0; j < kw; j++) {
								long ih = static_cast<long>(oh * stride + i * dil) - static_cast<long>(pad);
								long iw = static_cast<long>(ow * stride + j * dil) - static_cast<long>(pad);
								if (ih < 0 || iw < 0 || ih >= static_cast<long>(H) || iw >= static_cast<long>(W)) continue;
								acc += x.get({n, g * Cg + c, static_cast<size_t>(ih), static_cast<size_t>(iw)}) *
								       w.get({co, c, i, j});
							}
					out.at({n, co, oh, ow}) = acc;
				}
	return out;
}
}  // namespace

TEST_CASE("conv2d with dilation 1 and groups 1 is bitwise identical to the 5-argument overload") {
	Tensor xv = patterned({3, 2, 9, 7}, 0.07f), wv = patterned({4, 2, 3, 3}, 0.05f), bv = patterned({4}, 0.1f);
	Value y5 = conv2d(Value::leaf(xv, false), Value::leaf(wv, false), Value::leaf(bv, false), 2, 1);
	Value y7 = conv2d(Value::leaf(xv, false), Value::leaf(wv, false), Value::leaf(bv, false), 2, 1, 1, 1);
	REQUIRE(y5.data().shape() == y7.data().shape());
	for (size_t i = 0; i < y5.data().size(); i++) CHECK(y5.data().data()[i] == y7.data().data()[i]);
}

TEST_CASE("conv2d general path matches a direct reference and is independent of the thread count") {
	struct Cfg { size_t N, Cin, H, W, Cout, kh, kw, stride, pad, dil, groups; };
	std::vector<Cfg> cfgs = {
		{2, 4, 9, 9, 4, 3, 3, 1, 2, 2, 1},   // dilation 2
		{2, 4, 9, 9, 6, 3, 3, 1, 1, 1, 2},   // groups 2
		{2, 4, 9, 9, 4, 3, 3, 2, 1, 1, 4},   // depthwise
		{2, 4, 9, 9, 8, 3, 3, 2, 2, 2, 2},   // everything at once
		{3, 2, 6, 8, 2, 1, 3, 1, 1, 2, 2},   // rectangular, depthwise
	};
	size_t saved = nn::kernels::num_threads();
	for (const Cfg& c : cfgs) {
		Tensor xv = patterned({c.N, c.Cin, c.H, c.W}, 0.07f);
		Tensor wv = patterned({c.Cout, c.Cin / c.groups, c.kh, c.kw}, 0.05f);
		Tensor bv = patterned({c.Cout}, 0.1f);
		Tensor expect = reference_conv_general(xv, wv, bv, c.stride, c.pad, c.dil, c.groups);
		Storage gx1, gw1, gb1;
		for (size_t threads : {size_t{1}, size_t{3}, size_t{8}}) {
			nn::kernels::set_num_threads(threads);
			Value x = Value::leaf(xv, true), w = Value::leaf(wv, true), b = Value::leaf(bv, true);
			Value y = conv2d(x, w, b, static_cast<int>(c.stride), static_cast<int>(c.pad), static_cast<int>(c.dil),
				static_cast<int>(c.groups));
			REQUIRE(y.data().shape() == expect.shape());
			for (size_t i = 0; i < expect.size(); i++) CHECK(y.data().data()[i] == doctest::Approx(expect.data()[i]).epsilon(1e-4));
			sum(y).backward();
			if (threads == 1) {
				gx1 = x.grad().data(); gw1 = w.grad().data(); gb1 = b.grad().data();
			} else {
				CHECK(x.grad().data() == gx1);
				CHECK(w.grad().data() == gw1);
				CHECK(b.grad().data() == gb1);
			}
		}
	}
	nn::kernels::set_num_threads(saved);
}

TEST_CASE("conv2d general path gradchecks input, weight and bias") {
	struct Cfg { std::vector<size_t> x, w; int stride, pad, dil, groups; };
	std::vector<Cfg> cfgs = {
		{{1, 2, 3, 3}, {2, 2, 3, 3}, 1, 2, 2, 1},  // dilation 2, padding 2
		{{2, 4, 7, 7}, {4, 2, 3, 3}, 1, 1, 1, 2},  // groups 2
		{{1, 3, 5, 5}, {3, 1, 3, 3}, 1, 1, 1, 3},  // depthwise
		{{1, 2, 5, 6}, {2, 2, 1, 3}, 1, 0, 1, 1},  // rectangular 1x3 (groups 1, dilation 1: delegated)
		{{1, 2, 5, 6}, {2, 1, 1, 3}, 1, 1, 2, 2},  // rectangular, dilated, depthwise
	};
	for (const Cfg& c : cfgs) {
		Tensor xv = Tensor::random(c.x, -1.0f, 1.0f), wv = Tensor::random(c.w, -0.5f, 0.5f);
		Tensor bv = Tensor::random({c.w[0]}, -0.2f, 0.2f);
		Value x = Value::leaf(xv, true), w = Value::leaf(wv, true), b = Value::leaf(bv, true);
		auto op = [&](Value xi, Value wi, Value bi) { return conv2d(xi, wi, bi, c.stride, c.pad, c.dil, c.groups); };
		CHECK(gradcheck([&](Value v) { return sum(op(v, w, b)); }, x, 1e-3f, 0.1f));
		CHECK(gradcheck([&](Value v) { return sum(op(x, v, b)); }, w, 1e-3f, 0.1f));
		CHECK(gradcheck([&](Value v) { return sum(op(x, w, v)); }, b, 1e-3f, 0.1f));
	}
}

TEST_CASE("conv2d general path rejects bad arguments") {
	Value x = Value::leaf(Tensor({1, 4, 5, 5}, 1.0f), false);
	Value w = Value::leaf(Tensor({4, 2, 3, 3}, 1.0f), false);
	Value b = Value::leaf(Tensor({4}, 0.0f), false);
	CHECK_NOTHROW(conv2d(x, w, b, 1, 0, 1, 2));
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0, 1, 3), TensorShapeError);  // groups does not divide channels
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0, 1, 4), TensorShapeError);  // weight Cin/groups mismatch
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0, 0, 2), TensorShapeError);  // dilation 0
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0, 3, 2), TensorShapeError);  // effective kernel 7 > 5
	CHECK_THROWS_AS(conv2d(x, w, b, 1, 0, 1, 0), TensorShapeError);  // groups 0
}

TEST_CASE("weighted cross_entropy_loss with unit weights and no smoothing equals the unweighted loss") {
	Tensor lv = Tensor::random({7, 4}, -2.0f, 2.0f);
	Tensor targets({7}, std::vector<float>{0, 1, 2, 3, 3, 1, 0});
	Value a = Value::leaf(lv, true), b = Value::leaf(lv, true);
	Value la = cross_entropy_loss(a, targets), lb = cross_entropy_loss(b, targets, Tensor({4}, 1.0f));
	CHECK(lb.data().data()[0] == doctest::Approx(la.data().data()[0]).epsilon(1e-6));
	la.backward(); lb.backward();
	for (size_t i = 0; i < lv.size(); i++) CHECK(b.grad().data()[i] == doctest::Approx(a.grad().data()[i]).epsilon(1e-5));
}

TEST_CASE("weighted cross_entropy_loss matches a hand-computed value") {
	// Uniform logits: every -log p is log 3, so the weighted mean is log 3 whatever the weights
	Value logits = Value::leaf(Tensor({3, 3}, 0.0f), false);
	Tensor targets({3}, std::vector<float>{0, 1, 2});
	Tensor w({3}, std::vector<float>{1.0f, 2.0f, 0.5f});
	CHECK(cross_entropy_loss(logits, targets, w).data().data()[0] == doctest::Approx(std::log(3.0f)));
	// Non-uniform: sample losses l = {log(1+e^-1)+..., } computed directly
	Value lg = Value::leaf(Tensor({3, 3}, std::vector<float>{1, 0, 0, 0, 2, 0, 0, 0, 3}), false);
	float l0 = std::log(std::exp(1.0f) + 2.0f) - 1.0f;
	float l1 = std::log(std::exp(2.0f) + 2.0f) - 2.0f;
	float l2 = std::log(std::exp(3.0f) + 2.0f) - 3.0f;
	float expect = (1.0f * l0 + 2.0f * l1 + 0.5f * l2) / 3.5f;
	CHECK(cross_entropy_loss(lg, targets, w).data().data()[0] == doctest::Approx(expect).epsilon(1e-5));
}

TEST_CASE("cross_entropy_loss label smoothing on a two-class example") {
	Value logits = Value::leaf(Tensor({1, 2}, std::vector<float>{2.0f, 0.0f}), false);
	Tensor targets({1}, std::vector<float>{0});
	float lse = std::log(std::exp(2.0f) + 1.0f);
	float nll0 = lse - 2.0f, nll1 = lse;
	float s = 0.2f;
	float expect = (1.0f - s) * nll0 + s * 0.5f * (nll0 + nll1);
	CHECK(cross_entropy_loss(logits, targets, Tensor({2}, 1.0f), s).data().data()[0] == doctest::Approx(expect).epsilon(1e-5));
}

TEST_CASE("weighted smoothed cross_entropy_loss gradcheck") {
	for (size_t N : {size_t{1}, size_t{7}, size_t{33}}) {
		Tensor targets({N});
		for (size_t i = 0; i < N; i++) targets.data()[i] = static_cast<float>((i * 5) % 4);
		Tensor w({4}, std::vector<float>{1.0f, 2.0f, 0.5f, 1.5f});
		auto f = [targets, w](Value l) { return cross_entropy_loss(l, targets, w, 0.1f); };
		Value x = Value::leaf(Tensor::random({N, 4}, -2.0f, 2.0f), true);
		CHECK(gradcheck(f, x, 1e-3f, 0.05f));
	}
}

TEST_CASE("weighted cross_entropy_loss rejects bad input") {
	Value logits = Value::leaf(Tensor({2, 3}, 0.0f), false);
	Tensor targets({2}, std::vector<float>{0, 1});
	Tensor w({3}, 1.0f);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, Tensor({2}, 1.0f)), TensorShapeError);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, w, 1.0f), std::invalid_argument);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, w, -0.1f), std::invalid_argument);
	CHECK_THROWS_AS(cross_entropy_loss(logits, Tensor({2}, std::vector<float>{0, 3}), w), std::invalid_argument);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, Tensor({3}, std::vector<float>{1, -1, 1})), std::invalid_argument);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, Tensor({3}, std::vector<float>{1, std::nanf(""), 1})), std::invalid_argument);
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, Tensor({3}, std::vector<float>{1, INFINITY, 1})), std::invalid_argument);
	// Both samples belong to zero-weight classes
	CHECK_THROWS_AS(cross_entropy_loss(logits, targets, Tensor({3}, std::vector<float>{0, 0, 1})), std::invalid_argument);
}

TEST_CASE("sequence_cross_entropy equals cross_entropy_loss when nothing is ignored") {
	Tensor lv = Tensor::random({2, 3, 4}, -2.0f, 2.0f);
	Tensor tg({2, 3}, std::vector<float>{0, 1, 2, 3, 3, 1});
	Tensor flat_lv = lv.reshape({6, 4});
	Tensor flat_tg = tg.reshape({6});
	Value a = Value::leaf(lv, true), b = Value::leaf(flat_lv, true);
	Value la = sequence_cross_entropy(a, tg), lb = cross_entropy_loss(b, flat_tg);
	CHECK(la.data().data()[0] == doctest::Approx(lb.data().data()[0]).epsilon(1e-6));
	// smoothing 0.1 against the weighted overload with unit weights
	Value c = Value::leaf(lv, true), d = Value::leaf(flat_lv, true);
	Value ls = sequence_cross_entropy(c, tg, -100, 0.1f), ld = cross_entropy_loss(d, flat_tg, Tensor({4}, 1.0f), 0.1f);
	CHECK(ls.data().data()[0] == doctest::Approx(ld.data().data()[0]).epsilon(1e-6));
	la.backward(); lb.backward(); ls.backward(); ld.backward();
	for (size_t i = 0; i < lv.size(); i++) {
		CHECK(a.grad().data()[i] == doctest::Approx(b.grad().data()[i]).epsilon(1e-5));
		CHECK(c.grad().data()[i] == doctest::Approx(d.grad().data()[i]).epsilon(1e-5));
	}
}

TEST_CASE("sequence_cross_entropy skips ignored positions") {
	Tensor lv = Tensor::random({2, 3, 4}, -2.0f, 2.0f);
	Tensor tg({2, 3}, std::vector<float>{0, -100, 2, 3, -100, 1});
	Value x = Value::leaf(lv, true);
	Value loss = sequence_cross_entropy(x, tg);
	// reference: mean over the four kept positions via cross_entropy_loss
	Tensor kl({4, 4}), kt({4});
	size_t keep[4] = {0, 2, 3, 5};
	for (size_t i = 0; i < 4; i++) {
		for (size_t c = 0; c < 4; c++) kl.data()[i * 4 + c] = lv.data()[keep[i] * 4 + c];
		kt.data()[i] = tg.data()[keep[i]];
	}
	Value ref = cross_entropy_loss(Value::leaf(kl, false), kt);
	CHECK(loss.data().data()[0] == doctest::Approx(ref.data().data()[0]).epsilon(1e-6));
	loss.backward();
	for (size_t p : {size_t{1}, size_t{4}}) {
		for (size_t c = 0; c < 4; c++) CHECK(x.grad().data()[p * 4 + c] == 0.0f);
	}
	// a custom ignore id works too
	Tensor tg2 = tg;
	for (float& v : tg2.data()) if (v == -100.0f) v = 7.0f;
	CHECK(sequence_cross_entropy(Value::leaf(lv, false), tg2, 7).data().data()[0]
		== doctest::Approx(ref.data().data()[0]).epsilon(1e-6));
}

TEST_CASE("sequence_cross_entropy gradcheck") {
	for (auto shape : {std::vector<size_t>{1, 1, 3}, std::vector<size_t>{2, 5, 7}}) {
		size_t P = shape[0] * shape[1], V = shape[2];
		Tensor tg({shape[0], shape[1]});
		for (size_t i = 0; i < P; i++) tg.data()[i] = (i % 4 == 3 && P > 1) ? -100.0f : static_cast<float>((i * 5) % V);
		float count = 0.0f;
		for (size_t i = 0; i < P; i++) count += tg.data()[i] != -100.0f ? 1.0f : 0.0f;
		// scaled by the count of scored positions so the gradients are O(1) and the tolerance bites
		auto f = [tg, count](Value l) { return mul(sequence_cross_entropy(l, tg, -100, 0.1f), Value::leaf(Tensor({1}, count), false)); };
		Value x = Value::leaf(Tensor::random(shape, -2.0f, 2.0f), true);
		CHECK(gradcheck(f, x, 1e-3f, 1e-2f));
	}
}

TEST_CASE("sequence_cross_entropy rejects bad input") {
	Value logits = Value::leaf(Tensor({1, 2, 3}, 0.0f), false);
	CHECK_THROWS_AS(sequence_cross_entropy(logits, Tensor({2}, 0.0f)), TensorShapeError);
	CHECK_THROWS_AS(sequence_cross_entropy(Value::leaf(Tensor({2, 3}, 0.0f), false), Tensor({2}, 0.0f)), TensorShapeError);
	CHECK_THROWS_AS(sequence_cross_entropy(logits, Tensor({1, 2}, std::vector<float>{0, 3})), std::invalid_argument);
	CHECK_THROWS_AS(sequence_cross_entropy(logits, Tensor({1, 2}, std::vector<float>{0, 1}), -100, 1.0f), std::invalid_argument);
	CHECK_THROWS_AS(sequence_cross_entropy(logits, Tensor({1, 2}, std::vector<float>{-100, -100})), std::invalid_argument);
}

TEST_CASE("axis reductions over a zero-length axis throw") {
	Value x = Value::leaf(Tensor({0, 3}, 0.0f), false);
	CHECK_THROWS_AS(mean(x, 0), TensorShapeError);
	CHECK_THROWS_AS(sum(x, 0), TensorShapeError);
}

TEST_CASE("exp log sqrt div gelu forward values") {
	auto at = [](const Value& v) { return v.data().data()[0]; };
	CHECK(at(exp(Value::leaf(Tensor({1}, 0.0f), false))) == doctest::Approx(1.0f));
	CHECK(at(log(Value::leaf(Tensor({1}, 2.7182818f), false))) == doctest::Approx(1.0f));
	CHECK(at(sqrt(Value::leaf(Tensor({1}, 4.0f), false))) == doctest::Approx(2.0f));
	Value q = div(Value::leaf(Tensor({2}, std::vector<float>{2, 3}), false), Value::leaf(Tensor({1}, 2.0f), false));
	CHECK(q.data().data()[0] == doctest::Approx(1.0f));
	CHECK(q.data().data()[1] == doctest::Approx(1.5f));
	CHECK(at(gelu(Value::leaf(Tensor({1}, 0.0f), false))) == doctest::Approx(0.0f));
	CHECK(at(gelu(Value::leaf(Tensor({1}, 1.0f), false))) == doctest::Approx(0.841192f).epsilon(1e-4));
	CHECK(at(gelu(Value::leaf(Tensor({1}, -1.0f), false))) == doctest::Approx(-0.158808f).epsilon(1e-4));
}

TEST_CASE("exp log sqrt gelu gradcheck") {
	for (auto shape : {std::vector<size_t>{1}, std::vector<size_t>{7}, std::vector<size_t>{3, 5}}) {
		Value x = Value::leaf(Tensor::random(shape, 0.5f, 2.0f), true);
		CHECK(gradcheck([](Value v) { return sum(exp(v)); }, x));
		CHECK(gradcheck([](Value v) { return sum(log(v)); }, x));
		CHECK(gradcheck([](Value v) { return sum(sqrt(v)); }, x));
		CHECK(gradcheck([](Value v) { return sum(gelu(v)); }, x));
	}
}

TEST_CASE("div gradcheck both operands including broadcast") {
	for (auto shape : {std::vector<size_t>{1}, std::vector<size_t>{7}, std::vector<size_t>{3, 5}}) {
		Value a = Value::leaf(Tensor::random(shape, 0.5f, 2.0f), true);
		Value b = Value::leaf(Tensor::random(shape, 0.5f, 2.0f), true);
		CHECK(gradcheck([b](Value v) { return sum(div(v, b)); }, a));
		CHECK(gradcheck([a](Value v) { return sum(div(a, v)); }, b));
	}
	Value a = Value::leaf(Tensor::random({3, 5}, 0.5f, 2.0f), true);
	Value b = Value::leaf(Tensor::random({5}, 0.5f, 2.0f), true);
	CHECK(gradcheck([b](Value v) { return sum(div(v, b)); }, a));
	CHECK(gradcheck([a](Value v) { return sum(div(a, v)); }, b));
}

TEST_CASE("masked_fill forward and gradient") {
	Value x = Value::leaf(Tensor({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}), true);
	Tensor mask({1, 3}, std::vector<float>{0, 1, 0});
	Value y = masked_fill(x, mask, -1e9f);
	const Storage& d = y.data().data();
	CHECK(d[0] == 1.0f);
	CHECK(d[1] == -1e9f);
	CHECK(d[2] == 3.0f);
	CHECK(d[4] == -1e9f);
	sum(y).backward();
	const Storage& g = x.grad().data();
	CHECK(g == std::vector<float>{1, 0, 1, 1, 0, 1});
	CHECK_THROWS_AS(masked_fill(x, Tensor({4}, 1.0f), 0.0f), TensorShapeError);
	CHECK_THROWS_AS(masked_fill(x, Tensor({3, 2, 3}, 1.0f), 0.0f), TensorShapeError);
}

TEST_CASE("slice concat split forward and errors") {
	Value x = Value::leaf(Tensor::random({2, 6}, -1.0f, 1.0f), false);
	Value s = slice(x, 1, 2, 5);
	CHECK(s.data().shape() == std::vector<size_t>{2, 3});
	CHECK(s.data().data()[0] == x.data().data()[2]);
	CHECK(s.data().data()[3] == x.data().data()[8]);
	Value r = slice(x, -2, 1, 2);
	CHECK(r.data().shape() == std::vector<size_t>{1, 6});
	CHECK(r.data().data()[0] == x.data().data()[6]);
	CHECK_THROWS_AS(slice(x, 1, 3, 3), TensorShapeError);
	CHECK_THROWS_AS(slice(x, 1, 0, 7), TensorShapeError);
	CHECK_THROWS_AS(slice(x, 2, 0, 1), TensorIndexError);

	Value a = Value::leaf(Tensor::random({2, 3}, -1.0f, 1.0f), false);
	Value b = Value::leaf(Tensor::random({2, 4}, -1.0f, 1.0f), false);
	Value c = concat({a, b}, 1);
	CHECK(c.data().shape() == std::vector<size_t>{2, 7});
	Value parts = concat({a, b}, 1);
	std::vector<Value> back = {slice(parts, 1, 0, 3), slice(parts, 1, 3, 7)};
	CHECK(back[0].data().data() == a.data().data());
	CHECK(back[1].data().data() == b.data().data());
	std::vector<Value> halves = split(c, 0, 2);
	CHECK(halves.size() == 2);
	CHECK(halves[1].data().data()[0] == c.data().data()[7]);
	CHECK_THROWS_AS(split(c, 1, 2), TensorShapeError);
	CHECK_THROWS_AS(split(c, 1, 0), TensorShapeError);
	CHECK_THROWS_AS(concat({}, 0), TensorShapeError);
	CHECK_THROWS_AS(concat({a, b}, 0), TensorShapeError);
}

TEST_CASE("masked_fill slice concat split gradcheck") {
	for (size_t n : {size_t(1), size_t(7), size_t(33)}) {
		for (int ax : {0, 1, -1}) {
			std::vector<size_t> shape = ax == 0 ? std::vector<size_t>{n, 3} : std::vector<size_t>{3, n};
			size_t ai = ax == -1 ? 1 : static_cast<size_t>(ax);
			size_t a = shape[ai];
			Value x = Value::leaf(Tensor::random(shape, -1.0f, 1.0f), true);
			Value w1 = Value::leaf(Tensor::random(shape, 0.5f, 2.0f), false);

			// Alternating 0/1 mask along the cut axis; broadcast over the other axis.
			std::vector<float> flags(a);
			for (size_t i = 0; i < a; i++) flags[i] = static_cast<float>(i % 2);
			Tensor mask(ax == 0 ? std::vector<size_t>{a, 1} : std::vector<size_t>{a}, flags);
			CHECK(gradcheck([&](Value v) { return sum(mul(masked_fill(v, mask, 3.0f), w1)); }, x));

			size_t lo = a > 1 ? 1 : 0;
			std::vector<size_t> ws = shape;
			ws[ai] = a - lo;
			Value w2 = Value::leaf(Tensor::random(ws, 0.5f, 2.0f), false);
			CHECK(gradcheck([&](Value v) { return sum(mul(slice(v, ax, lo, a), w2)); }, x));

			Value w3 = concat({w1, w1}, ax);
			CHECK(gradcheck([&](Value v) { return sum(mul(concat({v, v}, ax), w3)); }, x));
		}
	}
	// masked_fill with a real mask, weighted loss
	Value x = Value::leaf(Tensor::random({4, 7}, -1.0f, 1.0f), true);
	Tensor mask({1, 7}, std::vector<float>{0, 1, 0, 0, 1, 0, 1});
	Tensor w = Tensor::random({4, 7}, 0.5f, 2.0f);
	CHECK(gradcheck([&](Value v) { return sum(mul(masked_fill(v, mask, 3.0f), Value::leaf(w, false))); }, x));
	// split: three parts of 33 along the cut axis, weighted differently per part
	Value y = Value::leaf(Tensor::random({2, 33}, -1.0f, 1.0f), true);
	Tensor wp = Tensor::random({2, 11}, 0.5f, 2.0f);
	CHECK(gradcheck([&](Value v) {
		std::vector<Value> p = split(v, 1, 3);
		return sum(add(mul(p[0], Value::leaf(wp, false)), scale(mul(p[2], Value::leaf(wp, false)), 2.0f)));
	}, y));
}

TEST_CASE("split inverts concat bit for bit and oversized mask dim is rejected") {
	Value a = Value::leaf(Tensor::random({2, 5}, -1.0f, 1.0f), false);
	std::vector<Value> halves = split(concat({a, a}, 1), 1, 2);
	CHECK(halves[0].data().data() == a.data().data());
	CHECK(halves[1].data().data() == a.data().data());
	Value x = Value::leaf(Tensor({1, 3}, 1.0f), false);
	CHECK_THROWS_AS(masked_fill(x, Tensor({2, 3}, 1.0f), 0.0f), TensorShapeError);
}

TEST_CASE("fused batchnorm and layernorm match the composed ops and ignore the thread count") {
	struct Restore {
		size_t saved = kernels::num_threads();
		~Restore() { kernels::set_num_threads(saved); }
	} restore;
	manual_seed(3);
	const float eps = 1e-5f;
	auto max_diff = [](const Storage& a, const Storage& b) {
		float d = 0.0f;
		for (size_t i = 0; i < a.size(); i++) d = std::max(d, std::fabs(a[i] - b[i]));
		return d;
	};

	// Batch norm over {N, C, H, W}: the composed formula is what BatchNorm used before it was fused
	const size_t N = 6, C = 5, H = 7, W = 9;
	Tensor xt = Tensor::random({N, C, H, W}, -2.0f, 3.0f), wt = Tensor::random({C}, 0.5f, 1.5f), bt = Tensor::random({C}, -1.0f, 1.0f);
	Tensor probe = Tensor::random({N, C, H, W}, 0.5f, 2.0f);
	auto composed_bn = [&](Value x, Value w, Value b) {
		Value mu = x;
		for (int axis : {3, 2, 0}) mu = mean(mu, axis, true);
		Value c = sub(x, mu);
		Value var = mul(c, c);
		for (int axis : {3, 2, 0}) var = mean(var, axis, true);
		Value inv = rsqrt(add(var, Value::leaf(Tensor({1, C, 1, 1}, eps))));
		return add(mul(mul(c, inv), reshape(w, {1, C, 1, 1})), reshape(b, {1, C, 1, 1}));
	};
	auto run_bn = [&](bool fused, size_t threads) {
		kernels::set_num_threads(threads);
		Value x = Value::leaf(xt, true), w = Value::leaf(wt, true), b = Value::leaf(bt, true);
		Value y = fused ? batchnorm(x, w, b, eps) : composed_bn(x, w, b);
		sum(mul(y, Value::leaf(probe, false))).backward();
		return std::vector<Storage>{y.data().data(), x.grad().data(), w.grad().data(), b.grad().data()};
	};
	auto ref = run_bn(false, 1), one = run_bn(true, 1), four = run_bn(true, 4);
	for (size_t i = 0; i < ref.size(); i++) CHECK(max_diff(ref[i], one[i]) < 2e-3f);
	CHECK(one == four);

	// Layer norm over the last axis of {40, 30, 16}
	const size_t D = 16;
	Tensor lx = Tensor::random({40, 30, D}, -2.0f, 3.0f), gt = Tensor::random({D}, 0.5f, 1.5f), betat = Tensor::random({D}, -1.0f, 1.0f);
	Tensor lprobe = Tensor::random({40, 30, D}, 0.5f, 2.0f);
	auto composed_ln = [&](Value x, Value g, Value b) {
		Value c = sub(x, mean(x, 2, true));
		Value var = mean(mul(c, c), 2, true);
		return add(mul(mul(c, rsqrt(add(var, Value::leaf(Tensor({1, 1, 1}, eps))))), g), b);
	};
	auto run_ln = [&](bool fused, size_t threads) {
		kernels::set_num_threads(threads);
		Value x = Value::leaf(lx, true), g = Value::leaf(gt, true), b = Value::leaf(betat, true);
		Value y = fused ? layernorm(x, g, b, eps) : composed_ln(x, g, b);
		sum(mul(y, Value::leaf(lprobe, false))).backward();
		return std::vector<Storage>{y.data().data(), x.grad().data(), g.grad().data(), b.grad().data()};
	};
	auto lref = run_ln(false, 1), lone = run_ln(true, 1), lfour = run_ln(true, 4);
	for (size_t i = 0; i < lref.size(); i++) CHECK(max_diff(lref[i], lone[i]) < 2e-3f);
	CHECK(lone == lfour);
}

TEST_CASE("scaled_masked_softmax matches scale, masked_fill and softmax and ignores the thread count") {
	struct Restore {
		size_t saved = kernels::num_threads();
		~Restore() { kernels::set_num_threads(saved); }
	} restore;
	manual_seed(4);
	const size_t N = 3, H = 4, T = 40;
	const float sc = 0.37f;
	Tensor st = Tensor::random({N, H, T, T}, -3.0f, 3.0f), probe = Tensor::random({N, H, T, T}, 0.5f, 2.0f);
	Tensor pad = Tensor::ones({N, T});
	for (size_t j = 30; j < T; j++) pad.data()[1 * T + j] = 0.0f;  // sample 1 has 10 padded keys
	for (size_t j = 0; j < T; j++) pad.data()[2 * T + j] = 0.0f;   // sample 2 is padded everywhere
	Tensor pmask = Tensor::zeros({N, 1, 1, T}), cmask = Tensor::zeros({1, 1, T, T});
	for (size_t e = 0; e < N * T; e++) pmask.data()[e] = pad.data()[e] == 0.0f ? 1.0f : 0.0f;
	for (size_t i = 0; i < T; i++)
		for (size_t j = i + 1; j < T; j++) cmask.data()[i * T + j] = 1.0f;

	for (bool causal : {false, true}) {
		auto run = [&](bool fused, size_t threads) {
			kernels::set_num_threads(threads);
			Value x = Value::leaf(st, true);
			Value y = fused ? scaled_masked_softmax(x, sc, pad, causal) : [&] {
				Value s = masked_fill(scale(x, sc), pmask, -1e9f);
				if (causal) s = masked_fill(s, cmask, -2e9f);
				return softmax(s, 3);
			}();
			sum(mul(y, Value::leaf(probe, false))).backward();
			return std::vector<Storage>{y.data().data(), x.grad().data()};
		};
		auto ref = run(false, 1), one = run(true, 1), four = run(true, 4);
		CHECK(ref[0] == one[0]);  // same operations in the same order: forward is bitwise equal
		float d = 0.0f;
		for (size_t i = 0; i < ref[1].size(); i++) d = std::max(d, std::fabs(ref[1][i] - one[1][i]));
		CHECK(d < 1e-5f);
		CHECK(one == four);
	}
}

TEST_CASE("matmul and matmul_bt with a batched left operand and a plain matrix: values and gradients") {
	manual_seed(6);
	const size_t N = 3, T = 4, K = 5, E = 6;
	Tensor xt = Tensor::random({N, T, K}, -1.0f, 1.0f), wt = Tensor::random({K, E}, -1.0f, 1.0f), wbt = Tensor::random({E, K}, -1.0f, 1.0f);

	// The folded forward equals one product per sample
	Tensor folded = matmul(xt, wt), folded_bt = matmul(xt, wbt, false, true);
	CHECK(folded.shape() == std::vector<size_t>{N, T, E});
	for (size_t b = 0; b < N; b++) {
		Tensor xs({T, K});
		for (size_t i = 0; i < T * K; i++) xs.data()[i] = xt.data()[b * T * K + i];
		Tensor ref = matmul(xs, wt), ref_bt = matmul(xs, wbt, false, true);
		for (size_t i = 0; i < T * E; i++) {
			CHECK(std::fabs(folded.data()[b * T * E + i] - ref.data()[i]) < 1e-5f);
			CHECK(std::fabs(folded_bt.data()[b * T * E + i] - ref_bt.data()[i]) < 1e-5f);
		}
	}

	// Gradients that accumulate straight into grad() match finite differences, and add to what is there
	Tensor probe = Tensor::random({N, T, E}, 0.5f, 1.5f);
	Value x = Value::leaf(xt, true), w = Value::leaf(wt, true), wb = Value::leaf(wbt, true);
	auto loss_mm = [&](Value in, Value wv) { return sum(mul(matmul(in, wv), Value::leaf(probe, false))); };
	auto loss_bt = [&](Value in, Value wv) { return sum(mul(matmul_bt(in, wv), Value::leaf(probe, false))); };
	CHECK(gradcheck([&](Value v) { return loss_mm(v, w); }, x));
	CHECK(gradcheck([&](Value v) { return loss_mm(x, v); }, w));
	CHECK(gradcheck([&](Value v) { return loss_bt(v, wb); }, x));
	CHECK(gradcheck([&](Value v) { return loss_bt(x, v); }, wb));

	w.zero_grad();
	loss_mm(x, w).backward();
	Storage once = w.grad().data();
	loss_mm(x, w).backward();  // a second backward must add to the first, not overwrite it
	for (size_t i = 0; i < once.size(); i++) CHECK(std::fabs(w.grad().data()[i] - 2.0f * once[i]) < 1e-4f);
}

namespace {

/**
 * Runs f under each thread count and checks the forward data and the leaf gradients are bitwise equal
 *
 * Parameters:
 * function<Value()> f: Builds a scalar loss from the leaves
 * vector<Value> leaves: Leaves whose gradients are compared
 *
 * Returns bool: true when every thread count gives the bits of the single-threaded run
 */
bool same_bits_for_any_thread_count(const std::function<Value()>& f, const std::vector<Value>& leaves) {
	const size_t saved = kernels::num_threads();
	std::vector<std::vector<float>> want;
	bool ok = true;
	for (size_t t : {1, 2, 3, 8}) {
		kernels::set_num_threads(t);
		for (const Value& l : leaves) l.zero_grad();
		Value loss = f();
		loss.backward();
		std::vector<std::vector<float>> got;
		got.emplace_back(loss.data().data().begin(), loss.data().data().end());
		for (const Value& l : leaves) got.emplace_back(l.grad().data().begin(), l.grad().data().end());
		if (t == 1) {
			want = got;
			continue;
		}
		for (size_t i = 0; i < got.size(); i++)
			ok = ok && got[i].size() == want[i].size() &&
				std::memcmp(got[i].data(), want[i].data(), got[i].size() * sizeof(float)) == 0;
	}
	kernels::set_num_threads(saved);
	return ok;
}

/**
 * The attention core built from the basic ops, the reference for the fused attention op
 *
 * Parameters:
 * Value q, k, v: {N, T, D} operands
 * size_t heads: Number of heads
 * Tensor mask: Key padding mask, or an empty tensor
 * bool causal: Hide later keys
 *
 * Returns Value: {N, Tq, D}
 */
Value composed_attention(const Value& q, const Value& k, const Value& v, size_t heads, const Tensor& mask, bool causal) {
	const size_t N = q.data().shape()[0], Tq = q.data().shape()[1], Tk = k.data().shape()[1], D = q.data().shape()[2];
	const size_t Dh = D / heads;
	auto split = [&](const Value& x, size_t T) { return transpose(reshape(x, {N, T, heads, Dh}), {0, 2, 1, 3}); };
	Value w = scaled_masked_softmax(matmul_bt(split(q, Tq), split(k, Tk)), 1.0f / std::sqrt(static_cast<float>(Dh)), mask, causal);
	return reshape(transpose(matmul(w, split(v, Tk)), {0, 2, 1, 3}), {N, Tq, D});
}

}  // namespace

TEST_CASE("linear gradcheck for input, weight and bias") {
	Value x = Value::leaf(sine_tensor({2, 5, 3}, 0.3f), true);
	Value w = Value::leaf(sine_tensor({4, 3}, 0.7f), true);
	Value b = Value::leaf(sine_tensor({4}, 1.1f), true);
	CHECK(gradcheck([w, b](Value v) mutable { return weighted_sum(linear(v, w, b)); }, x));
	CHECK(gradcheck([x, b](Value v) mutable { return weighted_sum(linear(x, v, b)); }, w));
	CHECK(gradcheck([x, w](Value v) mutable { return weighted_sum(linear(x, w, v)); }, b));
}

TEST_CASE("linear gives the bits of matmul_bt followed by add, forward and backward") {
	for (std::vector<size_t> xs : {std::vector<size_t>{7, 5}, std::vector<size_t>{3, 6, 5}}) {
		Value x = Value::leaf(sine_tensor(xs, 0.3f), true);
		Value w = Value::leaf(sine_tensor({9, 5}, 0.7f), true);
		Value b = Value::leaf(sine_tensor({9}, 1.1f), true);
		Value fused = linear(x, w, b);
		weighted_sum(fused).backward();
		std::vector<std::vector<float>> got;
		for (const Value& l : {x, w, b}) got.emplace_back(l.grad().data().begin(), l.grad().data().end());
		for (const Value& l : {x, w, b}) l.zero_grad();
		Value composed = add(matmul_bt(x, w), b);
		weighted_sum(composed).backward();
		CHECK(fused.data().shape() == composed.data().shape());
		CHECK(std::memcmp(fused.data().data().data(), composed.data().data().data(), fused.data().size() * sizeof(float)) == 0);
		size_t i = 0;
		for (const Value& l : {x, w, b}) {
			CHECK(std::memcmp(got[i].data(), l.grad().data().data(), got[i].size() * sizeof(float)) == 0);
			i++;
		}
	}
}

TEST_CASE("linear throws on mismatched shapes") {
	Value w = Value::leaf(Tensor({4, 3}), true);
	Value b = Value::leaf(Tensor({4}), true);
	CHECK_THROWS_AS(linear(Value::leaf(Tensor({3})), w, b), TensorShapeError);
	CHECK_THROWS_AS(linear(Value::leaf(Tensor({2, 5})), w, b), TensorShapeError);
	CHECK_THROWS_AS(linear(Value::leaf(Tensor({2, 3})), w, Value::leaf(Tensor({3}))), TensorShapeError);
}

TEST_CASE("attention gradcheck for queries, keys and values, causal and padded") {
	Tensor mask({2, 4}, std::vector<float>{1, 1, 1, 0, 1, 1, 1, 1});
	for (bool causal : {false, true}) {
		Value q = Value::leaf(sine_tensor({2, 4, 6}, 0.3f), true);
		Value k = Value::leaf(sine_tensor({2, 4, 6}, 0.7f), true);
		Value v = Value::leaf(sine_tensor({2, 4, 6}, 1.3f), true);
		CHECK(gradcheck([k, v, mask, causal](Value t) mutable { return weighted_sum(attention(t, k, v, 2, mask, causal)); }, q));
		CHECK(gradcheck([q, v, mask, causal](Value t) mutable { return weighted_sum(attention(q, t, v, 2, mask, causal)); }, k));
		CHECK(gradcheck([q, k, mask, causal](Value t) mutable { return weighted_sum(attention(q, k, t, 2, mask, causal)); }, v));
	}
	// One node as query, key and value: its gradient is the sum of the three roles
	Value x = Value::leaf(sine_tensor({2, 4, 6}, 0.5f), true);
	CHECK(gradcheck([](Value t) { return weighted_sum(attention(t, t, t, 3, Tensor::empty({0}), true)); }, x));
}

TEST_CASE("attention gives the bits of the composed ops, forward and backward") {
	Tensor mask({3, 5}, std::vector<float>{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
	struct Case { size_t tq, tk; bool causal, padded; };
	for (Case c : {Case{5, 5, true, false}, Case{5, 5, true, true}, Case{4, 5, false, true}, Case{5, 5, false, false}}) {
		Value q = Value::leaf(sine_tensor({3, c.tq, 8}, 0.3f), true);
		Value k = Value::leaf(sine_tensor({3, c.tk, 8}, 0.7f), true);
		Value v = Value::leaf(sine_tensor({3, c.tk, 8}, 1.3f), true);
		const Tensor m = c.padded ? mask : Tensor::empty({0});
		Value fused = attention(q, k, v, 4, m, c.causal);
		weighted_sum(fused).backward();
		std::vector<std::vector<float>> got;
		for (const Value& l : {q, k, v}) got.emplace_back(l.grad().data().begin(), l.grad().data().end());
		for (const Value& l : {q, k, v}) l.zero_grad();
		Value composed = composed_attention(q, k, v, 4, m, c.causal);
		weighted_sum(composed).backward();
		CHECK(fused.data().shape() == composed.data().shape());
		CHECK(std::memcmp(fused.data().data().data(), composed.data().data().data(), fused.data().size() * sizeof(float)) == 0);
		size_t i = 0;
		for (const Value& l : {q, k, v}) {
			CHECK(std::memcmp(got[i].data(), l.grad().data().data(), got[i].size() * sizeof(float)) == 0);
			i++;
		}
	}
}

TEST_CASE("attention and linear are bitwise identical for any thread count") {
	Value q = Value::leaf(sine_tensor({9, 16, 32}, 0.3f), true);
	Value k = Value::leaf(sine_tensor({9, 16, 32}, 0.7f), true);
	Value v = Value::leaf(sine_tensor({9, 16, 32}, 1.3f), true);
	CHECK(same_bits_for_any_thread_count([&] { return weighted_sum(attention(q, k, v, 4, Tensor::empty({0}), true)); }, {q, k, v}));
	// 1100 rows: the weight gradient is a small output over a long depth, which sgemm cuts into pieces
	Value x = Value::leaf(sine_tensor({1100, 48}, 0.3f), true);
	Value w = Value::leaf(sine_tensor({40, 48}, 0.7f), true);
	Value b = Value::leaf(sine_tensor({40}, 1.1f), true);
	CHECK(same_bits_for_any_thread_count([&] { return weighted_sum(linear(x, w, b)); }, {x, w, b}));
}

TEST_CASE("attention throws on mismatched shapes") {
	Value q = Value::leaf(Tensor({2, 4, 6}));
	CHECK_THROWS_AS(attention(q, q, q, 4, Tensor::empty({0}), false), TensorShapeError);
	CHECK_THROWS_AS(attention(q, Value::leaf(Tensor({2, 5, 6})), Value::leaf(Tensor({2, 4, 6})), 2, Tensor::empty({0}), false), TensorShapeError);
	CHECK_THROWS_AS(attention(q, Value::leaf(Tensor({2, 5, 6})), Value::leaf(Tensor({2, 5, 6})), 2, Tensor::empty({0}), true), TensorShapeError);
	CHECK_THROWS_AS(attention(q, q, q, 2, Tensor({2, 3}), false), TensorShapeError);
}

TEST_CASE("an intermediate gradient is stored by its first consumer and added to by the next") {
	// h feeds two ops; neither zeroes it first, yet its gradient is the sum of both contributions
	Value x = Value::leaf(Tensor({4}, std::vector<float>{1.0f, -2.0f, 3.0f, -4.0f}), true);
	Value h = scale(x, 2.0f);
	Value loss = sum(add(relu(h), reshape(h, {4})));
	loss.backward();
	CHECK(h.grad().data() == std::vector<float>{2.0f, 1.0f, 2.0f, 1.0f});
	CHECK(x.grad().data() == std::vector<float>{4.0f, 2.0f, 4.0f, 2.0f});
	// A second backward over the same graph starts from nothing again instead of accumulating
	x.zero_grad();
	loss.backward();
	CHECK(h.grad().data() == std::vector<float>{2.0f, 1.0f, 2.0f, 1.0f});
}

TEST_CASE("a stored gradient has the bits of adding to zero: negative zero becomes positive zero") {
	// relu passes -0 * 1 = -0 back; accumulated into a zeroed buffer that is +0
	Value x = Value::leaf(Tensor({2}, std::vector<float>{1.0f, 2.0f}), true);
	Value h = reshape(x, {2});
	Value loss = sum(mul(relu(h), Value::leaf(Tensor({2}, std::vector<float>{-0.0f, 3.0f}))));
	loss.backward();
	CHECK(!std::signbit(h.grad().data()[0]));
	CHECK(h.grad().data()[1] == 3.0f);
}

TEST_CASE("a non-leaf gradient nobody wrote reads as zeros") {
	// The slice reaches only half of h, through a rule that adds into a zeroed buffer
	Value x = Value::leaf(Tensor({4}, 1.0f), true);
	Value h = scale(x, 3.0f);
	sum(slice(h, 0, 0, 2)).backward();
	CHECK(h.grad().data() == std::vector<float>{1.0f, 1.0f, 0.0f, 0.0f});
}

TEST_CASE("maxpool2d 2x2 window keeps the first of equal maxima") {
	Value x = Value::leaf(Tensor({1, 1, 2, 4}, std::vector<float>{5, 5, 1, 7, 5, 5, 7, 7}), true);
	Value y = maxpool2d(x, 2, 2);
	CHECK(y.data().data() == std::vector<float>{5.0f, 7.0f});
	sum(y).backward();
	CHECK(x.grad().data() == std::vector<float>{1, 0, 0, 1, 0, 0, 0, 0});
}

TEST_CASE("cross_entropy_loss throws on a target outside the classes") {
	Value logits = Value::leaf(Tensor({2, 3}, 0.5f), true);
	CHECK_THROWS_AS(cross_entropy_loss(logits, Tensor({2}, std::vector<float>{0.0f, 3.0f})), TensorIndexError);
	CHECK_THROWS_AS(cross_entropy_loss(logits, Tensor({2}, std::vector<float>{-1.0f, 1.0f})), TensorIndexError);
}

TEST_CASE("gelu matches the tanh formula and is bitwise identical for any thread count") {
	Tensor t({70000});
	for (size_t i = 0; i < t.size(); i++) t.data()[i] = -6.0f + 12.0f * static_cast<float>(i) / static_cast<float>(t.size());
	Value x = Value::leaf(t, true);
	Value y = gelu(x);
	for (size_t i = 0; i < t.size(); i += 97) {
		const double v = t.data()[i];
		const double want = 0.5 * v * (1.0 + std::tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
		CHECK(std::fabs(y.data().data()[i] - want) < 2e-6);
	}
	CHECK(same_bits_for_any_thread_count([&] { return sum(gelu(x)); }, {x}));
}
