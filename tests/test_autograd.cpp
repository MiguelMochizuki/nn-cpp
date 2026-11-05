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

#include <algorithm>
#include <cmath>
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
		std::vector<float> out1, gx1, gw1, gb1;
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
			std::vector<float> out1, g1;
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
