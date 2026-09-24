# Math-readable operators, allocation-churn reduction, basic parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add operator overloads to `Tensor`/`Value`, cut allocation/copy churn in the autograd/optimizer hot paths, and add a basic stdlib-only thread pool applied to the library's embarrassingly-parallel loops (elementwise ops, matmul with cache tiling, axis reductions, conv2d/maxpool2d loops).

**Architecture:** Mechanical, behavior-preserving changes layered onto the existing four-layer library (`Tensor` → autograd → `nn` → `optim`/`data`): new operator overloads wrap existing free functions, grad-accumulation sites switch from allocate-and-reassign to in-place compound assignment, and a new `nn::parallel_for` primitive gets dropped into existing hot loops without changing their math.

**Tech Stack:** C++17, CMake 3.16+, doctest (fetched by the existing `cmake/FetchDoctest.cmake`), `<thread>`/`<mutex>`/`<condition_variable>` from the standard library only.

**Spec:** `docs/superpowers/specs/2026-09-24-math-memory-parallel-design.md`

## Global Constraints

- No new external dependencies. Parallelism uses only `<thread>`, `<mutex>`, `<condition_variable>`, `<functional>`; CMake must add `find_package(Threads REQUIRED)` + `target_link_libraries(nn_core PUBLIC Threads::Threads)` so the pthread runtime links on Linux/GCC.
- `matmul` stays a free function on both `Tensor` and `Value` — never overloaded onto `operator*` (elementwise multiply only).
- `Tensor` compound-assignment operators (`+= -= *= /=`) require the RHS to have the exact same shape as `*this`; throw `TensorShapeError` on any mismatch, including same-total-size-different-shape. No broadcasting into a differently-shaped LHS.
- `Value` gets no compound-assignment operators — graph nodes are immutable once built; only `.grad()`'s underlying `Tensor` is ever mutated in place.
- The thread pool is a single lazily-initialized global singleton; no dependency-injected/configurable pool object anywhere in the public API.
- `parallel_for`'s size threshold and matmul's tile size are fixed internal constants, each tagged with a `ponytail:` comment naming the ceiling (not adaptive/auto-tuned) and the upgrade path (revisit if profiling shows the constant wrong).
- Match existing file conventions exactly: tabs for indentation, `/** filename \n * Author: Miguel Mochizuki Silva \n * Description: ... */` header comment on every new file, `/* Public, ... */` doc-comment block on every new public declaration.
- No SIMD, no copy-on-write/arena/view `Tensor` storage, no `operator()` indexing sugar, no `DataLoader` parallelization — all explicitly out of scope (spec's Non-goals and Future Work sections).

## Review Focus

- Compound-assignment operators (`+= -= *= /=`) must reject a same-total-element-count-but-different-shape RHS (e.g. `{2,3} += {3,2}` or `{2,3} += {6}`), not just a different element count — a naive `data_.size() == other.data_.size()` check would silently corrupt data instead of throwing. Pinned in Task 1.
- `parallel_for`'s below/above-threshold fallback must be correct exactly at the boundary (`n == threshold - 1`, `n == threshold`, `n == threshold + 1`) — an off-by-one here would either serialize large tensors that should parallelize or dispatch tiny tensors to the pool. Pinned in Task 6.
- New `Value` operators (`operator-` unary, `operator*` with a negative scalar) must carry the correct sign through `backward()`, not just through the forward value — a sign bug here would silently produce wrong gradients rather than a crash. Pinned in Task 2.
- Matmul tiling must not drop or double-count partial tiles when `k` or `n` isn't an exact multiple of the block size — an off-by-one in a tile boundary would corrupt just the last row/column block, easy to miss with only evenly-divisible test shapes. Pinned in Task 7.
- The parallel code paths in `im2col`/`col2im`/`conv2d`'s bias-add/`maxpool2d` are only exercised when the batch dimension `N` is above the parallel threshold — every existing conv2d/maxpool2d test uses small `N`, so none of them touch the parallel branch at all. Pinned in Task 8.

---

## File Structure

- `include/nn/tensor.hpp`, `src/tensor.cpp` — `Tensor` operators, `zero_()`, parallelized/tiled hot loops. (Tasks 1, 7)
- `include/nn/autograd/ops.hpp`, `src/autograd/ops.cpp` — `Value` operators, in-place grad accumulation, dropped redundant copies, parallelized conv/pool loops. (Tasks 2, 3, 8)
- `src/autograd/value.cpp` — in-place grad zeroing. (Task 4)
- `src/optim/sgd.cpp` — in-place parameter update. (Task 5)
- `include/nn/parallel.hpp`, `src/parallel.cpp` — new: the thread pool and `parallel_for`. (Task 6)
- `CMakeLists.txt` — add `src/parallel.cpp` to `nn_core`, link `Threads::Threads`. (Task 6)
- `tests/test_parallel.cpp`, `tests/CMakeLists.txt` — new test file, wired into `nn_tests`. (Tasks 6, 7, 8)
- `README.md` — "Scope" section update. (Task 9)

---

### Task 1: Tensor operator overloads and `zero_()`

**Files:**
- Modify: `include/nn/tensor.hpp` (member declarations near the existing `data()` overloads around line 143, and free-function declarations after the existing `matmul` declaration around line 311)
- Modify: `src/tensor.cpp` (member definitions near line 140, free-function definitions after the existing `matmul` definition around line 346)
- Test: `tests/test_tensor.cpp`

**Interfaces:**
- Consumes: existing `Tensor` class, existing free functions `add/sub/mul/div/scale` (`src/tensor.cpp`), existing `TensorShapeError`.
- Produces: `Tensor& Tensor::operator+=/-=/*=/= (const Tensor&)`, `Tensor& Tensor::operator*=(float)`, `void Tensor::zero_()`, free functions `operator+ operator- operator* operator/ (const Tensor&, const Tensor&)`, `operator*(const Tensor&, float)`, `operator*(float, const Tensor&)`, unary `operator-(const Tensor&)`. All consumed by Tasks 3, 4, 5, 7.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tensor.cpp` (after the existing `"matmul inner dimension mismatch throws"` test case):

```cpp
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
```

- [ ] **Step 2: Configure and build, verify the new tests fail to compile**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL — compile error, `Tensor` has no `operator+=`/`operator+`/`zero_` etc.

- [ ] **Step 3: Implement in `include/nn/tensor.hpp`**

Add inside the `Tensor` class body, after the existing `data()` read-write overload (around line 143), before `reshape`:

```cpp
	/* Public, in-place elementwise addition, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element incremented by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator+=(const Tensor& other);

	/* Public, in-place elementwise subtraction, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element decremented by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator-=(const Tensor& other);

	/* Public, in-place elementwise multiplication, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element multiplied by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator*=(const Tensor& other);

	/* Public, in-place elementwise division, same shape only
	 *
	 * Parameters:
	 * Tensor other: operand; must have the exact same shape as this tensor
	 *
	 * Returns Tensor&: *this, each element divided by the matching element of other;
	 * throws TensorShapeError if shapes differ
	 */
	Tensor& operator/=(const Tensor& other);

	/* Public, in-place scalar multiplication
	 *
	 * Parameters:
	 * float scalar: multiplier
	 *
	 * Returns Tensor&: *this, every element multiplied by scalar
	 */
	Tensor& operator*=(float scalar);

	/* Public, set every element to zero in place
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_();
```

Add after the existing `matmul` free-function declaration (around line 311), before the closing `}  // namespace nn`:

```cpp
/* Public, elementwise addition operator, alias for add(a, b) */
Tensor operator+(const Tensor& a, const Tensor& b);

/* Public, elementwise subtraction operator, alias for sub(a, b) */
Tensor operator-(const Tensor& a, const Tensor& b);

/* Public, elementwise multiplication operator, alias for mul(a, b) */
Tensor operator*(const Tensor& a, const Tensor& b);

/* Public, elementwise division operator, alias for div(a, b) */
Tensor operator/(const Tensor& a, const Tensor& b);

/* Public, scalar multiplication operator, alias for scale(a, scalar) */
Tensor operator*(const Tensor& a, float scalar);
Tensor operator*(float scalar, const Tensor& a);

/* Public, negation operator, alias for scale(a, -1.0f) */
Tensor operator-(const Tensor& a);
```

- [ ] **Step 4: Implement in `src/tensor.cpp`**

Add after the existing `Tensor::data()` read-write definition (around line 140), before `Tensor::reshape`:

```cpp
Tensor& Tensor::operator+=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] += other.data_[i];
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] -= other.data_[i];
	return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] *= other.data_[i];
	return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
	if (shape_ != other.shape_) throw TensorShapeError();
	for (size_t i = 0; i < data_.size(); i++) data_[i] /= other.data_[i];
	return *this;
}

Tensor& Tensor::operator*=(float scalar) {
	for (float& v : data_) v *= scalar;
	return *this;
}

void Tensor::zero_() {
	std::fill(data_.begin(), data_.end(), 0.0f);
}
```

Add after the existing `matmul` free-function definition (around line 346), before the closing `}  // namespace nn`:

```cpp
Tensor operator+(const Tensor& a, const Tensor& b) { return add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return div(a, b); }
Tensor operator*(const Tensor& a, float scalar) { return scale(a, scalar); }
Tensor operator*(float scalar, const Tensor& a) { return scale(a, scalar); }
Tensor operator-(const Tensor& a) { return scale(a, -1.0f); }
```

`<algorithm>` is already included in `src/tensor.cpp` (used by `std::max`), so `std::fill` needs no new include.

- [ ] **Step 5: Build and run tests**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS, including all tests added in Step 1.

- [ ] **Step 6: Commit**

```bash
git add include/nn/tensor.hpp src/tensor.cpp tests/test_tensor.cpp
git commit -m "feat: add Tensor operator overloads and in-place zero_()"
```

---

### Task 2: Value operator overloads

**Files:**
- Modify: `include/nn/autograd/ops.hpp` (add after the existing `mse_loss` declaration, around line 188, before the closing namespace)
- Modify: `src/autograd/ops.cpp` (add after the existing `mse_loss` definition, around line 452, before the closing namespace)
- Test: `tests/test_autograd.cpp`

**Interfaces:**
- Consumes: existing `Value` class, existing free functions `add/sub/mul/scale` for `Value` (`src/autograd/ops.cpp`).
- Produces: free functions `operator+ operator- operator* (const Value&, const Value&)`, `operator*(const Value&, float)`, `operator*(float, const Value&)`, unary `operator-(const Value&)`. Used by Task 3 only insofar as later ops.cpp edits may read this file; no other task consumes these directly, but they are part of the library's public surface per the spec.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_autograd.cpp` (after the existing `"mse_loss gradcheck"` test case, at the end of the file):

```cpp
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
```

- [ ] **Step 2: Build, verify the new tests fail to compile**

Run: `cmake --build build -j`
Expected: FAIL — compile error, `Value` has no `operator+`/`operator*`/`operator-` etc.

- [ ] **Step 3: Implement in `include/nn/autograd/ops.hpp`**

Add after the existing `mse_loss` declaration, before the closing `}  // namespace nn`:

```cpp
/* Public, differentiable elementwise addition operator, alias for add(a, b) */
Value operator+(const Value& a, const Value& b);

/* Public, differentiable elementwise subtraction operator, alias for sub(a, b) */
Value operator-(const Value& a, const Value& b);

/* Public, differentiable elementwise multiplication operator, alias for mul(a, b) */
Value operator*(const Value& a, const Value& b);

/* Public, differentiable scalar multiplication operator, alias for scale(a, scalar) */
Value operator*(const Value& a, float scalar);
Value operator*(float scalar, const Value& a);

/* Public, differentiable negation operator, alias for scale(a, -1.0f) */
Value operator-(const Value& a);
```

- [ ] **Step 4: Implement in `src/autograd/ops.cpp`**

Add after the existing `mse_loss` definition, before the closing `}  // namespace nn`:

```cpp
Value operator+(const Value& a, const Value& b) { return add(a, b); }
Value operator-(const Value& a, const Value& b) { return sub(a, b); }
Value operator*(const Value& a, const Value& b) { return mul(a, b); }
Value operator*(const Value& a, float scalar) { return scale(a, scalar); }
Value operator*(float scalar, const Value& a) { return scale(a, scalar); }
Value operator-(const Value& a) { return scale(a, -1.0f); }
```

- [ ] **Step 5: Build and run tests**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS, including all tests added in Step 1.

- [ ] **Step 6: Commit**

```bash
git add include/nn/autograd/ops.hpp src/autograd/ops.cpp tests/test_autograd.cpp
git commit -m "feat: add Value operator overloads"
```

---

### Task 3: Grad-accumulation and copy churn in `src/autograd/ops.cpp`

**Files:**
- Modify: `src/autograd/ops.cpp` (backward closures of `add`, `sub`, `mul`, `scale`, `sum`, `transpose`, `reshape`, `matmul`, `relu`, `sigmoid`, `tanh`, `softmax`, `conv2d`, `maxpool2d`, `cross_entropy_loss`)

**Interfaces:**
- Consumes: `Tensor::operator+= -= *=` from Task 1. No public signatures change — every function in this file keeps its existing name, parameters, and return type.
- Produces: nothing new; this task only reduces allocations inside existing functions. No later task depends on new interfaces from this one.

This is a behavior-preserving refactor: every existing test already pins the math, so the safety net is the full existing suite, not a new test. Do not add new test cases in this task — a diff that changes `X.grad() = add(X.grad(), g)` to `X.grad() += g` or removes a redundant copy must produce bit-identical results to before, since nothing about the arithmetic changes, only how it's stored.

- [ ] **Step 1: Confirm the baseline passes before touching anything**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS (this is the pre-refactor baseline; if anything already fails, stop and investigate before proceeding — this task must not be the one that introduces or hides a pre-existing failure).

- [ ] **Step 2: Rewrite grad accumulation in `add`, `sub`, `mul`, `scale`**

In `add`, replace:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), sum_to_shape(grad_out, a.data().shape()));
		if (b.requires_grad()) b.grad() = add(b.grad(), sum_to_shape(grad_out, b.data().shape()));
	});
```
with:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() += sum_to_shape(grad_out, a.data().shape());
		if (b.requires_grad()) b.grad() += sum_to_shape(grad_out, b.data().shape());
	});
```

In `sub`, replace:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), sum_to_shape(grad_out, a.data().shape()));
		if (b.requires_grad()) b.grad() = sub(b.grad(), sum_to_shape(grad_out, b.data().shape()));
	});
```
with:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() += sum_to_shape(grad_out, a.data().shape());
		if (b.requires_grad()) b.grad() -= sum_to_shape(grad_out, b.data().shape());
	});
```

In `mul`, replace:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			a.grad() = add(a.grad(), sum_to_shape(mul(grad_out, b.data()), a.data().shape()));
		}
		if (b.requires_grad()) {
			b.grad() = add(b.grad(), sum_to_shape(mul(grad_out, a.data()), b.data().shape()));
		}
	});
```
with:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			a.grad() += sum_to_shape(mul(grad_out, b.data()), a.data().shape());
		}
		if (b.requires_grad()) {
			b.grad() += sum_to_shape(mul(grad_out, a.data()), b.data().shape());
		}
	});
```

In `scale`, replace:
```cpp
	return Value::node(result, {a}, [a, scalar](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() = add(a.grad(), scale(grad_out, scalar));
	});
```
with:
```cpp
	return Value::node(result, {a}, [a, scalar](const Tensor& grad_out) {
		if (a.requires_grad()) a.grad() += scale(grad_out, scalar);
	});
```

- [ ] **Step 3: Rewrite grad accumulation in `sum`, `transpose`, `reshape`, `matmul`**

In `sum`, replace:
```cpp
	return Value::node(result, {x}, [x](const Tensor& grad_out) {
		if (x.requires_grad()) {
			Tensor ones(x.data().shape(), grad_out.data()[0]);
			x.grad() = add(x.grad(), ones);
		}
	});
```
with:
```cpp
	return Value::node(result, {x}, [x](const Tensor& grad_out) {
		if (x.requires_grad()) {
			Tensor ones(x.data().shape(), grad_out.data()[0]);
			x.grad() += ones;
		}
	});
```

In `transpose`, replace:
```cpp
	return Value::node(result, {x}, [x, inverse](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() = add(x.grad(), grad_out.transpose(inverse));
	});
```
with:
```cpp
	return Value::node(result, {x}, [x, inverse](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() += grad_out.transpose(inverse);
	});
```

In `reshape`, replace:
```cpp
	return Value::node(result, {x}, [x, orig_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() = add(x.grad(), grad_out.reshape(orig_shape));
	});
```
with:
```cpp
	return Value::node(result, {x}, [x, orig_shape](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		x.grad() += grad_out.reshape(orig_shape);
	});
```

In `matmul`, replace:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, transpose_last_two(b.data()));
			a.grad() = add(a.grad(), sum_to_shape(grad_a, a.data().shape()));
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(transpose_last_two(a.data()), grad_out);
			b.grad() = add(b.grad(), sum_to_shape(grad_b, b.data().shape()));
		}
	});
```
with:
```cpp
	return Value::node(result, {a, b}, [a, b](const Tensor& grad_out) {
		if (a.requires_grad()) {
			Tensor grad_a = matmul(grad_out, transpose_last_two(b.data()));
			a.grad() += sum_to_shape(grad_a, a.data().shape());
		}
		if (b.requires_grad()) {
			Tensor grad_b = matmul(transpose_last_two(a.data()), grad_out);
			b.grad() += sum_to_shape(grad_b, b.data().shape());
		}
	});
```

- [ ] **Step 4: Rewrite `relu`, and drop the redundant `saved` copy in `sigmoid`/`tanh`/`softmax`**

In `relu`, replace the last line of the backward closure, `x.grad() = add(x.grad(), local_grad);`, with `x.grad() += local_grad;`. Nothing else in `relu` changes (it never had a `saved` copy — it reads `x.data()` directly).

Replace the whole `sigmoid` function:
```cpp
Value sigmoid(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = 1.0f / (1.0f + std::exp(-v));
	Tensor saved = result;
	return Value::node(result, {x}, [x, saved](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(saved.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float s = saved.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * s * (1.0f - s);
		}
		x.grad() = add(x.grad(), local_grad);
	});
}
```
with:
```cpp
Value sigmoid(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = 1.0f / (1.0f + std::exp(-v));
	return Value::node(result, {x}, [x, result](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(result.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float s = result.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * s * (1.0f - s);
		}
		x.grad() += local_grad;
	});
}
```

Replace the whole `tanh` function:
```cpp
Value tanh(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = std::tanh(v);
	Tensor saved = result;
	return Value::node(result, {x}, [x, saved](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(saved.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float t = saved.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * (1.0f - t * t);
		}
		x.grad() = add(x.grad(), local_grad);
	});
}
```
with:
```cpp
Value tanh(const Value& x) {
	Tensor result = x.data();
	for (float& v : result.data()) v = std::tanh(v);
	return Value::node(result, {x}, [x, result](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor local_grad(result.shape());
		for (size_t i = 0; i < local_grad.size(); i++) {
			float t = result.data()[i];
			local_grad.data()[i] = grad_out.data()[i] * (1.0f - t * t);
		}
		x.grad() += local_grad;
	});
}
```

Replace the whole `softmax` function:
```cpp
Value softmax(const Value& x, int axis) {
	Tensor max_val = insert_axis(x.data().max(axis), axis);
	Tensor shifted = sub(x.data(), max_val);
	Tensor exp_val = shifted;
	for (float& v : exp_val.data()) v = std::exp(v);
	Tensor sum_val = insert_axis(exp_val.sum(axis), axis);
	Tensor result = div(exp_val, sum_val);
	Tensor saved = result;

	return Value::node(result, {x}, [x, saved, axis](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor prod = mul(grad_out, saved);
		Tensor sum_prod = insert_axis(prod.sum(axis), axis);
		Tensor local_grad = mul(saved, sub(grad_out, sum_prod));
		x.grad() = add(x.grad(), local_grad);
	});
}
```
with:
```cpp
Value softmax(const Value& x, int axis) {
	Tensor max_val = insert_axis(x.data().max(axis), axis);
	Tensor shifted = sub(x.data(), max_val);
	Tensor exp_val = shifted;
	for (float& v : exp_val.data()) v = std::exp(v);
	Tensor sum_val = insert_axis(exp_val.sum(axis), axis);
	Tensor result = div(exp_val, sum_val);

	return Value::node(result, {x}, [x, result, axis](const Tensor& grad_out) {
		if (!x.requires_grad()) return;
		Tensor prod = mul(grad_out, result);
		Tensor sum_prod = insert_axis(prod.sum(axis), axis);
		Tensor local_grad = mul(result, sub(grad_out, sum_prod));
		x.grad() += local_grad;
	});
}
```

- [ ] **Step 5: Drop `saved_col` and rewrite grad accumulation in `conv2d`**

In `conv2d`, replace:
```cpp
	Tensor saved_col = col;
	std::vector<size_t> x_shape = x.data().shape();

	return Value::node(
		result, {x, weight, bias},
		[x, weight, bias, saved_col, x_shape, Cout, Cin, kh, kw, stride, padding, h_out, w_out](
			const Tensor& grad_out) {
			Tensor grad_out_mat = grad_out.reshape({grad_out.shape()[0], Cout, h_out * w_out});

			if (bias.requires_grad()) {
				Tensor grad_bias = grad_out_mat.sum(0).sum(1);
				bias.grad() = add(bias.grad(), grad_bias);
			}
			if (weight.requires_grad()) {
				Tensor grad_weight_batched = matmul(grad_out_mat, transpose_last_two(saved_col));
				Tensor grad_weight = grad_weight_batched.sum(0).reshape({Cout, Cin, kh, kw});
				weight.grad() = add(weight.grad(), grad_weight);
			}
			if (x.requires_grad()) {
				Tensor weight_mat_t = transpose_last_two(weight.data().reshape({Cout, Cin * kh * kw}));
				Tensor grad_col = matmul(weight_mat_t, grad_out_mat);
				Tensor grad_x =
					col2im(grad_col, x_shape, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
				x.grad() = add(x.grad(), grad_x);
			}
		});
```
with:
```cpp
	std::vector<size_t> x_shape = x.data().shape();

	return Value::node(
		result, {x, weight, bias},
		[x, weight, bias, col, x_shape, Cout, Cin, kh, kw, stride, padding, h_out, w_out](
			const Tensor& grad_out) {
			Tensor grad_out_mat = grad_out.reshape({grad_out.shape()[0], Cout, h_out * w_out});

			if (bias.requires_grad()) {
				Tensor grad_bias = grad_out_mat.sum(0).sum(1);
				bias.grad() += grad_bias;
			}
			if (weight.requires_grad()) {
				Tensor grad_weight_batched = matmul(grad_out_mat, transpose_last_two(col));
				Tensor grad_weight = grad_weight_batched.sum(0).reshape({Cout, Cin, kh, kw});
				weight.grad() += grad_weight;
			}
			if (x.requires_grad()) {
				Tensor weight_mat_t = transpose_last_two(weight.data().reshape({Cout, Cin * kh * kw}));
				Tensor grad_col = matmul(weight_mat_t, grad_out_mat);
				Tensor grad_x =
					col2im(grad_col, x_shape, kh, kw, static_cast<size_t>(stride), static_cast<size_t>(padding));
				x.grad() += grad_x;
			}
		});
```

- [ ] **Step 6: Rewrite grad accumulation in `maxpool2d` and `cross_entropy_loss`**

In `maxpool2d`, replace the last line of the backward closure, `x.grad() = add(x.grad(), grad_x);`, with `x.grad() += grad_x;`.

In `cross_entropy_loss`, replace the last line of the backward closure, `logits.grad() = add(logits.grad(), local_grad);`, with `logits.grad() += local_grad;`.

- [ ] **Step 7: Build and run the full suite, confirm no regression**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: PASS — every test that passed in Step 1 still passes, with identical numeric results (this refactor changes storage, not arithmetic).

- [ ] **Step 8: Commit**

```bash
git add src/autograd/ops.cpp
git commit -m "refactor: use in-place grad accumulation and drop redundant tensor copies in ops.cpp"
```

---

### Task 4: In-place grad zeroing in `src/autograd/value.cpp`

**Files:**
- Modify: `src/autograd/value.cpp`

**Interfaces:**
- Consumes: `Tensor::zero_()` from Task 1.
- Produces: nothing new; `Value::backward()` and `Value::zero_grad()` keep their existing signatures.

- [ ] **Step 1: Confirm the baseline passes**

Run: `ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS.

- [ ] **Step 2: Add `#include <algorithm>`**

At the top of `src/autograd/value.cpp`, add `#include <algorithm>` alongside the existing `#include <unordered_set>`.

- [ ] **Step 3: Rewrite the backward-pass grad reset**

Replace:
```cpp
	for (const Value& v : topo_order) {
		if (v.impl()->backward_fn) {
			v.impl()->grad = Tensor(v.impl()->data.shape(), 0.0f);
		}
	}
	impl_->grad = Tensor(impl_->data.shape(), 1.0f);
```
with:
```cpp
	for (const Value& v : topo_order) {
		if (v.impl()->backward_fn) {
			v.impl()->grad.zero_();
		}
	}
	std::fill(impl_->grad.data().begin(), impl_->grad.data().end(), 1.0f);
```

- [ ] **Step 4: Rewrite `Value::zero_grad()`**

Replace:
```cpp
void Value::zero_grad() const { impl_->grad = Tensor(impl_->data.shape(), 0.0f); }
```
with:
```cpp
void Value::zero_grad() const { impl_->grad.zero_(); }
```

- [ ] **Step 5: Build and run the full suite**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: PASS — in particular `"zero_grad resets grad to zero"`, `"calling backward twice without zero_grad accumulates"`, `"calling backward twice on the SAME graph accumulates onto the leaf without compounding intermediate grads"`, and `"backward, zero_grad, backward again on the same graph gives one pass worth of gradient"` (`tests/test_autograd.cpp`) must all still pass — these are exactly the tests that pin the reset-per-call behavior this task touches.

- [ ] **Step 6: Commit**

```bash
git add src/autograd/value.cpp
git commit -m "refactor: zero grad tensors in place instead of reallocating"
```

---

### Task 5: In-place update in `src/optim/sgd.cpp`

**Files:**
- Modify: `src/optim/sgd.cpp`

**Interfaces:**
- Consumes: `Tensor::operator+= -= *=` from Task 1.
- Produces: nothing new; `SGD::step()` keeps its existing signature.

- [ ] **Step 1: Confirm the baseline passes**

Run: `ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS.

- [ ] **Step 2: Rewrite `SGD::step()`**

Replace:
```cpp
void SGD::step() {
	for (size_t i = 0; i < params_.size(); i++) {
		Tensor update = params_[i].grad();
		if (momentum_ > 0.0f) {
			velocity_[i] = add(scale(velocity_[i], momentum_), params_[i].grad());
			update = velocity_[i];
		}
		params_[i].data() = sub(params_[i].data(), scale(update, lr_));
	}
}
```
with:
```cpp
void SGD::step() {
	for (size_t i = 0; i < params_.size(); i++) {
		if (momentum_ > 0.0f) {
			velocity_[i] *= momentum_;
			velocity_[i] += params_[i].grad();
			params_[i].data() -= velocity_[i] * lr_;
		} else {
			params_[i].data() -= params_[i].grad() * lr_;
		}
	}
}
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS — `tests/test_optim.cpp`'s three test cases (no-momentum update, `zero_grad`, momentum accumulation across two steps) all still pass with identical expected values.

- [ ] **Step 4: Commit**

```bash
git add src/optim/sgd.cpp
git commit -m "refactor: use in-place tensor updates in SGD::step"
```

---

### Task 6: Thread pool and `parallel_for`

**Files:**
- Create: `include/nn/parallel.hpp`
- Create: `src/parallel.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/test_parallel.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `void nn::parallel_for(size_t n, const std::function<void(size_t start, size_t end)>& fn)`. Consumed by Tasks 7 and 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_parallel.cpp`:

```cpp
/**
 * test_parallel.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for nn::parallel_for
 */

#include "doctest/doctest.h"
#include "nn/parallel.hpp"

#include <atomic>
#include <vector>

using namespace nn;

TEST_CASE("parallel_for covers every index exactly once, below threshold") {
	size_t n = 100;
	std::vector<std::atomic<int>> hits(n);
	for (auto& h : hits) h = 0;
	parallel_for(n, [&](size_t start, size_t end) {
		for (size_t i = start; i < end; i++) hits[i]++;
	});
	for (size_t i = 0; i < n; i++) CHECK(hits[i] == 1);
}

TEST_CASE("parallel_for covers every index exactly once, above threshold") {
	size_t n = 50000;
	std::vector<std::atomic<int>> hits(n);
	for (auto& h : hits) h = 0;
	parallel_for(n, [&](size_t start, size_t end) {
		for (size_t i = start; i < end; i++) hits[i]++;
	});
	for (size_t i = 0; i < n; i++) CHECK(hits[i] == 1);
}

TEST_CASE("parallel_for covers every index exactly once at the threshold boundary") {
	// kParallelThreshold is 8192 (internal constant in src/parallel.cpp); exercise
	// one below, exactly at, and one above to catch an off-by-one in the fallback check.
	for (size_t n : {static_cast<size_t>(8191), static_cast<size_t>(8192), static_cast<size_t>(8193)}) {
		std::vector<std::atomic<int>> hits(n);
		for (auto& h : hits) h = 0;
		parallel_for(n, [&](size_t start, size_t end) {
			for (size_t i = start; i < end; i++) hits[i]++;
		});
		for (size_t i = 0; i < n; i++) CHECK(hits[i] == 1);
	}
}

TEST_CASE("parallel_for with n = 0 does not call fn") {
	bool called = false;
	parallel_for(0, [&](size_t, size_t) { called = true; });
	CHECK_FALSE(called);
}
```

Add `test_parallel.cpp` to `tests/CMakeLists.txt`'s `nn_tests` executable sources:
```cmake
add_executable(nn_tests
	test_tensor.cpp
	test_autograd.cpp
	test_nn.cpp
	test_optim.cpp
	test_dataloader.cpp
	test_parallel.cpp
)
```

- [ ] **Step 2: Configure and build, verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: FAIL — `nn/parallel.hpp` does not exist yet.

- [ ] **Step 3: Create `include/nn/parallel.hpp`**

```cpp
/**
 * parallel.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Minimal stdlib-only thread pool for splitting elementwise-style loops across cores
 */

#pragma once

#include <cstddef>
#include <functional>

namespace nn {

/* Public, run fn over disjoint contiguous chunks of [0, n) on a shared thread pool
 *
 * Parameters:
 * size_t n: total number of items to process
 * function<void(size_t start, size_t end)> fn: called once per chunk with a disjoint
 * [start, end) range; each call's range only touches memory disjoint from every other
 * call's range, so fn itself needs no locking
 *
 * Returns void: blocks until every chunk has run; for n below an internal threshold,
 * calls fn(0, n) synchronously on the calling thread instead of dispatching to the pool
 */
void parallel_for(size_t n, const std::function<void(size_t start, size_t end)>& fn);

}  // namespace nn
```

- [ ] **Step 4: Create `src/parallel.cpp`**

```cpp
/**
 * parallel.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Minimal stdlib-only thread pool for splitting elementwise-style loops across cores
 */

#include "nn/parallel.hpp"

#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace nn {

namespace {

// ponytail: fixed threshold, not adaptive to measured per-call overhead;
// revisit if profiling shows this constant wrong for a given workload.
constexpr size_t kParallelThreshold = 8192;

class ThreadPool {
public:
	static ThreadPool& instance() {
		static ThreadPool pool;
		return pool;
	}

	void run(size_t n, const std::function<void(size_t, size_t)>& fn) {
		std::unique_lock<std::mutex> lock(mutex_);
		size_t chunk_size = (n + num_threads_ - 1) / num_threads_;
		for (size_t t = 0; t < num_threads_; t++) {
			size_t start = std::min(t * chunk_size, n);
			size_t end = std::min(start + chunk_size, n);
			chunks_[t] = {start, end};
		}
		task_ = &fn;
		pending_ = num_threads_;
		generation_++;
		cv_start_.notify_all();
		cv_done_.wait(lock, [this] { return pending_ == 0; });
		task_ = nullptr;
	}

private:
	ThreadPool() : num_threads_(std::max<size_t>(1, std::thread::hardware_concurrency())) {
		chunks_.resize(num_threads_, {0, 0});
		for (size_t i = 0; i < num_threads_; i++) {
			workers_.emplace_back([this, i] { worker_loop(i); });
		}
	}

	~ThreadPool() {
		{
			std::lock_guard<std::mutex> lock(mutex_);
			shutdown_ = true;
			generation_++;
		}
		cv_start_.notify_all();
		for (std::thread& t : workers_) t.join();
	}

	void worker_loop(size_t index) {
		size_t seen_generation = 0;
		while (true) {
			std::unique_lock<std::mutex> lock(mutex_);
			cv_start_.wait(lock, [this, seen_generation] { return shutdown_ || generation_ != seen_generation; });
			if (shutdown_) return;
			seen_generation = generation_;
			auto [start, end] = chunks_[index];
			const std::function<void(size_t, size_t)>* task = task_;
			lock.unlock();

			if (task && start < end) (*task)(start, end);

			lock.lock();
			pending_--;
			if (pending_ == 0) cv_done_.notify_one();
		}
	}

	size_t num_threads_;
	std::vector<std::thread> workers_;
	std::vector<std::pair<size_t, size_t>> chunks_;

	std::mutex mutex_;
	std::condition_variable cv_start_;
	std::condition_variable cv_done_;
	const std::function<void(size_t, size_t)>* task_ = nullptr;
	size_t generation_ = 0;
	size_t pending_ = 0;
	bool shutdown_ = false;
};

}  // namespace

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
	if (n == 0) return;
	if (n < kParallelThreshold) {
		fn(0, n);
		return;
	}
	ThreadPool::instance().run(n, fn);
}

}  // namespace nn
```

- [ ] **Step 5: Wire `src/parallel.cpp` and threading into `CMakeLists.txt`**

In the top-level `CMakeLists.txt`, replace:
```cmake
add_library(nn_core STATIC
	src/tensor.cpp
	src/autograd/value.cpp
	src/autograd/ops.cpp
	src/nn/linear.cpp
	src/nn/conv2d.cpp
	src/nn/pool2d.cpp
	src/nn/activation.cpp
	src/nn/sequential.cpp
	src/optim/sgd.cpp
	src/data/dataloader.cpp
)

target_include_directories(nn_core PUBLIC include)
```
with:
```cmake
add_library(nn_core STATIC
	src/tensor.cpp
	src/autograd/value.cpp
	src/autograd/ops.cpp
	src/nn/linear.cpp
	src/nn/conv2d.cpp
	src/nn/pool2d.cpp
	src/nn/activation.cpp
	src/nn/sequential.cpp
	src/optim/sgd.cpp
	src/data/dataloader.cpp
	src/parallel.cpp
)

target_include_directories(nn_core PUBLIC include)

find_package(Threads REQUIRED)
target_link_libraries(nn_core PUBLIC Threads::Threads)
```

- [ ] **Step 6: Configure, build, and run tests**

Run: `cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS, including all four `test_parallel.cpp` cases from Step 1.

- [ ] **Step 7: Commit**

```bash
git add include/nn/parallel.hpp src/parallel.cpp CMakeLists.txt tests/test_parallel.cpp tests/CMakeLists.txt
git commit -m "feat: add stdlib-only thread pool and nn::parallel_for"
```

---

### Task 7: Parallelize and tile `src/tensor.cpp` hot loops

**Files:**
- Modify: `src/tensor.cpp` (`elementwise()`, `matmul()`, `Tensor::sum()`, `Tensor::max()`)
- Test: `tests/test_parallel.cpp`

**Interfaces:**
- Consumes: `nn::parallel_for` from Task 6.
- Produces: nothing new; `add/sub/mul/div`, `matmul`, `Tensor::sum/mean/max` keep their existing signatures and results, only their internal loops change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_parallel.cpp`. Add `#include "nn/tensor.hpp"` near the top, alongside the existing `#include "nn/parallel.hpp"`:

```cpp
TEST_CASE("parallelized elementwise add matches expected result above threshold") {
	size_t n = 20000;
	Tensor a({n}, 1.0f);
	Tensor b({n}, 1.0f);
	Tensor c = add(a, b);
	for (float v : c.data()) CHECK(v == doctest::Approx(2.0f));
}

TEST_CASE("parallelized matmul matches expected result above threshold") {
	size_t m = 10000, k = 4, n = 4;
	Tensor a({m, k}, 1.0f);
	Tensor b({k, n}, 1.0f);
	Tensor c = matmul(a, b);
	CHECK(c.shape() == std::vector<size_t>{m, n});
	for (float v : c.data()) CHECK(v == doctest::Approx(static_cast<float>(k)));
}

TEST_CASE("matmul tiling handles k and n not divisible by the tile size") {
	// Tile size is 64 (internal constant in src/tensor.cpp); use dimensions that
	// leave a partial tile on both the k and n axes, above the parallel threshold
	// so the tiled code path (not just the serial fallback) is exercised.
	size_t m = 9000, k = 70, n = 70;
	Tensor a({m, k}, 1.0f);
	Tensor b({k, n}, 1.0f);
	Tensor c = matmul(a, b);
	CHECK(c.shape() == std::vector<size_t>{m, n});
	for (float v : c.data()) CHECK(v == doctest::Approx(static_cast<float>(k)));
}

TEST_CASE("parallelized sum along axis matches expected result above threshold") {
	size_t outer = 10000, axis_size = 5;
	Tensor t({outer, axis_size}, 1.0f);
	Tensor s = t.sum(1);
	CHECK(s.shape() == std::vector<size_t>{outer});
	for (float v : s.data()) CHECK(v == doctest::Approx(static_cast<float>(axis_size)));
}
```

- [ ] **Step 2: Build, verify the new tests fail or the build breaks meaningfully**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: since `add`/`matmul`/`sum` already exist and are already correct today, these specific tests should actually PASS even before Step 3 — they pin *current* behavior, not new behavior. Confirm they pass now, then proceed to Step 3 to add parallelism without breaking them (this task's real regression net is "still passes after," not "fails before").

- [ ] **Step 3: Add `#include "nn/parallel.hpp"` to `src/tensor.cpp`**

At the top of `src/tensor.cpp`, add `#include "nn/parallel.hpp"` alongside the existing includes.

- [ ] **Step 4: Parallelize the `elementwise()` fast path**

Replace:
```cpp
	if (a.shape() == b.shape()) {
		Tensor out(a.shape());
		const float* a_data = a.data().data();
		const float* b_data = b.data().data();
		float* out_data = out.data().data();
		size_t n = a.size();
		for (size_t i = 0; i < n; i++) {
			out_data[i] = op(a_data[i], b_data[i]);
		}
		return out;
	}
```
with:
```cpp
	if (a.shape() == b.shape()) {
		Tensor out(a.shape());
		const float* a_data = a.data().data();
		const float* b_data = b.data().data();
		float* out_data = out.data().data();
		size_t n = a.size();
		parallel_for(n, [&](size_t start, size_t end) {
			for (size_t i = start; i < end; i++) {
				out_data[i] = op(a_data[i], b_data[i]);
			}
		});
		return out;
	}
```

- [ ] **Step 5: Parallelize and tile `matmul()`'s inner loop**

Replace:
```cpp
		for (size_t i = 0; i < m; i++) {
			for (size_t kk = 0; kk < k; kk++) {
				float a_val = a.data()[a_base + i * a_row_stride + kk * a_col_stride];
				for (size_t j = 0; j < n; j++) {
					out.data()[out_base + i * out_row_stride + j * out_col_stride] +=
						a_val * b.data()[b_base + kk * b_row_stride + j * b_col_stride];
				}
			}
		}
```
with:
```cpp
		// ponytail: fixed 64-element tile, not auto-tuned to detected cache size;
		// revisit if profiling shows a different size wins on target hardware.
		constexpr size_t kMatmulTileSize = 64;
		parallel_for(m, [&](size_t i_start, size_t i_end) {
			for (size_t jb = 0; jb < n; jb += kMatmulTileSize) {
				size_t j_end = std::min(jb + kMatmulTileSize, n);
				for (size_t kb = 0; kb < k; kb += kMatmulTileSize) {
					size_t k_end = std::min(kb + kMatmulTileSize, k);
					for (size_t i = i_start; i < i_end; i++) {
						for (size_t kk = kb; kk < k_end; kk++) {
							float a_val = a.data()[a_base + i * a_row_stride + kk * a_col_stride];
							for (size_t j = jb; j < j_end; j++) {
								out.data()[out_base + i * out_row_stride + j * out_col_stride] +=
									a_val * b.data()[b_base + kk * b_row_stride + j * b_col_stride];
							}
						}
					}
				}
			}
		});
```

- [ ] **Step 6: Parallelize `Tensor::sum(axis)`**

Replace:
```cpp
	for (size_t outer = 0; outer < outer_size; outer++) {
		for (size_t a = 0; a < axis_size; a++) {
			size_t src_base = outer * axis_size * inner_size + a * inner_size;
			size_t dst_base = outer * inner_size;
			for (size_t inner = 0; inner < inner_size; inner++) {
				dst[dst_base + inner] += src[src_base + inner];
			}
		}
	}
	return out;
}

Tensor Tensor::mean(int axis) const {
```
with:
```cpp
	parallel_for(outer_size, [&](size_t outer_start, size_t outer_end) {
		for (size_t outer = outer_start; outer < outer_end; outer++) {
			for (size_t a = 0; a < axis_size; a++) {
				size_t src_base = outer * axis_size * inner_size + a * inner_size;
				size_t dst_base = outer * inner_size;
				for (size_t inner = 0; inner < inner_size; inner++) {
					dst[dst_base + inner] += src[src_base + inner];
				}
			}
		}
	});
	return out;
}

Tensor Tensor::mean(int axis) const {
```

- [ ] **Step 7: Parallelize `Tensor::max(axis)`**

Replace:
```cpp
	for (size_t outer = 0; outer < outer_size; outer++) {
		for (size_t a = 0; a < axis_size; a++) {
			size_t src_base = outer * axis_size * inner_size + a * inner_size;
			size_t dst_base = outer * inner_size;
			for (size_t inner = 0; inner < inner_size; inner++) {
				dst[dst_base + inner] = std::max(dst[dst_base + inner], src[src_base + inner]);
			}
		}
	}
	return out;
}

Tensor Tensor::zeros(std::vector<size_t> shape)
```
with:
```cpp
	parallel_for(outer_size, [&](size_t outer_start, size_t outer_end) {
		for (size_t outer = outer_start; outer < outer_end; outer++) {
			for (size_t a = 0; a < axis_size; a++) {
				size_t src_base = outer * axis_size * inner_size + a * inner_size;
				size_t dst_base = outer * inner_size;
				for (size_t inner = 0; inner < inner_size; inner++) {
					dst[dst_base + inner] = std::max(dst[dst_base + inner], src[src_base + inner]);
				}
			}
		}
	});
	return out;
}

Tensor Tensor::zeros(std::vector<size_t> shape)
```

- [ ] **Step 8: Build and run the full suite**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: PASS — the full suite, including every existing `tests/test_tensor.cpp`/`tests/test_autograd.cpp`/`tests/test_nn.cpp` case (small tensors, serial fallback, must stay bit-identical) and the new above-threshold cases from Step 1.

- [ ] **Step 9: Commit**

```bash
git add src/tensor.cpp tests/test_parallel.cpp
git commit -m "perf: parallelize and tile Tensor elementwise/matmul/reduction loops"
```

---

### Task 8: Parallelize `src/autograd/ops.cpp` conv/pool loops

**Files:**
- Modify: `src/autograd/ops.cpp` (`im2col`, `col2im`, the bias-add loop inside `conv2d`, `maxpool2d`'s forward loop)
- Test: `tests/test_parallel.cpp`

**Interfaces:**
- Consumes: `nn::parallel_for` from Task 6.
- Produces: nothing new; `conv2d` and `maxpool2d` keep their existing signatures and results.

- [ ] **Step 1: Verify the `col2im` disjoint-write invariant before parallelizing it**

`col2im` accumulates into `grad_x` (a freshly allocated, contiguous `Tensor grad_x(x_shape)`) via `x_data[n * sN + c * sC + ih * sH + iw * sW] += ...` inside the `for (size_t n = 0; ...)` loop. Confirm in the current code (`src/autograd/ops.cpp`, `col2im` function) that `sN` — `grad_x.strides()[0]` — equals `Cin * H * W`, i.e. the full size of one batch item's slice, so distinct `n` values can never write into the same memory region even though a single input pixel is written by multiple `(ph, pw)` kernel positions *within* the same `n`. This holds because `grad_x` is always a fresh, standard-strides `Tensor` (never a view), so splitting the outer loop by `n` is safe with no atomics.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_parallel.cpp`. Add `#include "nn/autograd/value.hpp"` and `#include "nn/autograd/ops.hpp"` near the top:

```cpp
TEST_CASE("parallelized conv2d forward matches expected result above threshold") {
	size_t N = 9000;
	Value x = Value::leaf(Tensor({N, 1, 3, 3}, 1.0f), false);
	Value weight = Value::leaf(Tensor({1, 1, 3, 3}, 1.0f), false);
	Value bias = Value::leaf(Tensor({1}, 0.0f), false);
	Value out = conv2d(x, weight, bias, 1, 0);
	CHECK(out.data().shape() == std::vector<size_t>{N, 1, 1, 1});
	for (float v : out.data().data()) CHECK(v == doctest::Approx(9.0f));
}

TEST_CASE("parallelized maxpool2d forward matches expected result above threshold") {
	size_t N = 9000;
	Value x = Value::leaf(Tensor({N, 1, 2, 2}, 3.0f), false);
	Value out = maxpool2d(x, 2, 2);
	CHECK(out.data().shape() == std::vector<size_t>{N, 1, 1, 1});
	for (float v : out.data().data()) CHECK(v == doctest::Approx(3.0f));
}
```

- [ ] **Step 3: Build, confirm these pass now (pinning current behavior)**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure -R nn_tests`
Expected: PASS — same reasoning as Task 7 Step 2, this task's net is "still passes after."

- [ ] **Step 4: Add `#include "nn/parallel.hpp"` to `src/autograd/ops.cpp`**

At the top of `src/autograd/ops.cpp`, add `#include "nn/parallel.hpp"` alongside the existing includes.

- [ ] **Step 5: Parallelize `im2col`**

Replace:
```cpp
	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < Cin; c++) {
			for (size_t ph = 0; ph < kh; ph++) {
				for (size_t pw = 0; pw < kw; pw++) {
					size_t row = (c * kh + ph) * kw + pw;
					size_t col_row_base = n * col_sN + row * col_sRow;
					for (size_t oh = 0; oh < h_out; oh++) {
						long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
						bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
						for (size_t ow = 0; ow < w_out; ow++) {
							long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
							size_t col_idx = oh * w_out + ow;
							float v = 0.0f;
							if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
								v = x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW];
							}
							col_data[col_row_base + col_idx] = v;
						}
					}
				}
			}
		}
	}
	return col;
```
with:
```cpp
	parallel_for(N, [&](size_t n_start, size_t n_end) {
		for (size_t n = n_start; n < n_end; n++) {
			for (size_t c = 0; c < Cin; c++) {
				for (size_t ph = 0; ph < kh; ph++) {
					for (size_t pw = 0; pw < kw; pw++) {
						size_t row = (c * kh + ph) * kw + pw;
						size_t col_row_base = n * col_sN + row * col_sRow;
						for (size_t oh = 0; oh < h_out; oh++) {
							long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
							bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
							for (size_t ow = 0; ow < w_out; ow++) {
								long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
								size_t col_idx = oh * w_out + ow;
								float v = 0.0f;
								if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
									v = x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW];
								}
								col_data[col_row_base + col_idx] = v;
							}
						}
					}
				}
			}
		}
	});
	return col;
```

- [ ] **Step 6: Parallelize `col2im`**

Replace:
```cpp
	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < Cin; c++) {
			for (size_t ph = 0; ph < kh; ph++) {
				for (size_t pw = 0; pw < kw; pw++) {
					size_t row = (c * kh + ph) * kw + pw;
					size_t col_row_base = n * col_sN + row * col_sRow;
					for (size_t oh = 0; oh < h_out; oh++) {
						long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
						bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
						for (size_t ow = 0; ow < w_out; ow++) {
							long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
							if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
								size_t col_idx = oh * w_out + ow;
								x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW] +=
									col_data[col_row_base + col_idx];
							}
						}
					}
				}
			}
		}
	}
	return grad_x;
```
with:
```cpp
	parallel_for(N, [&](size_t n_start, size_t n_end) {
		for (size_t n = n_start; n < n_end; n++) {
			for (size_t c = 0; c < Cin; c++) {
				for (size_t ph = 0; ph < kh; ph++) {
					for (size_t pw = 0; pw < kw; pw++) {
						size_t row = (c * kh + ph) * kw + pw;
						size_t col_row_base = n * col_sN + row * col_sRow;
						for (size_t oh = 0; oh < h_out; oh++) {
							long ih = static_cast<long>(oh * stride + ph) - static_cast<long>(padding);
							bool ih_valid = ih >= 0 && ih < static_cast<long>(H);
							for (size_t ow = 0; ow < w_out; ow++) {
								long iw = static_cast<long>(ow * stride + pw) - static_cast<long>(padding);
								if (ih_valid && iw >= 0 && iw < static_cast<long>(W)) {
									size_t col_idx = oh * w_out + ow;
									x_data[n * sN + c * sC + static_cast<size_t>(ih) * sH + static_cast<size_t>(iw) * sW] +=
										col_data[col_row_base + col_idx];
								}
							}
						}
					}
				}
			}
		}
	});
	return grad_x;
```

- [ ] **Step 7: Parallelize `conv2d`'s bias-add loop**

Replace:
```cpp
	{
		size_t out_batch = out_biased.shape()[0];
		size_t hw = out_biased.shape()[2];
		float* out_data = out_biased.data().data();
		const float* bias_data = bias.data().data().data();
		for (size_t n = 0; n < out_batch; n++) {
			for (size_t c = 0; c < Cout; c++) {
				float bias_val = bias_data[c];
				float* row = out_data + (n * Cout + c) * hw;
				for (size_t i = 0; i < hw; i++) row[i] += bias_val;
			}
		}
	}
```
with:
```cpp
	{
		size_t out_batch = out_biased.shape()[0];
		size_t hw = out_biased.shape()[2];
		float* out_data = out_biased.data().data();
		const float* bias_data = bias.data().data().data();
		parallel_for(out_batch, [&](size_t n_start, size_t n_end) {
			for (size_t n = n_start; n < n_end; n++) {
				for (size_t c = 0; c < Cout; c++) {
					float bias_val = bias_data[c];
					float* row = out_data + (n * Cout + c) * hw;
					for (size_t i = 0; i < hw; i++) row[i] += bias_val;
				}
			}
		});
	}
```

- [ ] **Step 8: Parallelize `maxpool2d`'s forward loop**

Replace:
```cpp
	for (size_t n = 0; n < N; n++) {
		for (size_t c = 0; c < C; c++) {
			for (size_t oh = 0; oh < h_out; oh++) {
				for (size_t ow = 0; ow < w_out; ow++) {
					float best = -std::numeric_limits<float>::infinity();
					size_t best_flat = 0;
					for (size_t ph = 0; ph < kernel_size; ph++) {
						for (size_t pw = 0; pw < kernel_size; pw++) {
							size_t ih = oh * stride + ph;
							size_t iw = ow * stride + pw;
							size_t flat = ((n * C + c) * H + ih) * W + iw;
							float v = x_data[flat];
							if (v > best) {
								best = v;
								best_flat = flat;
							}
						}
					}
					size_t out_idx = ((n * C + c) * h_out + oh) * w_out + ow;
					result_data[out_idx] = best;
					argmax[out_idx] = best_flat;
				}
			}
		}
	}
```
with:
```cpp
	parallel_for(N, [&](size_t n_start, size_t n_end) {
		for (size_t n = n_start; n < n_end; n++) {
			for (size_t c = 0; c < C; c++) {
				for (size_t oh = 0; oh < h_out; oh++) {
					for (size_t ow = 0; ow < w_out; ow++) {
						float best = -std::numeric_limits<float>::infinity();
						size_t best_flat = 0;
						for (size_t ph = 0; ph < kernel_size; ph++) {
							for (size_t pw = 0; pw < kernel_size; pw++) {
								size_t ih = oh * stride + ph;
								size_t iw = ow * stride + pw;
								size_t flat = ((n * C + c) * H + ih) * W + iw;
								float v = x_data[flat];
								if (v > best) {
									best = v;
									best_flat = flat;
								}
							}
						}
						size_t out_idx = ((n * C + c) * h_out + oh) * w_out + ow;
						result_data[out_idx] = best;
						argmax[out_idx] = best_flat;
					}
				}
			}
		}
	});
```

- [ ] **Step 9: Build and run the full suite**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: PASS — full suite, including every existing small-`N` conv2d/maxpool2d test (serial fallback, unchanged results) and the two new above-threshold tests from Step 2.

- [ ] **Step 10: Commit**

```bash
git add src/autograd/ops.cpp tests/test_parallel.cpp
git commit -m "perf: parallelize conv2d/maxpool2d im2col-family loops"
```

---

### Task 9: README Scope update

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (documentation only).

- [ ] **Step 1: Update the Scope section**

In `README.md`, replace:
```
This project targets clarity and correctness over raw throughput: it uses single-threaded, cache-friendly loops rather than multi-threading or SIMD intrinsics. The tensor's dtype is fixed to `float`; a runtime-selectable dtype is a natural future extension but is not implemented here.
```
with:
```
This project targets clarity and correctness over raw throughput: hot loops (elementwise ops, matmul, axis reductions, and the conv2d/maxpool2d im2col-family loops) are split across a small stdlib-only thread pool (`nn::parallel_for`, see `include/nn/parallel.hpp`) rather than using SIMD intrinsics. The tensor's dtype is fixed to `float`; a runtime-selectable dtype is a natural future extension but is not implemented here.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update Scope section for basic parallelism"
```
