# Math-readable operators, allocation-churn reduction, basic parallelism

Date: 2026-09-24
Status: approved, pending implementation plan

## Goal

Three related improvements to the existing `nn-cpp` library, done as one phased
change since they touch the same call sites:

1. Make math read closer to the underlying formulas by adding operator
   overloads (`+ - * /`, unary `-`, `+= -= *= /=`) on `Tensor` and `Value`,
   as thin wrappers over the existing free functions.
2. Cut allocation/copy churn in the hot paths that run every training step:
   gradient accumulation, backward-pass grad zeroing, and duplicate tensor
   copies taken purely to satisfy `Value::node`'s no-self-capture rule.
3. Add basic CPU parallelism (a hand-rolled, stdlib-only thread pool) over
   the loops that are naturally embarrassingly parallel: elementwise ops,
   matmul, axis reductions, and the conv2d/maxpool2d im2col-family loops.

## Non-goals

- SIMD intrinsics — explicitly out of scope per the existing README "Scope"
  section, staying that way.
- A configurable/injectable thread pool object threaded through public APIs
  — a single lazy-init global pool is enough for this library's use.
- Copy-on-write / arena / pooled Tensor storage — would remove the two
  unavoidable copies inherent to `Value::node`'s cycle-avoidance rule
  (graph-owned data + backward-closure snapshot), but is a storage-model
  redesign, not "cut churn."
- `operator()(i, j, ...)` indexing sugar or an expression-template /
  lazy-evaluated math DSL — operator overloads only, per design discussion.
- Parallelizing `DataLoader` batch preparation — not a hot enough path to
  justify it.
- Any change to the broadcast (non-fast-path) branch of `elementwise()` —
  low-value, more complex to parallelize safely, left serial.

## 1. Operator overloads

### `Tensor` (`include/nn/tensor.hpp`, `src/tensor.cpp`)

Add as free functions alongside the existing `add`/`sub`/`mul`/`div`/`scale`:

```cpp
Tensor operator+(const Tensor& a, const Tensor& b);   // = add(a, b)
Tensor operator-(const Tensor& a, const Tensor& b);   // = sub(a, b)
Tensor operator*(const Tensor& a, const Tensor& b);   // = mul(a, b), elementwise
Tensor operator/(const Tensor& a, const Tensor& b);   // = div(a, b)
Tensor operator*(const Tensor& a, float scalar);      // = scale(a, scalar)
Tensor operator*(float scalar, const Tensor& a);      // = scale(a, scalar)
Tensor operator-(const Tensor& a);                    // = scale(a, -1.0f)
```

`matmul` is deliberately **not** overloaded onto `operator*` — it stays a
named call (`matmul(a, b)` / member-style `a.matmul(b)` is not being added
either, free function only, matching existing convention) so batched matrix
multiply stays visually distinct from elementwise multiply, matching the
existing `mul` vs `matmul` naming.

In-place compound assignment, implemented as real in-place loops (not
allocate-and-reassign) — this is what backs the memory-churn work in
section 2:

```cpp
Tensor& Tensor::operator+=(const Tensor& other);
Tensor& Tensor::operator-=(const Tensor& other);
Tensor& Tensor::operator*=(const Tensor& other);
Tensor& Tensor::operator/=(const Tensor& other);
Tensor& Tensor::operator*=(float scalar);
```

Compound assignment requires `other.shape() == this->shape()` exactly
(no broadcasting into a differently-shaped LHS) — throws `TensorShapeError`
otherwise. Every call site this design touches only ever accumulates into a
tensor already the target shape, so this restriction costs nothing in
practice while keeping the in-place loop simple (single pass, no
broadcast-offset computation).

### `Value` (`include/nn/autograd/ops.hpp`, `src/autograd/ops.cpp`)

Same surface, wrapping the existing `add`/`sub`/`mul`/`scale` free functions
for `Value`:

```cpp
Value operator+(const Value& a, const Value& b);
Value operator-(const Value& a, const Value& b);
Value operator*(const Value& a, const Value& b);   // elementwise
Value operator*(const Value& a, float scalar);
Value operator*(float scalar, const Value& a);
Value operator-(const Value& a);
```

No compound assignment on `Value` — graph nodes are immutable once
constructed; only their `.grad()` `Tensor&` is mutated in place (via
`Tensor::operator+=`, section 2), never `.data()`.

No `operator/` for `Value` — there's no existing `div(Value, Value)` free
function to wrap (only `Tensor` has `div`), and nothing in this design needs
one; skipped rather than adding new differentiable-op surface unasked.

Example effect: `add(matmul(a, b), c)` becomes `matmul(a, b) + c` (`matmul`
stays a free function for `Value` too, no method form added).

## 2. Allocation/copy churn reduction

Every change below is a mechanical rewrite of an existing call site to use
the new operators; no algorithmic behavior changes.

**Gradient accumulation** — every backward closure in `src/autograd/ops.cpp`
currently does `X.grad() = add(X.grad(), g);` (full new-Tensor allocation,
every parent, every op, every backward call). Rewrite all such sites
(`add`, `sub`, `mul`, `scale`, `sum`, `transpose`, `reshape`, `matmul`,
`relu`, `sigmoid`, `tanh`, `softmax`, `conv2d` ×3, `maxpool2d`,
`cross_entropy_loss` — roughly 15 sites) to `X.grad() += g;`.

**Backward-pass grad zeroing** (`src/autograd/value.cpp`) —
`Value::backward()`'s reset loop does
`v.impl()->grad = Tensor(v.impl()->data.shape(), 0.0f);` for every non-leaf
node, and `Value::zero_grad()` does the same for one node. Both reallocate
even though the grad tensor is already the correct shape (sized once in
`NodeImpl`'s constructor and never resized). Add a small in-place
`Tensor::zero_()` method (`std::fill(data_.begin(), data_.end(), 0.0f)`) and
use it in both places instead of constructing a new `Tensor`.

**Redundant `saved` copies** — `sigmoid`, `tanh`, `softmax` (all in
`src/autograd/ops.cpp`) each compute `result`, then do
`Tensor saved = result;` purely to have something to capture in the
backward closure, then pass `result` into `Value::node(...)` by value —
three copies of the same buffer where two are unavoidable. `conv2d` has the
identical pattern with `Tensor saved_col = col;`. Drop the `saved`/
`saved_col` local entirely; capture `result`/`col` directly in the closure
(`[x, result](...)`), letting the closure-capture and the
pass-by-value-into-`node()` each make their own copy from the same source —
order-independent since neither is a move, both just read `result`. Cuts
one full-tensor-sized allocation per call to each of these four ops.

**`SGD::step()`** (`src/optim/sgd.cpp`) — currently:
```cpp
Tensor update = params_[i].grad();
if (momentum_ > 0.0f) {
	velocity_[i] = add(scale(velocity_[i], momentum_), params_[i].grad());
	update = velocity_[i];
}
params_[i].data() = sub(params_[i].data(), scale(update, lr_));
```
Rewritten with in-place operators:
```cpp
if (momentum_ > 0.0f) {
	velocity_[i] *= momentum_;
	velocity_[i] += params_[i].grad();
	params_[i].data() -= velocity_[i] * lr_;
} else {
	params_[i].data() -= params_[i].grad() * lr_;
}
```
Cuts from ~4 allocations to 1 (the unavoidable `lr * grad`/`lr * velocity`
scaled temporary — no fused axpy helper is being added; that's a further
optimization not required here) per parameter per step.

## 3. Basic parallelism

New files: `include/nn/parallel.hpp`, `src/parallel.cpp`. Added to
`nn_core`'s source list in the top-level `CMakeLists.txt`.

```cpp
namespace nn {

// Splits [0, n) into one contiguous chunk per worker thread and runs
// fn(start, end) for each chunk on a shared, lazily-initialized,
// process-wide thread pool sized to hardware_concurrency(). Blocks until
// every chunk completes. Falls back to a single synchronous call
// fn(0, n) when n is below an internal threshold, so small tensors (unit
// tests, the XOR demo) never pay thread dispatch overhead.
// ponytail: fixed threshold, not adaptive to actual measured overhead —
// revisit if profiling shows the constant wrong for a given workload.
void parallel_for(size_t n, const std::function<void(size_t start, size_t end)>& fn);

}  // namespace nn
```

Implementation: `std::thread`/`std::atomic`/`std::condition_variable` only
(no new dependency). Threshold constant (e.g. `8192` elements) lives as an
internal constant in `parallel.cpp`, not exposed as configuration.

Applied only where each worker's writes are provably disjoint from every
other worker's — no atomics, no locks, no false-sharing analysis needed
anywhere in this design:

- `elementwise()` fast path in `src/tensor.cpp` (equal-shape `add/sub/mul/div`)
  — split the flat `n`-element loop across workers; each writes disjoint
  output indices.
- `matmul()` in `src/tensor.cpp` — split by output row `i` within each
  batch; each worker owns a disjoint set of output rows.
- `Tensor::sum/mean/max(axis)` in `src/tensor.cpp` — split by `outer`
  index; each outer index's destination slice (`dst_base = outer *
  inner_size`, width `inner_size`) is disjoint by construction, so this is
  safe without any reduction/merge step.
- `im2col`/`col2im` and the bias-add loop inside `conv2d` (all in
  `src/autograd/ops.cpp`) — split by batch index `n`.
- `maxpool2d` forward loop (`src/autograd/ops.cpp`) — split by batch index
  `n` (or `n*C`, whichever divides more evenly — implementation detail for
  the plan).

Explicitly left serial: the broadcast (non-fast-path) branch of
`elementwise()`. `col2im` scatters into `grad_x` with `+=` per input pixel
touched by multiple `(ph, pw)` kernel positions, but always within the same
batch index `n` — never across different `n` — so splitting by `n` is still
disjoint per worker and safe to parallelize the same way as `im2col`; the
implementation plan should re-verify this invariant against the code before
relying on it, since it's less immediately obvious than the other cases.

## 4. Testing

Existing unit tests use small tensors (well under the parallel threshold),
so they stay on the serial fallback path and remain bit-identical — no
tolerance risk from floating-point reordering. Existing `gradcheck`
tolerances (`5e-2` default, `0.1`–`0.15` explicit in several tests) already
have ample margin for the reordering that *does* happen on larger,
above-threshold tensors, since `parallel_for` always assigns the same
contiguous chunk boundaries for a given `n` and thread count — results are
deterministic run-to-run, just not bit-identical to the pure-serial sum
order.

Add `tests/test_parallel.cpp`, added to `tests/CMakeLists.txt`'s
`nn_tests` executable:

- `parallel_for` coverage check: for a small `n`, record which indices each
  chunk claims (e.g. into a `std::vector<std::atomic<int>>` counter) and
  assert every index was covered exactly once.
- One parallelized op (large `matmul` or elementwise `add`, sized above the
  threshold) checked against a manually-computed serial reference within
  existing tolerance conventions, run at both a below-threshold and an
  above-threshold size to exercise both code paths.

## Files touched

- `include/nn/tensor.hpp`, `src/tensor.cpp` — operators, `zero_()`, parallel
  fast-path/matmul/reduction loops.
- `include/nn/autograd/ops.hpp`, `src/autograd/ops.cpp` — `Value` operators,
  `+=`-based grad accumulation, dropped `saved` copies, parallel
  im2col/col2im/maxpool2d loops.
- `src/autograd/value.cpp` — in-place grad zeroing in `backward()` and
  `zero_grad()`.
- `src/optim/sgd.cpp` — in-place update.
- `include/nn/parallel.hpp`, `src/parallel.cpp` — new.
- `CMakeLists.txt` — add `src/parallel.cpp` to `nn_core`.
- `tests/test_parallel.cpp`, `tests/CMakeLists.txt` — new test file wired in.
- `README.md` — "Scope" section currently states "single-threaded,
  cache-friendly loops rather than multi-threading" — update to reflect
  the new basic parallelism, keep the "no SIMD" part.
