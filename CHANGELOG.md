# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-11-13

Performance release: the native build is faster than PyTorch 2.15 (CPU, eager) on five of the six bundled workloads and ties the sixth (see [docs/performance](docs/performance/README.md#v200-against-pytorch)). It contains one source-breaking change (`Tensor::data()` returns `nn::Storage`), hence the major version. Results are still bitwise identical for any thread count, but they are **not** bitwise equal to 1.3.0 (see Changed).

### Changed (breaking)

- `Tensor::data()` returns `nn::Storage` (`std::vector<float>` with a non-initialising allocator) instead of `std::vector<float>`. Code that binds the result to `std::vector<float>&`, or passes it where that type is required, must use `nn::Storage` or copy; `Storage` compares equal to a `std::vector<float>` with the same elements, `Tensor(shape, Storage)` was added next to the `std::vector<float>` constructor, and `Tensor::empty(shape)` allocates without filling. `Tensor(shape)` and `Tensor(shape, fill)` still fill.
- Numerical results change in the last bits: a GEMM with a small output and a long depth (`m * n <= 65536`, `k >= 512`) is summed in depth pieces, GELU uses a vectorised `tanh` (absolute error under 1.4e-7), and the convolution bias gradient is summed per sample and then in sample order. Training losses shift by a few ulp.
- A node's gradient is allocated on first accumulation and written instead of zero-filled and added to (`Value::grad_accum`).
- `Sequential` folds a `Conv2d` or `BatchNorm` followed by `ReLU` into one node (`Module::can_fuse_relu()` and `forward_relu()`, new virtuals with defaults).
- The library compiles with `-ffp-contract=off` (non-MSVC), because FMA contraction made fused loops differ between thread counts on `-march=native`.

### Added

- Ops: `linear(x, weight, bias)`, `attention(q, k, v, heads, key_padding_mask, causal)`, `conv2d_relu`, `batchnorm(..., fuse_relu)` (now one fused op with per-channel statistics), `layernorm` and `scaled_masked_softmax`; `softmax` has a fast path on the last axis. `Linear`, `MultiHeadAttention` (without attention dropout), `LayerNorm`, `Conv2d` and `BatchNorm` use them. Each is tested against the composition it replaces, for any thread count.
- `nn/kernels/vmath.hpp`: `exp_one`/`exp_inplace` and `tanh_one`/`tanh_inplace`, identical across AVX2, scalar, native and portable builds.
- `kernels::spin_microseconds()`, `kernels::set_spin_microseconds(us)` and the `NN_SPIN_US` variable: idle pool workers spin up to that long (default 100 µs, at most 10000, 0 disables) before sleeping.
- `Tensor::empty`, `Tensor::transpose_into`, `Value::grad_accum` and `Dropout::active()`.
- `apps/bench/bench_torch.py`: the same six workloads in PyTorch (CPU, eager), writing the same CSV with `config=torch`, and the comparison in `docs/performance/`.

### Performance

- GEMM: AVX2 microkernel with named accumulators (no stack spills), A packed once per block, depth-cut products for weight gradients, `matmul` over a batch runs one matrix per task, `matmul`/`matmul_bt` backward with a leading batch folds into one GEMM and accumulates in place.
- Convolution works one sample per task with a private column buffer that stays in cache and recomputes `im2col` in the backward; `col2im` accumulates straight into the gradient.
- Parallel `transpose`, `masked_fill`, `sum`/`max` over an axis, zero-fill of large tensors, and the SGD, Adam and AdamW update loops (fused into one pass). Max-pooling uses `uint32` indices. On glibc the library raises the `mmap` and trim thresholds with `mallopt`.
- Native build, 8 threads, step time before and after on this release (ms): `mlp_wide` 10.35 → 8.28, `mlp_adam` 11.35 → 8.81, `cnn_mnist` 1.19 → 0.59, `cnn_fer` 38.44 → 15.01, `cnn_large` 101.78 → 48.12, `transformer_tiny` 14.33 → 4.56; PyTorch 2.15: 8.24, 9.80, 1.29, 20.51, 72.66, 6.24.

### Documentation

- `docs/architecture.md` no longer says nothing is fused and describes the fused ops, the convolution path, storage and the pool; `docs/api.md` lists the new functions; the README speed table compares with PyTorch.

## [1.3.0] - 2026-11-12

Adds the pieces for a small decoder-only Transformer on character data. The public API of 1.2.0 is unchanged.

### Added

- Ops: `exp`, `log`, `sqrt`, `div`, `gelu` (tanh approximation), `masked_fill`, `slice`, `concat`, `split`, `embedding` and `sequence_cross_entropy` (mean over positions of a `{N, T, V}` logits tensor, with `ignore_index` and label smoothing). Each has a finite-difference gradient check.
- Layers: `Embedding`, `LayerNorm`, `PositionalEncoding` (sinusoidal buffer or learned table), `MultiHeadAttention` (optional causal mask and key-padding mask, attention dropout), `GELU`, `TransformerBlock` (pre-norm) and `Transformer` configured by `TransformerConfig`. All are composed from existing differentiable ops, so none defines a backward pass.
- `nn/data/text.hpp`: `CharVocab` (byte-level vocabulary) and `SequenceDataset` (sliding next-token windows).
- `train_transformer`: trains a small Transformer on the generated `reverse`, `copy` or `add` string tasks and reports exact-match accuracy of greedy generation on held-out samples.
- `transformer_tiny` workload in `apps/bench` (vocabulary 64, width 64, 4 heads, 2 layers, batch 16 x 64 tokens, AdamW).

## [1.2.0] - 2026-11-05

### Added

- `SGD` takes `nesterov` and `weight_decay` (L2 added to the gradient before the velocity update); `ReduceLROnPlateau` with `PlateauMode` in `nn/optim/plateau.hpp`.
- Transforms `RandomCrop`, `RandomAffine` and `RandomErasing`, and the test-time views `ten_crop` and `ten_crop_batch` (four corners and the centre, plus their mirrors).
- `nn/metrics.hpp`: `confusion_matrix`, `accuracy_score`, `precision_recall_f1`, `precision_score`, `recall_score`, `f1_score` (micro, macro and weighted averages, `zero_division`), `classification_report` laid out like scikit-learn's, and the helpers `argmax_rows` and `to_labels`.
- `DataLoader` constructor with per-sample weights (draws with replacement, like PyTorch's `WeightedRandomSampler`, portable across platforms), `balanced_sample_weights`, and `Fer2013Dataset::labels()`.
- `train_fer2013` and `train_mnist` print a classification report; `train_fer2013` also keeps the best epoch by validation macro-F1 and scores it on the PrivateTest rows. All three apps write stdout unbuffered, so progress shows up immediately even when redirected to a file.

### Changed

- `train_fer2013` defaults to 100 epochs and a larger network (32-64-128-256 channels with dropout and a dense head), AdamW weight decay 5e-2, label smoothing 0.1 and class balancing by sampling.

## [1.1.0] - 2026-11-05

Adds what is needed to train a small CNN with regularisation on real image data (FER-2013). The public API of 1.0.0 is unchanged; new virtuals on `Module` and `Optimizer` have defaults, so existing subclasses compile as before.

### Added

- Seeding: `nn/random.hpp` with `manual_seed`, `is_seeded` and `derive_generator`. `Tensor::random`, `DataLoader`, `Dropout` and the random transforms each draw from their own stream, so one seed makes a run reproducible for a fixed thread count.
- Modes and buffers: `Module::train(bool)`, `eval()`, `is_training()` and `named_buffers()`; `Sequential` forwards them to its layers.
- Layers: `Dropout`, `BatchNorm1d` and `BatchNorm2d` (running statistics are buffers), `AvgPool2d` and `GlobalAvgPool2d`; `Conv2d` gains a `{kh, kw}` kernel, `dilation` and `groups`.
- Ops: `sum(x, axis, keepdim)`, `mean(x, axis, keepdim)`, `rsqrt`, `avgpool2d`, a general `conv2d(..., dilation, groups)`, and a `cross_entropy_loss` overload with per-class weights and label smoothing. `Conv2d` with `groups == 1` and `dilation == 1` keeps the 1.0.0 code path.
- Optimization: `AdamW`; the schedulers `StepLR`, `CosineAnnealingLR` and `LinearWarmup` (`LRScheduler::step`, `set_step`); `clip_grad_norm` and `clip_grad_value`; `Optimizer::lr()` and `set_lr()`.
- Data: `TensorDataset`, `Subset` and `random_split`, the transforms `Compose`, `Normalize`, `RandomHorizontalFlip`, `RandomShift` with `TransformedDataset` and `compute_mean_std`, `Fer2013Dataset` (FER-2013 CSV loader), and a `DataLoader` constructor with `drop_last`.
- Apps: `train_fer2013` (a Conv-BatchNorm-ReLU network on the FER-2013 CSV, which is downloaded separately) and a `cnn_fer` workload in `bench`. `train_fer2013` is installed with the other apps.
- `docs/performance/v1.1.0-results.md`: the performance guard against 1.0.0 (no regression) and a profile of `cnn_fer`.

### Changed

- The `DataLoader` shuffle order differs from 1.0.0. The shuffle is now a portable Fisher-Yates, so the same seed gives the same order on every platform.
- Checkpoints may now contain module buffers after the parameters. Files written by 1.0.0 still load; a file without buffers cannot be loaded into a model that has buffers (`std::runtime_error`). `load_state` of `SGD` and `Adam` now also accepts a snapshot without `lr` and keeps the current learning rate, where 1.0.0 rejected it as missing.
- Elementwise ops with broadcasting (add, sub, mul, div) are faster and run in parallel; results are unchanged.
- A non-leaf `Value::grad()` is now an empty placeholder (shape `{0}`) until `backward()` reaches it; in 1.0.x it was a zero tensor shaped like the data. Use `has_grad()` to test whether the gradient has been computed.

### Fixed

- `Value::backward()` ordered nodes wrongly when a node was reached by paths of different depth, which gave wrong gradients for shared subgraphs (a value used by several consumers). Version 1.0.1 has the same fix, and 1.1.0 contains it.

### Notes

- `Tensor::random` values come from `std::uniform_real_distribution`, so a seed reproduces initialisation for one standard library, not across libstdc++, libc++ and MSVC.
- `Dropout`'s random stream is not saved in checkpoints, so the masks of a resumed run differ from an uninterrupted one.
- Schedulers are not saved in checkpoints; restore one with `set_step`.
- MSVC on x86 still uses the unpacked `sgemm`.
- `Module` gained `train`, `eval`, `is_training`, `training_` and `named_buffers`; user subclasses that already define members with these names will hide or override them. `Optimizer::lr()` and `set_lr()` throw `std::logic_error` for custom optimizers that do not override them, so a scheduler needs an optimizer that implements them.
- Elementwise ops now run in parallel and `cnn_fer` goes from 716 ms to 165 ms per step at 8 threads (`docs/performance/v1.1.0-results.md`); the remaining serial work is reductions, memset and malloc.

## [1.0.1] - 2026-11-05

### Fixed

- `Value::backward()` computed wrong gradients when a node was consumed along paths of different depth (for example `c` feeding both `mul(c, mul(c, c))` and the final op directly). The traversal ordered such a node before some of its consumers, so it passed its gradient on before all of it had arrived. This affects any model with shared subgraphs, such as residual connections. Gradients of models without such sharing, including the bundled MLP and CNN examples, were correct. Upgrading is recommended for anyone training models with shared subgraphs.

## [1.0.0] - 2026-11-05

First release.

### Added

- A from-scratch C++17 neural network library with no external math or ML dependencies: an N-dimensional `float` `Tensor`, a reverse-mode autograd engine (`Value` and differentiable ops), `nn` layers (`Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, activations, `Sequential`), losses, `SGD` and `Adam` optimizers, checkpoint save/load, and a `Dataset`/`DataLoader` pair.
- Example apps: `xor_demo`, `train_mnist` (a small CNN on MNIST; the data files are not included) and the `bench` harness. A native-build run of `train_mnist` reached 98.58% test accuracy in 17.8 s (3 epochs, `docs/performance/phase3-results.md`).
- A doctest suite of 199 tests, run through `ctest`.
- `cmake --install` rules for the static library `nn_core`, its headers and the apps, plus a CMake package config: `find_package(nn_cpp)` and link `nn::core`.
- `Value::has_grad()`.
- A GitHub Actions workflow that builds, tests and packages Linux (x86-64, ARM64), Windows (x86-64, ARM64) and macOS (universal) and attaches the packages to the GitHub release of each `vX.Y.Z` tag.

### Performance

Measured on an Intel i5-12450HX with `NN_NUM_THREADS=8 taskset -c 0-7`; the per-phase reports in `docs/` have the protocol and raw numbers.

- Phase 0: benchmark harness (`apps/bench`), baseline CSVs and perf profiles, in `docs/performance/phase0-results.md`.
- Phase 1: cache-blocked, packed `sgemm` with AVX2/FMA and SSE2 microkernels (an unpacked loop on other targets) and transpose-free matmul paths. Native `cnn_mnist` 2.17x and `mlp_wide` 3.52x faster; portable `mlp_wide` 2.31x.
- Phase 2: a fork/join thread pool and a parallel `sgemm`. Native `mlp_wide` 1.40x and `cnn_mnist` 1.36x faster than Phase 1. The thread count defaults to `std::thread::hardware_concurrency()` and can be set with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`; results do not depend on it.
- Phase 3: fewer copies (results moved into graph nodes, lazy gradient allocation, broadcast fast paths) and parallel elementwise ops, `relu`, im2col, col2im and max-pooling. Native `cnn_mnist` 2.19x and `mlp_wide` 2.26x faster than Phase 2.
- Phase 4 follow-ups: a deterministic gradient check in place of a flaky random one, and parallel packing of the B matrix in `sgemm` (`docs/performance/followups-results.md`).
- Overall, from the Phase 0 baseline to the end of Phase 4 (training-step time, README table): native `cnn_mnist` 17.63 ms to 2.64 ms (6.7x), `mlp_wide` 173.48 ms to 14.18 ms (12.2x), `cnn_large` 1030.92 ms to 250.22 ms (4.1x), `mlp_adam` 171.24 ms to 16.39 ms (10.4x). Portable builds gain 3.9x to 8.1x.

### Changed

- `Value::grad()` on a non-leaf node is an empty placeholder (shape `{0}`) until `backward()` reaches the node. Leaves are unchanged. Use `has_grad()` to check whether a node's gradient has been computed.

### Portability

- Supported compilers and targets are GCC, Clang and MSVC on x86-64 and ARM64, built by CI on Linux, Windows and macOS. Only the Linux x86-64 builds have been run during development.
- The GEMM microkernel is picked at compile time. MSVC defines neither `__SSE2__` nor `__AVX2__`/`__FMA__`, so MSVC builds use the unpacked `sgemm`. It gives correct results but is slower than the GCC/Clang x86-64 builds, and there are no MSVC intrinsic paths yet.
- The GEMM packing buffers use C++17 aligned `operator new` instead of `std::aligned_alloc`, which MSVC lacks.
- `NN_NATIVE=ON` no longer passes `-march=native` to MSVC.
