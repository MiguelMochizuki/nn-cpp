# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-11-05

### Fixed

- `Value::backward()` computed wrong gradients when a node was consumed along paths of different depth (for example `c` feeding both `mul(c, mul(c, c))` and the final op directly). The traversal ordered such a node before some of its consumers, so it passed its gradient on before all of it had arrived. This affects any model with shared subgraphs, such as residual connections. Gradients of models without such sharing, including the bundled MLP and CNN examples, were correct. Upgrading is recommended for anyone training models with shared subgraphs.

## [1.0.0] - 2026-11-05

First release.

### Added

- A from-scratch C++17 neural network library with no external math or ML dependencies: an N-dimensional `float` `Tensor`, a reverse-mode autograd engine (`Value` and differentiable ops), `nn` layers (`Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, activations, `Sequential`), losses, `SGD` and `Adam` optimizers, checkpoint save/load, and a `Dataset`/`DataLoader` pair.
- Example apps: `xor_demo`, `train_mnist` (a small CNN on MNIST; the data files are not included) and the `bench` harness. A native-build run of `train_mnist` reached 98.58% test accuracy in 17.8 s (3 epochs, `docs/phase3-results.md`).
- A doctest suite of 199 tests, run through `ctest`.
- `cmake --install` rules for the static library `nn_core`, its headers and the apps, plus a CMake package config: `find_package(nn_cpp)` and link `nn::core`.
- `Value::has_grad()`.
- A GitHub Actions workflow that builds, tests and packages Linux (x86-64, ARM64), Windows (x86-64, ARM64) and macOS (universal) and attaches the packages to the GitHub release of each `vX.Y.Z` tag.

### Performance

Measured on an Intel i5-12450HX with `NN_NUM_THREADS=8 taskset -c 0-7`; the per-phase reports in `docs/` have the protocol and raw numbers.

- Phase 0: benchmark harness (`apps/bench`), baseline CSVs and perf profiles, in `docs/phase0-results.md`.
- Phase 1: cache-blocked, packed `sgemm` with AVX2/FMA and SSE2 microkernels (an unpacked loop on other targets) and transpose-free matmul paths. Native `cnn_mnist` 2.17x and `mlp_wide` 3.52x faster; portable `mlp_wide` 2.31x.
- Phase 2: a fork/join thread pool and a parallel `sgemm`. Native `mlp_wide` 1.40x and `cnn_mnist` 1.36x faster than Phase 1. The thread count defaults to `std::thread::hardware_concurrency()` and can be set with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`; results do not depend on it.
- Phase 3: fewer copies (results moved into graph nodes, lazy gradient allocation, broadcast fast paths) and parallel elementwise ops, `relu`, im2col, col2im and max-pooling. Native `cnn_mnist` 2.19x and `mlp_wide` 2.26x faster than Phase 2.
- Phase 4 follow-ups: a deterministic gradient check in place of a flaky random one, and parallel packing of the B matrix in `sgemm` (`docs/followups-results.md`).
- Overall, from the Phase 0 baseline to the end of Phase 4 (training-step time, README table): native `cnn_mnist` 17.63 ms to 2.64 ms (6.7x), `mlp_wide` 173.48 ms to 14.18 ms (12.2x), `cnn_large` 1030.92 ms to 250.22 ms (4.1x), `mlp_adam` 171.24 ms to 16.39 ms (10.4x). Portable builds gain 3.9x to 8.1x.

### Changed

- `Value::grad()` on a non-leaf node is an empty placeholder (shape `{0}`) until `backward()` reaches the node. Leaves are unchanged. Use `has_grad()` to check whether a node's gradient has been computed.

### Portability

- Supported compilers and targets are GCC, Clang and MSVC on x86-64 and ARM64, built by CI on Linux, Windows and macOS. Only the Linux x86-64 builds have been run during development.
- The GEMM microkernel is picked at compile time. MSVC defines neither `__SSE2__` nor `__AVX2__`/`__FMA__`, so MSVC builds use the unpacked `sgemm`. It gives correct results but is slower than the GCC/Clang x86-64 builds, and there are no MSVC intrinsic paths yet.
- The GEMM packing buffers use C++17 aligned `operator new` instead of `std::aligned_alloc`, which MSVC lacks.
- `NN_NATIVE=ON` no longer passes `-march=native` to MSVC.
