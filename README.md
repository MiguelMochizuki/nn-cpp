# nn-cpp

[![Build and Release](https://github.com/MiguelMochizuki/nn-cpp/actions/workflows/build-and-release.yml/badge.svg)](https://github.com/MiguelMochizuki/nn-cpp/actions/workflows/build-and-release.yml)
[![Release](https://img.shields.io/github/v/release/MiguelMochizuki/nn-cpp)](https://github.com/MiguelMochizuki/nn-cpp/releases)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A neural network library written from scratch in C++17. No dependencies, no BLAS, no framework.**
Tensors, reverse-mode autograd, layers, optimizers and a data pipeline, in readable code, with a GEMM fast enough to train a CNN on MNIST to **98.5%** in 18 seconds on a laptop CPU.

```cpp
Sequential model;
model.add(std::make_unique<Linear>(2, 8));
model.add(std::make_unique<ReLU>());
model.add(std::make_unique<Linear>(8, 1));

SGD optimizer(model.parameters(), 0.1f);
for (int epoch = 0; epoch < 2000; epoch++) {
    Value loss = mse_loss(model.forward(x), y);
    optimizer.zero_grad();
    loss.backward();           // gradients come from the ops that ran, not from the layers
    optimizer.step();
}
```

## Highlights

- **Autograd is the only place gradients are written.** A layer implements `forward()` out of differentiable ops and gets its backward pass for free. Each op has a finite-difference gradient check.
- **Fast where it counts.** A BLIS-style blocked GEMM (AVX2/FMA, SSE2 or portable) on a deterministic thread pool. Results are bitwise identical for any thread count.
- **Built to train real models.** BatchNorm, Dropout, AdamW, LR schedulers, gradient clipping, seeding, train/eval modes, checkpoints to resume training, and a `Dataset`/`DataLoader` pipeline with transforms.
- **Small and portable.** One static library plus headers, packaged for Linux, macOS and Windows on x86-64 and ARM64. Install it and `find_package(nn_cpp)`.

## Speed

Training-step time against the first measured baseline, on an Intel i5-12450HX (8 threads, pinned). Native is `-DNN_NATIVE=ON`; portable is the default build.

| workload | native | portable |
|---|---|---|
| `cnn_mnist` (small CNN) | 17.63 → 2.64 ms (**6.7x**) | 20.39 → 3.88 ms (5.3x) |
| `mlp_wide` (wide MLP, large GEMMs) | 173.48 → 14.18 ms (**12.2x**) | 215.13 → 26.56 ms (8.1x) |
| `mlp_adam` (MLP with Adam) | 171.24 → 16.39 ms (**10.4x**) | 228.33 → 29.01 ms (7.9x) |
| `cnn_large` | 1030.92 → 250.22 ms (**4.1x**) | 1180.54 → 304.10 ms (3.9x) |

On real MNIST (3 epochs, 8 threads) `train_mnist` reaches 98.58% test accuracy in 17.8 s native and 98.43% in 46.3 s portable; the unoptimised baseline needed about 1:42 on one thread. Method, profiles and raw CSVs are in [docs/performance](docs/performance/README.md).

## What is inside

| | |
|---|---|
| **Tensor** | N-D `float` arrays, broadcasting arithmetic, axis reductions, batched matmul |
| **Layers** | `Linear`, `Conv2d` (stride, padding, dilation, groups), `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `BatchNorm1d/2d`, `Dropout`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Sequential` |
| **Losses** | `cross_entropy_loss` (class weights, label smoothing), `mse_loss` |
| **Optimization** | `SGD` (momentum), `Adam`, `AdamW`, `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, `clip_grad_norm`, `clip_grad_value` |
| **Data** | `Dataset`, `DataLoader`, `TensorDataset`, `random_split`, `Normalize`, `RandomHorizontalFlip`, `RandomShift`, FER-2013 CSV loader |
| **Reproducibility** | `manual_seed`, an independent stream per component, portable shuffling |
| **Persistence** | `save_checkpoint` / `load_checkpoint`: weights, buffers and optimizer state, validated before anything is modified |

Full signatures: [API reference](docs/api.md).

## Try it

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure

./build/apps/xor_demo                              # a 2-layer MLP learns XOR
./build/apps/train_mnist <dir-with-4-idx-files>    # CNN on MNIST (files not downloaded)
./build/apps/train_fer2013 <fer2013.csv>           # Conv-BN-ReLU network on FER-2013 (not downloaded)
```

`-DNN_NATIVE=ON` builds for the host CPU (AVX2/FMA microkernel where available); `-DNN_BUILD_TESTS=OFF` and `-DNN_BUILD_APPS=OFF` skip those targets. The thread count defaults to the hardware's and is set with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`. x86 MSVC builds use the unpacked `sgemm`: correct, but slower than GCC/Clang.

## Use it in your project

Each [release](https://github.com/MiguelMochizuki/nn-cpp/releases) ships the static library, headers and apps as a CMake install prefix:

```cmake
find_package(nn_cpp 1.1 REQUIRED)    # CMAKE_PREFIX_PATH = the unpacked package
target_link_libraries(my_app PRIVATE nn::core)
```

A regularised CNN with a schedule, clipping and a validation split takes about 30 lines; see the [guide](docs/guide.md) for that, MNIST, FER-2013, checkpoints and writing your own `Dataset`.

## Documentation

| | |
|---|---|
| [Guide](docs/guide.md) | worked examples: regularisation and schedules, MNIST, FER-2013, resuming, custom data |
| [API reference](docs/api.md) | one row per header |
| [Architecture](docs/architecture.md) | how autograd, convolution, the GEMM and the thread pool fit together; checkpoint format and limits |
| [Performance](docs/performance/README.md) | benchmark method, per-phase reports, raw data |
| [CHANGELOG](CHANGELOG.md) | what changed in each version (Semantic Versioning) |
| [CONTRIBUTING](CONTRIBUTING.md) | build, test and code-documentation conventions |
| [Releasing](docs/releasing.md) | how a release is cut |

## Project layout

```
include/nn/   public headers: tensor, autograd, nn, optim, data, kernels
src/          implementations, mirroring include/nn
apps/         xor_demo, mnist (train_mnist), fer2013 (train_fer2013), bench (harness)
tests/        doctest suite, gradient checks, a naive GEMM oracle
docs/         guide, API, architecture, performance reports
```

## License

MIT, see [LICENSE](LICENSE).
