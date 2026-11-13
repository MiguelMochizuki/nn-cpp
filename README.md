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
- **Fast where it counts.** A BLIS-style blocked GEMM (AVX2/FMA, SSE2 or portable) on a deterministic thread pool, with fused `linear`, `attention`, `conv2d_relu`, `batchnorm` and `layernorm` ops, and faster than PyTorch's CPU eager mode on most of the bundled workloads (native build). Results are bitwise identical for any thread count.
- **Built to train real models.** BatchNorm, Dropout, AdamW, LR schedulers, gradient clipping, seeding, train/eval modes, checkpoints to resume training, and a `Dataset`/`DataLoader` pipeline with transforms.
- **Small and portable.** One static library plus headers, packaged for Linux, macOS and Windows on x86-64 and ARM64. Install it and `find_package(nn_cpp)`.

## Speed

Training-step time on an Intel i5-12450HX (8 threads, pinned, on mains power). Native is `-DNN_NATIVE=ON`; portable is the default build. "Baseline" is the first measured version of this library.

| workload | native | portable |
|---|---|---|
| `cnn_mnist` (small CNN) | 17.63 → 0.59 ms (**29.9x**) | 20.39 → 1.10 ms (18.5x) |
| `mlp_wide` (wide MLP, large GEMMs) | 173.48 → 8.28 ms (**20.9x**) | 215.13 → 21.15 ms (10.2x) |
| `mlp_adam` (MLP with Adam) | 171.24 → 8.81 ms (**19.4x**) | 228.33 → 22.22 ms (10.3x) |
| `cnn_large` | 1030.92 → 48.12 ms (**21.4x**) | 1180.54 → 85.95 ms (13.7x) |

Against PyTorch 2.15 (CPU, eager, same six workloads, same machine and protocol; below 1.00 means nn-cpp is faster):

| workload | PyTorch | native | portable |
|---|---|---|---|
| `mlp_wide` | 8.24 ms | 8.28 ms (1.00x, a tie) | 21.15 ms (2.57x) |
| `mlp_adam` | 9.80 ms | 8.81 ms (**0.90x**) | 22.22 ms (2.27x) |
| `cnn_mnist` | 1.29 ms | 0.59 ms (**0.46x**) | 1.10 ms (**0.85x**) |
| `cnn_fer` | 20.51 ms | 15.01 ms (**0.73x**) | 24.61 ms (1.20x) |
| `cnn_large` | 72.66 ms | 48.12 ms (**0.66x**) | 85.95 ms (1.18x) |
| `transformer_tiny` | 6.24 ms | 4.56 ms (**0.73x**) | 8.56 ms (1.37x) |

The native build wins five of six and ties the sixth; the portable build, which has no AVX2 microkernel, wins only `cnn_mnist`. PyTorch runs on MKL and oneDNN, and with 4 threads instead of 8 PyTorch reaches 7.59 ms on `mlp_wide`, so that tie is within the noise. The comparison is one CPU and one model of each workload, not a general claim. On real MNIST (3 epochs, measured at v1.3.0) `train_mnist` reached 98.58% test accuracy in 17.8 s native and 98.43% in 46.3 s portable. Method, profiles and raw CSVs are in [docs/performance](docs/performance/README.md).

## What is inside

| | |
|---|---|
| **Tensor** | N-D `float` arrays, broadcasting arithmetic, axis reductions, batched matmul |
| **Layers** | `Linear`, `Embedding`, `LayerNorm`, `PositionalEncoding`, `MultiHeadAttention`, `TransformerBlock`, `Transformer`, `Conv2d` (stride, padding, dilation, groups), `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `BatchNorm1d/2d`, `Dropout`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`, `Sequential` |
| **Losses** | `cross_entropy_loss` (class weights, label smoothing), `sequence_cross_entropy` (ignore index), `mse_loss` |
| **Optimization** | `SGD` (momentum, Nesterov, weight decay), `Adam`, `AdamW`, `StepLR`, `CosineAnnealingLR`, `LinearWarmup`, `ReduceLROnPlateau`, `clip_grad_norm`, `clip_grad_value` |
| **Data** | `Dataset`, `CharVocab`, `SequenceDataset`, `DataLoader`, `TensorDataset`, `random_split`, `Normalize`, `RandomHorizontalFlip`, `RandomShift`, `RandomCrop`, `RandomAffine`, `RandomErasing`, `ten_crop`, `ten_crop_batch`, weighted-sampling `DataLoader` with `balanced_sample_weights`, FER-2013 CSV loader |
| **Metrics** | `confusion_matrix`, `precision_score` / `recall_score` / `f1_score` (micro, macro, weighted), `classification_report` |
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
./build/apps/train_transformer --task copy         # a small Transformer learns a string task (no data needed)
```

`-DNN_NATIVE=ON` builds for the host CPU (AVX2/FMA microkernel where available); `-DNN_BUILD_TESTS=OFF` and `-DNN_BUILD_APPS=OFF` skip those targets. The thread count defaults to the hardware's and is set with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`. x86 MSVC builds use the unpacked `sgemm`: correct, but slower than GCC/Clang.

## Use it in your project

Each [release](https://github.com/MiguelMochizuki/nn-cpp/releases) ships the static library, headers and apps as a CMake install prefix:

```cmake
find_package(nn_cpp 1.3 REQUIRED)    # CMAKE_PREFIX_PATH = the unpacked package
target_link_libraries(my_app PRIVATE nn::core)
```

A regularised CNN with a schedule, clipping and a validation split takes about 30 lines; see the [guide](docs/guide.md) for that, MNIST, FER-2013, checkpoints and writing your own `Dataset`.

## Documentation

| | |
|---|---|
| [Guide](docs/guide.md) | worked examples: regularisation and schedules, MNIST, FER-2013, a character-level Transformer, resuming, custom data |
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
apps/         xor_demo, mnist (train_mnist), fer2013 (train_fer2013), transformer (train_transformer), bench (harness)
tests/        doctest suite, gradient checks, a naive GEMM oracle
docs/         guide, API, architecture, performance reports
```

## License

MIT, see [LICENSE](LICENSE).
