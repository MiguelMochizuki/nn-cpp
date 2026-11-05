# nn-cpp

A neural network library written from scratch in C++17: a general N-dimensional tensor type, a reverse-mode automatic differentiation engine, and the layers, loss functions, and optimizer needed to build and train a convolutional network. No external math or machine learning dependencies.

As a demonstration, the library trains a small CNN on MNIST to **98.5% test accuracy** in under two minutes on a single CPU core.

## Overview

The library is organized in four layers, each building on the one before it:

- **`Tensor`** — a general N-dimensional `float` array with row-major storage: construction, indexing, reshaping, broadcasting arithmetic, reductions, and batched matrix multiplication.
- **Autograd** — a `Value` handle wraps a `Tensor` in a computation graph. Differentiable operations (`add`, `matmul`, `conv2d`, `relu`, `cross_entropy_loss`, ...) build the graph as they run; `backward()` walks it in reverse to compute gradients.
- **`nn`** — layers (`Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool2d`, `BatchNorm1d/2d`, `Dropout`, `Flatten`, activations) and the `Sequential` container that chains them, with `train()`/`eval()` modes. A layer only implements `forward()`; gradients are derived automatically by the autograd engine underneath.
- **`optim` and `data`** — `SGD`, `Adam` and `AdamW` optimizers, learning-rate schedulers, gradient clipping, checkpoint save/load, and a `Dataset`/`DataLoader` pair for batching and shuffling training data from any source you implement, plus `TensorDataset`, `random_split`, input transforms and a FER-2013 CSV loader. `manual_seed()` makes initialisation, shuffling and dropout reproducible.

The design takes inspiration from PyTorch's vocabulary (`Module`, `Sequential`, `Optimizer`) without depending on it or anything else — every operation here is implemented directly.

## Project Organization

```
apps/
  bench/          benchmark harness (bench, bench_tests), run_baseline.sh / profile.sh, compare_csv.py (before/after CSV ratios)
  mnist/          IDX parser, MnistDataset, train_mnist and its tests
  xor/            xor_demo
  fer2013/        train_fer2013 (Conv-BN-ReLU network on a FER-2013 CSV)
  CMakeLists.txt
cmake/            FetchDoctest.cmake (fetches doctest for the tests), nn_cppConfig.cmake.in (package config)
docs/
  baseline/       recorded benchmark CSVs, perf profiles and MNIST runs: top level = Phase 0 baseline,
                  phase1/ to phase3/ and followups/ = later runs
  releasing.md    steps to cut a release
  phase0-results.md ... phase3-results.md, followups-results.md, v1.1.0-results.md   per-phase measurements
include/nn/
  autograd/       Value and the differentiable operations
  nn/             Module, layers (including Dropout, BatchNorm, pooling), losses and Sequential
  optim/          Optimizer, SGD, Adam, AdamW, LR schedulers and gradient clipping
  data/           Dataset, DataLoader, TensorDataset, Subset/random_split, transforms and the FER-2013 loader
  kernels/        gemm.hpp: blocked sgemm (AVX2/FMA and SSE2 microkernels, unpacked loop elsewhere);
                  parallel.hpp: fork/join thread pool (parallel_for, parallel_range)
  random.hpp      manual_seed and the per-component random streams
  tensor.hpp      the Tensor type
  checkpoint.hpp  save and load of weights and optimizer state
src/              implementations, mirroring include/nn
tests/            doctest unit tests, gradcheck.hpp (gradient check) and reference_gemm.hpp (naive GEMM oracle)
CMakeLists.txt, CHANGELOG.md, CONTRIBUTING.md, LICENSE
```

## Main Tools & Stack

- Language: C++17 (`CMAKE_CXX_STANDARD 17`), no external math or ML libraries.
- Build: CMake 3.16+; options `NN_BUILD_TESTS`, `NN_BUILD_APPS`, `NN_NATIVE` (`-march=native`).
- Tests: [doctest](https://github.com/doctest/doctest), fetched with `FetchContent` (`cmake/FetchDoctest.cmake`), run through `ctest`.
- Threading: `std::thread` pool in `nn/kernels/parallel.hpp` (linked via `Threads::Threads`).
- SIMD: the GEMM microkernel is chosen at compile time: AVX2 + FMA when both are enabled (for example `-DNN_NATIVE=ON` on a CPU that has them), SSE2 on other x86-64 builds, a scalar/unpacked path elsewhere.
- Benchmark tooling: `apps/bench` (C++ harness plus shell scripts) and `compare_csv.py` (Python 3, standard library only).

## Key Outputs & Reports

### Speedup since the Phase 0 baseline

Training-step time (`total_ms`: forward, backward and optimizer step) from the committed benchmark CSVs. Machine: Intel i5-12450HX (4 P-cores with hyperthreading = cpus 0-7). Protocol: `NN_NUM_THREADS=8 taskset -c 0-7`, 15 timed runs per row. "Phase 0" is the single pinned t8 baseline file (`docs/baseline/native-pinned-t8.csv`, `portable-pinned-t8.csv`); "Today" is the median of the three `new-r1..r3` runs in `docs/baseline/followups/packb-native/` and `packb-portable/`. The native build uses `-DNN_NATIVE=ON`; the portable build does not. The CPU governor was `powersave`, so absolute times are noisy; see [docs/followups-results.md](docs/followups-results.md).

| workload | native Phase 0 (ms) | native today (ms) | native speedup | portable Phase 0 (ms) | portable today (ms) | portable speedup |
|---|---|---|---|---|---|---|
| cnn_mnist | 17.63 | 2.64 | 6.7x | 20.39 | 3.88 | 5.3x |
| mlp_wide | 173.48 | 14.18 | 12.2x | 215.13 | 26.56 | 8.1x |
| cnn_large | 1030.92 | 250.22 | 4.1x | 1180.54 | 304.10 | 3.9x |
| mlp_adam | 171.24 | 16.39 | 10.4x | 228.33 | 29.01 | 7.9x |

`python3 apps/bench/compare_csv.py before.csv after.csv` prints the same kind of ratio for any two result files.

### Real MNIST

`train_mnist` on the full MNIST set (3 epochs, `NN_NUM_THREADS=8 taskset -c 0-7`). Weight init and shuffling are unseeded, so accuracy varies from run to run.

| build | test accuracy | wall time | source |
|---|---|---|---|
| native | 98.58% | 17.8 s | `docs/baseline/phase3/mnist-native.txt` |
| portable | 98.43% | 46.3 s | `docs/baseline/phase1/mnist-portable.txt` (latest portable run) |

The Phase 0 runs (98.11% native, 98.47% portable) took about 1:42 each on a single thread.

### Per-phase reports

[Phase 0](docs/phase0-results.md) (baseline and profile), [Phase 1](docs/phase1-results.md), [Phase 2](docs/phase2-results.md), [Phase 3](docs/phase3-results.md), [follow-ups](docs/followups-results.md), [v1.1.0](docs/v1.1.0-results.md) (performance guard against v1.0.0 and the `cnn_fer` workload).

## Build

Requirements: a C++17 compiler and CMake 3.16+.

```bash
cmake -S . -B build
cmake --build build -j
```

This builds the `nn_core` static library, the example apps, and the test suite (`ctest`/[doctest](https://github.com/doctest/doctest), fetched automatically). Tests can be skipped with `-DNN_BUILD_TESTS=OFF`; examples with `-DNN_BUILD_APPS=OFF`. `-DNN_NATIVE=ON` builds for the host CPU, which on a CPU with AVX2 and FMA enables the AVX2/FMA GEMM microkernel. The elementwise ops, ReLU, im2col, col2im and max-pooling run on a thread pool on every target, and on x86-64 matrix products do too; the thread count defaults to `std::thread::hardware_concurrency()` and can be set with the `NN_NUM_THREADS` environment variable or `nn::kernels::set_num_threads(n)`. Results do not depend on the thread count.

On MSVC the compiler defines neither `__SSE2__` nor `__AVX2__`/`__FMA__`, so x86 MSVC builds use the unpacked `sgemm`: correct, but slower than the GCC/Clang builds.

Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

## Releases and versioning

Versions follow [Semantic Versioning](https://semver.org/): a breaking API change bumps the major version, a new feature the minor version, and a fix the patch version. The changes of each version are in [CHANGELOG.md](CHANGELOG.md).

Each release is on the GitHub Releases page with packages for Linux (x86-64, ARM64), Windows (x86-64, ARM64) and macOS (universal), named `nn-cpp-<version>-<os>-<arch>.tar.gz` (`.zip` on Windows), plus a source archive. A package holds the static library, the headers and the apps (`bin/`), laid out as a CMake install prefix. To use it from another project, point `CMAKE_PREFIX_PATH` at the unpacked package (or at an install prefix from `cmake --install build --prefix <dir>`):

```cmake
find_package(nn_cpp 1.0 REQUIRED)
target_link_libraries(my_app PRIVATE nn::core)
```

`nn::core` carries the include path and the `Threads` dependency. The steps to cut a release are in [docs/releasing.md](docs/releasing.md).

## Usage

### Quick start: XOR

```cpp
#include "nn/nn/sequential.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/sgd.hpp"

using namespace nn;

Tensor x_data({4, 2}, std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1});
Tensor y_data({4, 1}, std::vector<float>{0, 1, 1, 0});
Value x = Value::leaf(x_data, false);
Value y = Value::leaf(y_data, false);

Sequential model;
model.add(std::make_unique<Linear>(2, 8));
model.add(std::make_unique<ReLU>());
model.add(std::make_unique<Linear>(8, 1));

SGD optimizer(model.parameters(), 0.1f);

for (int epoch = 0; epoch < 2000; epoch++) {
    Value pred = model.forward(x);
    Value loss = mse_loss(pred, y);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```

Run the built-in version of this example:

```bash
./build/apps/xor_demo
```

### Training a CNN on MNIST

`apps/mnist/train_mnist.cpp` builds a small convolutional network (two `Conv2d`+`MaxPool2d` blocks, then two `Linear` layers), trains it on MNIST with `SGD` and `cross_entropy_loss`, and reports test-set accuracy:

```bash
./build/apps/train_mnist <path-to-directory-containing-the-4-idx-files>
```

The directory must contain the four standard MNIST files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`. These are not downloaded automatically — obtain them from an MNIST mirror and point the program at the directory holding them.

### Regularisation, schedules and data pipelines (1.1)

```cpp
#include "nn/random.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/nn/dropout.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/scheduler.hpp"
#include "nn/optim/clip.hpp"
#include "nn/data/tensor_dataset.hpp"
#include "nn/data/subset.hpp"
#include "nn/data/transform.hpp"

nn::manual_seed(42);   // weights, shuffles, dropout masks and transforms repeat for the same seed and thread count

nn::TensorDataset all(images, labels);                         // images {N,1,H,W}, labels {N}
auto parts = nn::random_split(all, {0.8, 0.2}, /*seed=*/1);    // train / validation
auto flip = std::make_shared<nn::RandomHorizontalFlip>(0.5f);
nn::TransformedDataset train_set(parts[0], flip);
nn::DataLoader train_loader(train_set, 32, /*shuffle=*/true);

nn::Sequential model;
model.add(std::make_unique<nn::Conv2d>(1, 8, 3, 1, 1));
model.add(std::make_unique<nn::BatchNorm2d>(8));
model.add(std::make_unique<nn::ReLU>());
model.add(std::make_unique<nn::GlobalAvgPool2d>());
model.add(std::make_unique<nn::Dropout>(0.3f));
model.add(std::make_unique<nn::Linear>(8, 3));

nn::AdamW optimizer(model.parameters(), 1e-2f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/1e-2f);
nn::CosineAnnealingLR schedule(optimizer, /*t_max=*/epochs * train_loader.num_batches());

model.train();                                                 // batch statistics, dropout active
for (/* epochs and batches */) {
    auto [x, y] = train_loader.get_batch(b);
    nn::Value loss = nn::cross_entropy_loss(model.forward(x), y);
    optimizer.zero_grad();
    loss.backward();
    nn::clip_grad_norm(model.parameters(), 1.0f);
    optimizer.step();
    schedule.step();                                           // optimizer.lr() follows the cosine
}
model.eval();                                                  // running statistics, dropout is the identity
```

This is a tested program (built against an installed prefix with `find_package(nn_cpp 1.1)`), trimmed here to the relevant lines. `cross_entropy_loss` also takes per-class weights and a label-smoothing factor.

### Training on FER-2013 (1.1)

`apps/fer2013/train_fer2013.cpp` trains a Conv-BatchNorm-ReLU network with global average pooling and dropout on the FER-2013 facial-expression CSV (columns `emotion,pixels,Usage`; `Training` rows train, `PublicTest` rows validate). FER-2013 is not part of this repository and is not downloaded: obtain `fer2013.csv` separately, for example from its Kaggle challenge page.

```bash
./build/apps/train_fer2013 <fer2013.csv> [--epochs N] [--batch N] [--lr X] [--seed N] [--out ckpt]
```

Defaults are `--epochs 20 --batch 64 --lr 1e-3 --seed 1 --out fer2013.ckpt`. The `cnn_fer` row of `apps/bench` times one step of the same network shape.

### Saving and resuming

`save_checkpoint` writes a model's parameters, an optimizer's state (SGD learning rate, momentum and velocity) and a step counter of your choosing to one file; `load_checkpoint` restores them.

```cpp
#include "nn/checkpoint.hpp"

nn::save_checkpoint("model.ckpt", model, &optimizer, epoch);

// later, on any machine: rebuild the same model in code, then load into it
int64_t epoch = nn::load_checkpoint("model.ckpt", model, &optimizer);   // resume training
nn::load_checkpoint("model.ckpt", model);                               // inference only
```

The file is little-endian with fixed-width integers and IEEE-754 floats, so it loads identically anywhere. Parameters are stored by name (`0.weight`, `2.bias`, ...), followed by module buffers (BatchNorm running statistics); the optimizer's current learning rate is saved too, but a scheduler's step counter is not (restore it with `set_step`), nor is Dropout's random stream; loading checks every name and shape first and throws `std::runtime_error` naming what differs, leaving the model untouched. The architecture itself is not stored, so you must rebuild the same model in code. Layers without parameters leave no trace, so swapping `ReLU` for `Tanh` at the same position is not detected. Resuming does not restore `DataLoader` shuffle order. A v1.1 build reads every v1.0 file; a file saved without buffers cannot be loaded into a model that has them.

### Using your own data

`Dataset` is a two-method interface (`size()`, `get(index)`); implement it against whatever storage format you have, and `DataLoader` handles batching and shuffling for you:

```cpp
class MyDataset : public nn::Dataset {
public:
    size_t size() const override { /* ... */ }
    std::pair<nn::Tensor, nn::Tensor> get(size_t index) const override { /* ... */ }
};

MyDataset dataset;
nn::DataLoader loader(dataset, /*batch_size=*/32, /*shuffle=*/true);

for (size_t b = 0; b < loader.num_batches(); b++) {
    auto [x_batch, y_batch] = loader.get_batch(b);
    // forward / backward / step
}
```

The library ships no concrete `Dataset` implementation of its own — `MnistDataset`, used by the MNIST example, lives in `apps/mnist/` as a demonstration, not part of the core library.

## Architecture

```
include/nn/
├── tensor.hpp              Tensor: N-D array, arithmetic, reductions, matmul
├── autograd/
│   ├── value.hpp            Value, NodeImpl: the computation-graph handle
│   └── ops.hpp               differentiable ops: add, matmul, conv2d, relu, softmax,
│                              cross_entropy_loss, mse_loss, ...
├── nn/
│   ├── module.hpp            Module: base interface for layers
│   ├── linear.hpp, conv2d.hpp, pool2d.hpp, flatten.hpp, activation.hpp
│   ├── dropout.hpp, batchnorm.hpp
│   ├── sequential.hpp        Sequential: chains layers together
│   └── loss.hpp
├── optim/
│   ├── optimizer.hpp         Optimizer: base interface
│   ├── sgd.hpp                SGD (with optional momentum)
│   ├── adam.hpp, adamw.hpp    Adam, AdamW
│   ├── scheduler.hpp          LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup
│   └── clip.hpp               clip_grad_norm, clip_grad_value
├── random.hpp                  manual_seed, derive_generator
└── data/
    ├── dataset.hpp            Dataset: interface you implement for your own data
    ├── dataloader.hpp         DataLoader: batching and shuffling over any Dataset
    ├── tensor_dataset.hpp, subset.hpp, transform.hpp, fer2013.hpp
```

### How the autograd engine works

A `Value` is a reference-counted handle over a graph node holding a `Tensor`, its gradient, and (for non-leaf nodes) a list of parent `Value`s plus a closure describing how to propagate a gradient backward through the operation that produced it. Calling a differentiable function like `add(a, b)` or `conv2d(x, weight, bias, ...)` computes the forward result and returns a new `Value` wired into this graph; calling `.backward()` on a scalar `Value` walks the graph in reverse topological order, running each node's closure to accumulate gradients into its parents.

Because every layer's `forward()` is expressed purely in terms of these differentiable operations, no layer needs to implement its own `backward()` — the engine derives every gradient generically from the operations that were actually run.

### Convolution

`conv2d` is implemented via **im2col + matrix multiplication**: each convolution is reshaped into a single large matrix multiply (unfolding the input's receptive fields into columns, multiplying by the reshaped filter weights), which reuses the tensor library's one general-purpose, cache-friendly matmul routine for both the forward pass and its gradient, rather than a separate hand-written convolution loop.

## API Reference Summary

| Header | Provides |
|---|---|
| `nn/tensor.hpp` | `Tensor`; `add`, `sub`, `mul`, `div`, `scale`, `matmul` (free functions). `Tensor::random` draws from the library's seeded stream |
| `nn/random.hpp` | `manual_seed(seed)`, `is_seeded()`, `derive_generator(stream)` — one seed, an independent reproducible generator per component (`Tensor::random`, `DataLoader`, `Dropout`, transforms) |
| `nn/autograd/value.hpp` | `Value` — `leaf()`, `node()`, `data()`, `grad()`, `backward()`, `zero_grad()`, `requires_grad()`, `has_grad()` (false for a non-leaf node before `backward()` reaches it, as in PyTorch where such a grad is `None`); `NoGradGuard` (scope that skips graph building, for inference). On a temporary, `data()`/`grad()` return copies, so `model.forward(x).data()` is safe |
| `nn/autograd/ops.hpp` | Differentiable ops: `add`, `sub`, `mul`, `scale`, `sum` (all elements, or over an `axis` with `keepdim`), `mean(x, axis, keepdim)`, `rsqrt`, `transpose`, `reshape`, `matmul`, `matmul_bt` (`a * b^T` without a transpose copy), `relu`, `sigmoid`, `tanh`, `softmax`, `conv2d` (stride, padding, plus `dilation` and `groups`), `maxpool2d`, `avgpool2d`, `cross_entropy_loss` (optionally with per-class weights and label smoothing), `mse_loss` |
| `nn/nn/module.hpp` | `Module` — base interface (`forward()`, `parameters()`, `named_parameters()`, `named_buffers()`, `train(mode)`, `eval()`, `is_training()`) |
| `nn/nn/linear.hpp` | `Linear(in_features, out_features)` |
| `nn/nn/conv2d.hpp` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`, or with a `{kh, kw}` kernel plus `dilation=1, groups=1` |
| `nn/nn/pool2d.hpp` | `MaxPool2d(kernel_size, stride)`, `AvgPool2d(kernel_size, stride)`, `GlobalAvgPool2d()` |
| `nn/nn/batchnorm.hpp` | `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`, `BatchNorm2d(...)`; running statistics are buffers saved by checkpoints |
| `nn/nn/dropout.hpp` | `Dropout(p=0.5)` — inverted dropout in training mode, identity in eval mode |
| `nn/nn/flatten.hpp` | `Flatten` |
| `nn/nn/activation.hpp` | `ReLU`, `Sigmoid`, `Tanh`, `Softmax(axis)` |
| `nn/nn/sequential.hpp` | `Sequential` — `add(layer)`, `forward()`, `parameters()` |
| `nn/checkpoint.hpp` | `save_checkpoint(path, model, optimizer*, step)`, `load_checkpoint(path, model, optimizer*)` |
| `nn/nn/loss.hpp` | `cross_entropy_loss(logits, targets)` (and the class-weight / label-smoothing overload), `mse_loss(predictions, targets)` |
| `nn/optim/optimizer.hpp` | `Optimizer` — base interface (`step()`, `zero_grad()`, `lr()`, `set_lr(lr)`) |
| `nn/optim/sgd.hpp` | `SGD(params, lr, momentum=0)` |
| `nn/optim/adam.hpp` | `Adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)` |
| `nn/optim/adamw.hpp` | `AdamW(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)` — decoupled weight decay |
| `nn/optim/scheduler.hpp` | `LRScheduler` (`step()`, `set_step(n)`, `step_count()`, `base_lr()`), `StepLR(opt, step_size, gamma=0.1)`, `CosineAnnealingLR(opt, t_max, eta_min=0)`, `LinearWarmup(opt, inner, warmup_steps)` |
| `nn/optim/clip.hpp` | `clip_grad_norm(params, max_norm)` (returns the norm before clipping), `clip_grad_value(params, clip)` |
| `nn/data/dataset.hpp` | `Dataset` — interface (`size()`, `get(index)`) |
| `nn/data/tensor_dataset.hpp` | `TensorDataset(inputs, targets)` — rows of two tensors as samples |
| `nn/data/subset.hpp` | `Subset(dataset, indices)`, `random_split(dataset, fractions, seed)` |
| `nn/data/transform.hpp` | `Transform`, `Compose`, `Normalize(mean, std)`, `RandomHorizontalFlip(p)`, `RandomShift(max_shift)`, `TransformedDataset(dataset, transform)`, `compute_mean_std(dataset)` |
| `nn/data/fer2013.hpp` | `Fer2013Dataset(csv_path, usage=, image_size=48)` — FER-2013 CSV loader (the data is not included) |
| `nn/kernels/parallel.hpp` | `num_threads()`, `set_num_threads(n)`, `parallel_for(n_chunks, fn)`, `parallel_range(n, min_per_chunk, fn)` — fork/join thread pool used by `sgemm` (x86-64), the elementwise ops, ReLU, im2col/col2im and max-pooling |
| `nn/data/dataloader.hpp` | `DataLoader(dataset, batch_size, shuffle[, drop_last])` — `num_batches()`, `reset_epoch()`, `get_batch(i)`; the shuffle is a portable Fisher-Yates, so a seed gives the same order on every platform |

## Scope

This project targets clarity and correctness over raw throughput: matrix products go through a cache-blocked, multithreaded GEMM kernel (`nn/kernels/gemm.hpp`) with an AVX2/FMA microkernel (when compiled for AVX2 and FMA, for example `-DNN_NATIVE=ON` on such a CPU), an SSE2 microkernel on other x86-64 builds and an unpacked transpose-aware loop on other targets; the GEMM is threaded on x86-64 only, while the elementwise ops, ReLU, im2col, col2im and max-pooling are threaded on all targets (a fork/join pool in `nn/kernels/parallel.hpp`; GEMM results are bitwise identical for any thread count); op results are moved into the autograd graph rather than copied. The tensor's dtype is fixed to `float`; a runtime-selectable dtype is a natural future extension but is not implemented here. Version 1.1 adds what a small CNN training run needs (seeding, train/eval modes, BatchNorm, Dropout, AdamW, schedulers, clipping, a data pipeline) without touching the v1.0 API, but these pieces are composed from basic ops and are not yet fast: BatchNorm-heavy networks run mostly on serial elementwise code (see [docs/v1.1.0-results.md](docs/v1.1.0-results.md)). Dropout's random stream is not saved in checkpoints, and `Tensor::random` comes from `std::uniform_real_distribution`, so a seed reproduces initialisation per standard library, not across libstdc++, libc++ and MSVC.

## Documentation Conventions

Code comments and docstrings follow [CONTRIBUTING.md](CONTRIBUTING.md#documentation-conventions).

## License

MIT License, see [LICENSE](./LICENSE).
