# nn-cpp

A neural network library written from scratch in C++17: a general N-dimensional tensor type, a reverse-mode automatic differentiation engine, and the layers, loss functions, and optimizer needed to build and train a convolutional network. No external math or machine learning dependencies.

As a demonstration, the library trains a small CNN on MNIST to **98.5% test accuracy** in under two minutes on a single CPU core.

## Overview

The library is organized in four layers, each building on the one before it:

- **`Tensor`** — a general N-dimensional `float` array with row-major storage: construction, indexing, reshaping, broadcasting arithmetic, reductions, and batched matrix multiplication.
- **Autograd** — a `Value` handle wraps a `Tensor` in a computation graph. Differentiable operations (`add`, `matmul`, `conv2d`, `relu`, `cross_entropy_loss`, ...) build the graph as they run; `backward()` walks it in reverse to compute gradients.
- **`nn`** — layers (`Linear`, `Conv2d`, `MaxPool2d`, `Flatten`, activations) and the `Sequential` container that chains them. A layer only implements `forward()`; gradients are derived automatically by the autograd engine underneath.
- **`optim` and `data`** — `SGD` and `Adam` optimizers, checkpoint save/load, and a `Dataset`/`DataLoader` pair for batching and shuffling training data from any source you implement.

The design takes inspiration from PyTorch's vocabulary (`Module`, `Sequential`, `Optimizer`) without depending on it or anything else — every operation here is implemented directly.

## Project Organization

```
apps/           examples: xor/ (XOR demo) and mnist/ (IDX loader, MNIST trainer and its tests)
cmake/          FetchDoctest.cmake (fetches doctest for the tests)
include/nn/
  autograd/     Value and the differentiable operations
  nn/           Module, layers, losses and Sequential
  optim/        Optimizer, SGD and Adam
  data/         Dataset and DataLoader
  kernels/      gemm.hpp: blocked sgemm with AVX2/FMA and SSE2 microkernels, unpacked loop elsewhere
  tensor.hpp    the Tensor type
  checkpoint.hpp  save and load of weights and optimizer state
src/            implementations, mirroring include/nn
tests/          unit tests (doctest) and the gradient-check helper
```

## Build

Requirements: a C++17 compiler and CMake 3.16+.

```bash
cmake -S . -B build
cmake --build build -j
```

This builds the `nn_core` static library, the example apps, and the test suite (`ctest`/[doctest](https://github.com/doctest/doctest), fetched automatically). Tests can be skipped with `-DNN_BUILD_TESTS=OFF`; examples with `-DNN_BUILD_APPS=OFF`. `-DNN_NATIVE=ON` builds for the host CPU, which on a CPU with AVX2 and FMA enables the AVX2/FMA GEMM microkernel.

Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

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

### Saving and resuming

`save_checkpoint` writes a model's parameters, an optimizer's state (SGD learning rate, momentum and velocity) and a step counter of your choosing to one file; `load_checkpoint` restores them.

```cpp
#include "nn/checkpoint.hpp"

nn::save_checkpoint("model.ckpt", model, &optimizer, epoch);

// later, on any machine: rebuild the same model in code, then load into it
int64_t epoch = nn::load_checkpoint("model.ckpt", model, &optimizer);   // resume training
nn::load_checkpoint("model.ckpt", model);                               // inference only
```

The file is little-endian with fixed-width integers and IEEE-754 floats, so it loads identically anywhere. Parameters are stored by name (`0.weight`, `2.bias`, ...); loading checks every name and shape first and throws `std::runtime_error` naming what differs, leaving the model untouched. The architecture itself is not stored, so you must rebuild the same model in code. Layers without parameters leave no trace, so swapping `ReLU` for `Tanh` at the same position is not detected. Resuming does not restore `DataLoader` shuffle order.

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
│   ├── sequential.hpp        Sequential: chains layers together
│   └── loss.hpp
├── optim/
│   ├── optimizer.hpp         Optimizer: base interface
│   └── sgd.hpp                SGD (with optional momentum)
└── data/
    ├── dataset.hpp            Dataset: interface you implement for your own data
    └── dataloader.hpp         DataLoader: batching and shuffling over any Dataset
```

### How the autograd engine works

A `Value` is a reference-counted handle over a graph node holding a `Tensor`, its gradient, and (for non-leaf nodes) a list of parent `Value`s plus a closure describing how to propagate a gradient backward through the operation that produced it. Calling a differentiable function like `add(a, b)` or `conv2d(x, weight, bias, ...)` computes the forward result and returns a new `Value` wired into this graph; calling `.backward()` on a scalar `Value` walks the graph in reverse topological order, running each node's closure to accumulate gradients into its parents.

Because every layer's `forward()` is expressed purely in terms of these differentiable operations, no layer needs to implement its own `backward()` — the engine derives every gradient generically from the operations that were actually run.

### Convolution

`conv2d` is implemented via **im2col + matrix multiplication**: each convolution is reshaped into a single large matrix multiply (unfolding the input's receptive fields into columns, multiplying by the reshaped filter weights), which reuses the tensor library's one general-purpose, cache-friendly matmul routine for both the forward pass and its gradient, rather than a separate hand-written convolution loop.

## API Reference Summary

| Header | Provides |
|---|---|
| `nn/tensor.hpp` | `Tensor`; `add`, `sub`, `mul`, `div`, `scale`, `matmul` (free functions) |
| `nn/autograd/value.hpp` | `Value` — `leaf()`, `node()`, `data()`, `grad()`, `backward()`, `zero_grad()`, `requires_grad()`; `NoGradGuard` (scope that skips graph building, for inference). On a temporary, `data()`/`grad()` return copies, so `model.forward(x).data()` is safe |
| `nn/autograd/ops.hpp` | Differentiable ops: `add`, `sub`, `mul`, `scale`, `sum`, `transpose`, `reshape`, `matmul`, `matmul_bt` (`a * b^T` without a transpose copy), `relu`, `sigmoid`, `tanh`, `softmax`, `conv2d`, `maxpool2d`, `cross_entropy_loss`, `mse_loss` |
| `nn/nn/module.hpp` | `Module` — base interface (`forward()`, `parameters()`) |
| `nn/nn/linear.hpp` | `Linear(in_features, out_features)` |
| `nn/nn/conv2d.hpp` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)` |
| `nn/nn/pool2d.hpp` | `MaxPool2d(kernel_size, stride)` |
| `nn/nn/flatten.hpp` | `Flatten` |
| `nn/nn/activation.hpp` | `ReLU`, `Sigmoid`, `Tanh`, `Softmax(axis)` |
| `nn/nn/sequential.hpp` | `Sequential` — `add(layer)`, `forward()`, `parameters()` |
| `nn/checkpoint.hpp` | `save_checkpoint(path, model, optimizer*, step)`, `load_checkpoint(path, model, optimizer*)` |
| `nn/nn/loss.hpp` | `cross_entropy_loss(logits, targets)`, `mse_loss(predictions, targets)` |
| `nn/optim/optimizer.hpp` | `Optimizer` — base interface (`step()`, `zero_grad()`) |
| `nn/optim/sgd.hpp` | `SGD(params, lr, momentum=0)` |
| `nn/optim/adam.hpp` | `Adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)` |
| `nn/data/dataset.hpp` | `Dataset` — interface (`size()`, `get(index)`) |
| `nn/data/dataloader.hpp` | `DataLoader(dataset, batch_size, shuffle)` — `num_batches()`, `reset_epoch()`, `get_batch(i)` |

## Scope

This project targets clarity and correctness over raw throughput: matrix products go through a cache-blocked, single-threaded GEMM kernel (`nn/kernels/gemm.hpp`) with an AVX2/FMA microkernel (when compiled for AVX2 and FMA, for example `-DNN_NATIVE=ON` on such a CPU), an SSE2 microkernel on other x86-64 builds and an unpacked transpose-aware loop on other targets; there are still no threads. The tensor's dtype is fixed to `float`; a runtime-selectable dtype is a natural future extension but is not implemented here.

## Documentation Conventions

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License, see [LICENSE](./LICENSE).
