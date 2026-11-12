# Architecture

Four layers, each depending only on the ones before it: `Tensor`, autograd (`Value` and the ops), `nn` modules, then `optim`, `data` and `checkpoint`. Headers are in `include/nn/`, implementations in `src/` mirror that layout.

```
include/nn/
├── tensor.hpp              Tensor: N-D array, arithmetic, reductions, matmul
├── random.hpp              manual_seed, derive_generator
├── checkpoint.hpp          save and load of weights and optimizer state
├── autograd/
│   ├── value.hpp           Value, NodeImpl: the computation-graph handle
│   └── ops.hpp             differentiable ops: add, matmul, conv2d, relu, softmax, cross_entropy_loss, ...
├── nn/
│   ├── module.hpp          Module: base interface for layers
│   ├── linear.hpp, conv2d.hpp, pool2d.hpp, flatten.hpp, activation.hpp
│   ├── dropout.hpp, batchnorm.hpp
│   ├── embedding.hpp, layernorm.hpp, positional.hpp, attention.hpp
│   ├── transformer.hpp     TransformerBlock, Transformer, TransformerConfig
│   ├── sequential.hpp      Sequential: chains layers together
│   └── loss.hpp
├── optim/
│   ├── optimizer.hpp       Optimizer: base interface
│   ├── sgd.hpp, adam.hpp, adamw.hpp
│   ├── scheduler.hpp       LRScheduler, StepLR, CosineAnnealingLR, LinearWarmup
│   └── clip.hpp            clip_grad_norm, clip_grad_value
├── data/
│   ├── dataset.hpp, dataloader.hpp
│   └── tensor_dataset.hpp, subset.hpp, transform.hpp, fer2013.hpp, text.hpp
└── kernels/
    ├── gemm.hpp            blocked sgemm: AVX2/FMA and SSE2 microkernels, unpacked loop elsewhere
    └── parallel.hpp        fork/join thread pool: parallel_for, parallel_range
```

## Autograd

A `Value` is a reference-counted handle over a graph node holding a `Tensor`, its gradient, and, for non-leaf nodes, the parent `Value`s plus a closure that propagates a gradient backward through the operation that produced it. A differentiable function such as `add(a, b)` or `conv2d(x, weight, ...)` computes the forward result and returns a new `Value` wired into the graph. `backward()` on a scalar walks the graph in reverse topological order and runs each closure, accumulating gradients into the parents.

Every layer's `forward()` is written only with these ops, so no layer defines a `backward()`; gradients are derived from the operations that actually ran. A new differentiable op needs its backward closure and a gradient check built on `tests/gradcheck.hpp`.

`NoGradGuard` makes `Value::node` return plain leaves, so inference builds no graph. Graph edges point only from child to parent, so there are no cycles.

## Convolution

`conv2d` is im2col plus the shared `matmul`: each convolution becomes one large matrix product (receptive fields unfolded into columns, multiplied by the reshaped filters). The same cache-blocked GEMM serves the forward pass and both gradients, with no separate convolution loop. Dilation and groups use a general path; the default (`groups == 1`, `dilation == 1`) keeps the original one.

## Attention

`MultiHeadAttention` has no attention kernel of its own. It projects the input with four `Linear` layers, cuts the query, key and value into heads with `reshape` and `transpose`, scores with `matmul_bt` (so `K^T` is never materialised), scales by `1 / sqrt(D / heads)`, applies `masked_fill`, `softmax` and attention-weight `Dropout`, multiplies by V with `matmul`, merges the heads again and applies the output `Linear`. `LayerNorm` is `mean`, `sub`, `mul`, `add` and `rsqrt` followed by the learnable scale and shift; `Embedding` uses the `embedding` op, whose backward scatter-adds serially so repeated ids accumulate deterministically. Every one of them therefore gets its gradient from the existing ops and is covered by their gradient checks.

Masks use two fill values before the softmax: padded keys are set to -1e9 and causally hidden keys to -2e9. Because the causal value is lower, a query row whose real keys are all padded still gives the future exactly zero weight and averages V uniformly over its visible (non-future) keys, instead of reading the future or producing NaN. The headers document this as the behaviour of `MultiHeadAttention::attend`.

Nothing is fused. Every step above materialises its result and has its own backward closure. The `transformer_tiny` bench workload (vocabulary 64, width 64, 4 heads, 2 layers, batch 16 x 64 tokens, AdamW) takes a median of 60.7 ms per step on the native build and 73.4 ms on the portable one (3 runs, 8 threads); about 44% of a step is forward, 55% backward and 0.6% the optimizer. That is all the measurement says. `perf` was unavailable on the measuring machine (`perf_event_paranoid=4`), so there is no per-op profile and it is not known which ops dominate. A fused attention, softmax or LayerNorm is a possible follow-up only after profiling shows where the time goes. See the [performance notes](performance/README.md#v130-transformer_tiny-and-the-v120-guard).

## Performance kernels

- **GEMM** (`kernels/gemm.hpp`): BLIS-style blocking (panels of 96 x 256 x 1024, a 6 x 16 register tile). The microkernel is chosen at compile time: AVX2 + FMA when both are enabled (for example `-DNN_NATIVE=ON`), SSE2 on other x86-64 builds, an unpacked transpose-aware loop elsewhere (including x86 MSVC, which defines none of those macros).
- **Threads** (`kernels/parallel.hpp`): a fork/join pool with `parallel_for` and `parallel_range`. Work is split so that every output element is written by one thread, so results are bitwise identical for any thread count. The count defaults to `std::thread::hardware_concurrency()`; set it with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`.
- Elementwise ops, ReLU, im2col, col2im and max-pooling run on the pool on every target; GEMM does on x86-64.

## Checkpoints

`checkpoint.cpp` stores parameters by `named_parameters()` name, then module buffers (BatchNorm running statistics), then the optimizer state from `Optimizer::state()`. The format is little-endian with fixed-width integers and IEEE-754 floats. A load validates every name and shape before touching the model, and writes go to `path.tmp` and are then renamed.

Limits: the architecture is not stored, and layers without parameters leave no trace, so swapping `ReLU` for `Tanh` is not detected. Likewise a changed `Transformer` `heads` or `causal` is not detected when the parameter shapes match, since neither changes a shape; `learned_positions` adds a `pos.pe` parameter, so mixing it up is caught by the name check. `DataLoader` shuffle order, a scheduler's step counter (restore it with `set_step`), a `ReduceLROnPlateau`'s state (best value, bad-epoch counter, cooldown) and Dropout's random stream are not saved. A 1.1 build reads every 1.0 file; a file saved without buffers cannot be loaded into a model that has them.

## Scope

Clarity and correctness come before raw throughput. Tensors are single-threaded `float` only (no runtime dtype), and BatchNorm is composed from basic ops, so BatchNorm-heavy networks spend most of their time in serial elementwise code (see [performance/v1.1.0-results.md](performance/v1.1.0-results.md)). `Tensor::random` draws from `std::uniform_real_distribution`, so a seed reproduces initialisation per standard library, not across libstdc++, libc++ and MSVC. `Dataset` is an interface only: `MnistDataset` lives in `apps/mnist/` and `Fer2013Dataset` is the one concrete loader in the library.
