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

`conv2d` is im2col plus the shared `matmul`: each convolution becomes a matrix product (receptive fields unfolded into columns, multiplied by the reshaped filters), and the same cache-blocked GEMM serves the forward pass and both gradients. The default path (`groups == 1`, `dilation == 1`) works one sample per task: each thread unfolds its sample into a private column buffer that fits in the L2 cache, multiplies, and the backward pass unfolds it again instead of keeping the whole `col` tensor alive (about 100 MB for `cnn_large`). The weight and bias gradients are summed per sample and then in sample order, so they do not depend on the thread count. `conv2d_relu` folds the ReLU into the bias epilogue with a one-byte mask, and `Sequential` uses it (and the BatchNorm equivalent) when a `Conv2d` or `BatchNorm` is followed by a `ReLU`. Dilation and groups use a general path.

## Attention

`MultiHeadAttention` projects the input with four `Linear` layers (each one `linear(x, W, b)`, a single node for the matrix product and the bias) and then attends. Without attention dropout it calls the fused `attention(q, k, v, heads, key_padding_mask, causal)` op, which reads each head in place (`lda = D` and a column offset into `sgemm`), so no head is cut, transposed or merged into a copy, forward or backward. With attention dropout active (`Dropout::active()`) it keeps the composed path: `reshape` and `transpose` into heads, scores with `matmul_bt` (so `K^T` is never materialised), `scaled_masked_softmax`, `Dropout`, `matmul` with V, merge. Both paths give the same bits (a test compares forward and backward). `LayerNorm` is the fused `layernorm` op (one thread per row, fixed summation order); `Embedding` uses the `embedding` op, whose backward scatter-adds serially so repeated ids accumulate deterministically.

Masks use two fill values before the softmax: padded keys are set to -1e9 and causally hidden keys to -2e9. Because the causal value is lower, a query row whose real keys are all padded still gives the future exactly zero weight and averages V uniformly over its visible (non-future) keys, instead of reading the future or producing NaN. The headers document this as the behaviour of `MultiHeadAttention::attend`.

Fused ops are the exception and each one was added against a measurement: `linear` and `attention` took 7.3% and 18% off a `transformer_tiny` step, `conv2d_relu` and the per-sample convolution about 27% off `cnn_fer` and `cnn_large`, and `batchnorm`, `layernorm` and `scaled_masked_softmax` replace chains of elementwise nodes. Each fused op is tested against the composition it replaces (bit for bit for `linear` and `attention`) and the thread-count invariance is tested too. Anything else stays composed until a profile shows it matters: the `transformer_tiny` step (vocabulary 64, width 64, 4 heads, 2 layers, batch 16 x 64 tokens, AdamW) is 4.5 ms native and 8.6 ms portable, against 6.2 ms for PyTorch 2.15 CPU; see the [performance notes](performance/README.md#v200-against-pytorch). The GELU uses a vectorised `tanh` (`kernels/vmath.hpp`) built only from correctly rounded operations, so AVX2 and scalar code, and the native and portable builds, give the same bits (absolute error under 1.4e-7 against `std::tanh`).

## Performance kernels

- **GEMM** (`kernels/gemm.hpp`): BLIS-style blocking (panels of 96 x 256 x 1024, a 6 x 16 register tile). The microkernel is chosen at compile time: AVX2 + FMA when both are enabled (for example `-DNN_NATIVE=ON`), SSE2 on other x86-64 builds, an unpacked transpose-aware loop elsewhere (including x86 MSVC, which defines none of those macros). The AVX2 tile keeps its 12 accumulators in named registers, A is packed once per block, and a product with a small output but a long depth (`m * n <= 65536`, `k >= 512`, such as a weight gradient) is cut along `k` into `kKC / 2` pieces that are summed in order. The cut depends only on `(m, n, k)`, never on the thread count.
- **Threads** (`kernels/parallel.hpp`): a fork/join pool with `parallel_for` and `parallel_range`. Work is split so that every output element is written by one thread, so results are bitwise identical for any thread count. The count defaults to `std::thread::hardware_concurrency()`; set it with `NN_NUM_THREADS` or `nn::kernels::set_num_threads(n)`. Idle workers spin for up to `NN_SPIN_US` microseconds (default 100, at most 10000, 0 blocks at once; `set_spin_microseconds()`) before sleeping on the condition variable, which saves the wake-up latency of a pool that runs a few dozen short jobs per step. The library is built with `-ffp-contract=off` so the compiler cannot fuse a multiply and an add differently in two copies of the same loop; that is what keeps results identical across thread counts on `-march=native` builds. `matmul` over a batch runs one matrix per task once the batch reaches the thread count.
- Elementwise ops, ReLU, im2col, col2im, max-pooling, `transpose`, `masked_fill`, `sum`/`max` over an axis, the fused norm and softmax ops, the optimizers and the zero-fill of a large `Tensor` run on the pool on every target; GEMM does on x86-64.
- **Storage:** `Tensor::data()` is a `std::vector<float>` with an allocator that does not value-initialise (`nn::Storage`). `Tensor(shape)` still zero-fills (in parallel for large tensors), and `Tensor::empty(shape)` skips it for results that are written in full. A node's gradient is allocated on its first accumulation (`Value::grad_accum`) and written, not zeroed and added to. On glibc the library raises the `mmap` and trim thresholds with `mallopt` so a step reuses its large buffers instead of returning them to the kernel.

## Checkpoints

`checkpoint.cpp` stores parameters by `named_parameters()` name, then module buffers (BatchNorm running statistics), then the optimizer state from `Optimizer::state()`. The format is little-endian with fixed-width integers and IEEE-754 floats. A load validates every name and shape before touching the model, and writes go to `path.tmp` and are then renamed.

Limits: the architecture is not stored, and layers without parameters leave no trace, so swapping `ReLU` for `Tanh` is not detected. Likewise a changed `Transformer` `heads` or `causal` is not detected when the parameter shapes match, since neither changes a shape; `learned_positions` adds a `pos.pe` parameter, so mixing it up is caught by the name check. `DataLoader` shuffle order, a scheduler's step counter (restore it with `set_step`), a `ReduceLROnPlateau`'s state (best value, bad-epoch counter, cooldown) and Dropout's random stream are not saved. A 1.1 build reads every 1.0 file; a file saved without buffers cannot be loaded into a model that has them.

## Scope

Clarity and correctness come before raw throughput, and the speed work is bounded by the determinism promise: results are bitwise identical for any thread count. Tensors are `float` only (no runtime dtype). `Tensor::random` draws from `std::uniform_real_distribution`, so a seed reproduces initialisation per standard library, not across libstdc++, libc++ and MSVC. `Dataset` is an interface only: `MnistDataset` lives in `apps/mnist/` and `Fer2013Dataset` is the one concrete loader in the library.
