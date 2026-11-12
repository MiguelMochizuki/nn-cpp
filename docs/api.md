# API reference

One row per header under `include/nn/`. Every function and class is documented in its header (see [CONTRIBUTING.md](../CONTRIBUTING.md#documentation-conventions)).

| Header | Provides |
|---|---|
| `nn/tensor.hpp` | `Tensor`; `add`, `sub`, `mul`, `div`, `scale`, `matmul` (free functions). `Tensor::random` draws from the library's seeded stream |
| `nn/random.hpp` | `manual_seed(seed)`, `is_seeded()`, `derive_generator(stream)` — one seed, an independent reproducible generator per component (`Tensor::random`, `DataLoader`, `Dropout`, transforms) |
| `nn/autograd/value.hpp` | `Value` — `leaf()`, `node()`, `data()`, `grad()`, `backward()`, `zero_grad()`, `requires_grad()`, `has_grad()` (false for a non-leaf node before `backward()` reaches it, as in PyTorch where such a grad is `None`); `NoGradGuard` (scope that skips graph building, for inference). On a temporary, `data()`/`grad()` return copies, so `model.forward(x).data()` is safe |
| `nn/autograd/ops.hpp` | Differentiable ops: `add`, `sub`, `mul`, `scale`, `sum` (all elements, or over an `axis` with `keepdim`), `mean(x, axis, keepdim)`, `rsqrt`, `exp`, `log`, `sqrt`, `div`, `transpose`, `reshape`, `matmul`, `matmul_bt` (`a * b^T` without a transpose copy), `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`, `masked_fill(x, mask, value)`, `slice(x, axis, start, end)`, `concat(xs, axis)`, `split(x, axis, parts)`, `embedding(weight, ids)`, `conv2d` (stride, padding, plus `dilation` and `groups`), `maxpool2d`, `avgpool2d`, `cross_entropy_loss` (optionally with per-class weights and label smoothing), `sequence_cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0)`, `mse_loss` |
| `nn/nn/module.hpp` | `Module` — base interface (`forward()`, `parameters()`, `named_parameters()`, `named_buffers()`, `train(mode)`, `eval()`, `is_training()`) |
| `nn/nn/linear.hpp` | `Linear(in_features, out_features)` |
| `nn/nn/conv2d.hpp` | `Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)`, or with a `{kh, kw}` kernel plus `dilation=1, groups=1` |
| `nn/nn/pool2d.hpp` | `MaxPool2d(kernel_size, stride)`, `AvgPool2d(kernel_size, stride)`, `GlobalAvgPool2d()` |
| `nn/nn/batchnorm.hpp` | `BatchNorm1d(num_features, eps=1e-5, momentum=0.1)`, `BatchNorm2d(...)`; running statistics are buffers saved by checkpoints |
| `nn/nn/dropout.hpp` | `Dropout(p=0.5)` — inverted dropout in training mode, identity in eval mode |
| `nn/nn/flatten.hpp` | `Flatten` |
| `nn/nn/activation.hpp` | `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax(axis)` |
| `nn/nn/embedding.hpp` | `Embedding(num_embeddings, dim)` — float-valued ids of any shape in, `ids.shape() + {dim}` out |
| `nn/nn/layernorm.hpp` | `LayerNorm(dim, eps=1e-5)` — normalises the last axis; parameters `gamma`, `beta` |
| `nn/nn/positional.hpp` | `PositionalEncoding(max_len, dim, learned=false)` — sinusoidal table (no parameters) or a trainable one (`pe`), added to `{N, T, D}` |
| `nn/nn/attention.hpp` | `MultiHeadAttention(dim, heads, causal=false, dropout=0)` — `forward(x)` is self-attention, `attend(query, context, key_padding_mask)` is cross-attention |
| `nn/nn/transformer.hpp` | `TransformerConfig {vocab_size, dim, heads, layers, max_len, mlp_ratio=4, dropout=0, causal=true, learned_positions=false}`, `TransformerBlock(dim, heads, mlp_ratio, dropout, causal)` (pre-norm), `Transformer(config)` — ids `{N, T}` to logits `{N, T, vocab_size}` |
| `nn/nn/sequential.hpp` | `Sequential` — `add(layer)`, `forward()`, `parameters()` |
| `nn/checkpoint.hpp` | `save_checkpoint(path, model, optimizer*, step)`, `load_checkpoint(path, model, optimizer*)` |
| `nn/nn/loss.hpp` | `cross_entropy_loss(logits, targets)` (and the class-weight / label-smoothing overload), `mse_loss(predictions, targets)`, `sequence_cross_entropy` (all re-exported from `nn/autograd/ops.hpp`) |
| `nn/optim/optimizer.hpp` | `Optimizer` — base interface (`step()`, `zero_grad()`, `lr()`, `set_lr(lr)`) |
| `nn/optim/sgd.hpp` | `SGD(params, lr, momentum=0, nesterov=false, weight_decay=0)` — weight decay is added to the gradient (L2) before the velocity update; Nesterov needs momentum > 0 |
| `nn/optim/plateau.hpp` | `PlateauMode {Min, Max}`, `ReduceLROnPlateau(opt, mode=Min, factor=0.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0)` — `step(metric)`, `num_bad_epochs()`, `best()` |
| `nn/optim/adam.hpp` | `Adam(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)` |
| `nn/optim/adamw.hpp` | `AdamW(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2)` — decoupled weight decay |
| `nn/optim/scheduler.hpp` | `LRScheduler` (`step()`, `set_step(n)`, `step_count()`, `base_lr()`), `StepLR(opt, step_size, gamma=0.1)`, `CosineAnnealingLR(opt, t_max, eta_min=0)`, `LinearWarmup(opt, inner, warmup_steps)` |
| `nn/optim/clip.hpp` | `clip_grad_norm(params, max_norm)` (returns the norm before clipping), `clip_grad_value(params, clip)` |
| `nn/data/dataset.hpp` | `Dataset` — interface (`size()`, `get(index)`) |
| `nn/data/tensor_dataset.hpp` | `TensorDataset(inputs, targets)` — rows of two tensors as samples |
| `nn/data/subset.hpp` | `Subset(dataset, indices)`, `random_split(dataset, fractions, seed)` |
| `nn/data/transform.hpp` | `Transform`, `Compose`, `Normalize(mean, std)`, `RandomHorizontalFlip(p)`, `RandomShift(max_shift)`, `RandomCrop(size)`, `RandomAffine(max_degrees, max_shift, max_scale, p=0.5)`, `RandomErasing(p=0.5, min_area=0.02, max_area=0.33, min_ratio=0.3, max_ratio=3.3, value=0)`, `ten_crop(sample, size)`, `ten_crop_batch(batch, size)`, `TransformedDataset(dataset, transform)`, `compute_mean_std(dataset)` |
| `nn/data/text.hpp` | `CharVocab(text)` — `size()`, `encode(s)`, `decode(ids)` over the sorted distinct bytes; `SequenceDataset(tokens, block_size)` — sample `i` is tokens `[i, i+T)` and the same window shifted by one |
| `nn/data/fer2013.hpp` | `Fer2013Dataset(csv_path, usage=, image_size=48)` — FER-2013 CSV loader (the data is not included) |
| `nn/metrics.hpp` | scikit-learn-style classification metrics: `confusion_matrix`, `accuracy_score`, `precision_recall_f1` (per class), `precision_score` / `recall_score` / `f1_score` (`Average::Micro`, `Macro`, `Weighted`; `zero_division`), `classification_report`, plus `argmax_rows` and `to_labels` to get labels from logits and from the `DataLoader`'s float ids |
| `nn/kernels/parallel.hpp` | `num_threads()`, `set_num_threads(n)`, `parallel_for(n_chunks, fn)`, `parallel_range(n, min_per_chunk, fn)` — fork/join thread pool used by `sgemm` (x86-64), the elementwise ops, ReLU, im2col/col2im and max-pooling |
| `nn/data/dataloader.hpp` | `DataLoader(dataset, batch_size, shuffle[, drop_last])`, `DataLoader(dataset, batch_size, sample_weights[, num_samples=0, drop_last=false])` (weighted draws with replacement), `balanced_sample_weights(labels, num_classes)` — `num_batches()`, `reset_epoch()`, `get_batch(i)`; the shuffle is a portable Fisher-Yates, so a seed gives the same order on every platform |
