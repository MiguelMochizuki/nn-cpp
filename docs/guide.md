# Guide

Worked examples beyond the quick start in the [README](../README.md). The full list of headers is in the [API reference](api.md).

## Training a CNN on MNIST

`apps/mnist/train_mnist.cpp` builds a small convolutional network (two `Conv2d`+`MaxPool2d` blocks, then two `Linear` layers), trains it on MNIST with `SGD` and `cross_entropy_loss`, and reports test-set accuracy:

```bash
./build/apps/train_mnist <path-to-directory-containing-the-4-idx-files>
```

The directory must contain the four standard MNIST files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`. These are not downloaded automatically — obtain them from an MNIST mirror and point the program at the directory holding them.

## Regularisation, schedules and data pipelines

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

This is a tested program (built against an installed prefix with `find_package(nn_cpp 1.2)`), trimmed here to the relevant lines. `cross_entropy_loss` also takes per-class weights and a label-smoothing factor.

## Augmentation and test-time views

```cpp
#include "nn/data/transform.hpp"
#include "nn/optim/sgd.hpp"
#include "nn/optim/plateau.hpp"

std::vector<std::unique_ptr<nn::Transform>> steps;                           // {1,H,W} samples, H = W = 48
steps.push_back(std::make_unique<nn::RandomAffine>(10.0f, 0.06f, 0.1f));     // degrees, shift fraction, scale
steps.push_back(std::make_unique<nn::RandomCrop>(44));
steps.push_back(std::make_unique<nn::RandomErasing>(0.5f));
nn::TransformedDataset train_set(all, std::make_shared<nn::Compose>(std::move(steps)));

nn::SGD optimizer(model.parameters(), 0.05f, /*momentum=*/0.9f, /*nesterov=*/true, /*weight_decay=*/5e-4f);
nn::ReduceLROnPlateau plateau(optimizer, nn::PlateauMode::Min, /*factor=*/0.5f, /*patience=*/3);
// each epoch: train on train_set, then plateau.step(validation_loss);

model.eval();
nn::NoGradGuard no_grad;                                                     // no graph while predicting
nn::Tensor probs = nn::Tensor::zeros({n, classes});                          // n = images.shape()[0]
for (const nn::Tensor& view : nn::ten_crop_batch(images, 44))                // images {n,1,48,48}
    probs = probs + nn::softmax(model.forward(nn::Value::leaf(view)), 1).data();
```

`RandomCrop` and `RandomErasing` act on `{C,H,W}` samples. The ten views are the four corner crops and the centre crop plus their mirrors, so the test crop size must match the training crop. `ReduceLROnPlateau.step` takes the epoch's metric (use `PlateauMode::Max` for an accuracy or F1); `weight_decay` in `SGD` is L2 added to the gradient, as in PyTorch. Compiled and run against a small synthetic dataset before release.

## Training on FER-2013

`apps/fer2013/train_fer2013.cpp` trains a four-stage Conv-BatchNorm-ReLU network (32 to 256 channels, feature dropout, global average pooling, a dense head) on the FER-2013 facial-expression CSV (columns `emotion,pixels,Usage`). `Training` rows train, `PublicTest` rows pick the best epoch by macro-F1, and `PrivateTest` rows give the final classification report and confusion matrix. FER-2013 is not part of this repository and is not downloaded: obtain `fer2013.csv` separately, for example from [Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013).

```bash
./build/apps/train_fer2013 <fer2013.csv> [--epochs N] [--batch N] [--lr X] [--seed N] [--out ckpt]
                                         [--balance sampler|loss|none] [--smoothing X]
```

Defaults are `--epochs 100 --batch 64 --lr 1e-3 --seed 1 --balance sampler --smoothing 0.1 --out fer2013.ckpt`. `--balance sampler` draws the classes equally often with a weighted `DataLoader` (`balanced_sample_weights`); `loss` weights the loss by class instead. One epoch takes about 3.5 minutes on 8 threads of an i5-12450HX. The `cnn_fer` row of `apps/bench` times one step of a smaller network of the same kind.

## Saving and resuming

`save_checkpoint` writes a model's parameters and buffers, an optimizer's state (learning rate, momentum, velocity or Adam moments) and a step counter of your choosing to one file; `load_checkpoint` restores them.

```cpp
#include "nn/checkpoint.hpp"

nn::save_checkpoint("model.ckpt", model, &optimizer, epoch);

// later, on any machine: rebuild the same model in code, then load into it
int64_t epoch = nn::load_checkpoint("model.ckpt", model, &optimizer);   // resume training
nn::load_checkpoint("model.ckpt", model);                               // inference only
```

Parameters and module buffers are stored by name; loading validates every name and shape first and throws `std::runtime_error` naming what differs, leaving the model untouched. What is not stored (the architecture, schedulers, Dropout's stream, `DataLoader` order) is listed in [architecture.md](architecture.md#checkpoints).

## Using your own data

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

For in-memory data, `TensorDataset` already does this. `MnistDataset` (used by the MNIST example) lives in `apps/mnist/` as a demonstration, not in the core library.
