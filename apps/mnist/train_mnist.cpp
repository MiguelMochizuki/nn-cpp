/**
 * train_mnist.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a small CNN on MNIST and reports test-set accuracy
 */

#include "mnist_dataset.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/nn/flatten.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/sgd.hpp"
#include "nn/data/dataloader.hpp"

#include <cstdio>
#include <cmath>
#include <memory>
#include <string>

using namespace nn;

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <mnist_dir>\n", argv[0]);
		std::fprintf(stderr, "  mnist_dir must contain train-images-idx3-ubyte, train-labels-idx1-ubyte,\n");
		std::fprintf(stderr, "  t10k-images-idx3-ubyte, and t10k-labels-idx1-ubyte\n");
		return 1;
	}
	std::string dir = argv[1];

	MnistDataset train_set = MnistDataset::load(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte");
	MnistDataset test_set = MnistDataset::load(dir + "/t10k-images-idx3-ubyte", dir + "/t10k-labels-idx1-ubyte");

	Sequential model;
	model.add(std::make_unique<Conv2d>(1, 8, 3, 1, 1));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<MaxPool2d>(2, 2));
	model.add(std::make_unique<Conv2d>(8, 16, 3, 1, 1));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<MaxPool2d>(2, 2));
	model.add(std::make_unique<Flatten>());
	model.add(std::make_unique<Linear>(16 * 7 * 7, 128));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(128, 10));

	SGD optimizer(model.parameters(), 0.05f);

	DataLoader train_loader(train_set, 32, true);
	const int epochs = 3;
	for (int epoch = 0; epoch < epochs; epoch++) {
		train_loader.reset_epoch();
		float epoch_loss = 0.0f;
		for (size_t b = 0; b < train_loader.num_batches(); b++) {
			auto [x_batch, y_batch] = train_loader.get_batch(b);
			Value x_norm = scale(x_batch, 1.0f / 255.0f);
			Value logits = model.forward(x_norm);
			Tensor targets = y_batch.reshape({y_batch.shape()[0]});
			Value loss = cross_entropy_loss(logits, targets);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

			epoch_loss += loss.data().get({0});
		}
		std::printf("epoch %d avg loss %.4f\n", epoch, epoch_loss / static_cast<float>(train_loader.num_batches()));
	}

	DataLoader test_loader(test_set, 32, false);
	size_t correct = 0;
	size_t total = 0;
	for (size_t b = 0; b < test_loader.num_batches(); b++) {
		auto [x_batch, y_batch] = test_loader.get_batch(b);
		Value x_norm = scale(x_batch, 1.0f / 255.0f);
		Value logits = model.forward(x_norm);

		size_t batch_size = logits.data().shape()[0];
		size_t num_classes = logits.data().shape()[1];
		for (size_t i = 0; i < batch_size; i++) {
			size_t best_class = 0;
			float best_score = logits.data().get({i, 0});
			for (size_t c = 1; c < num_classes; c++) {
				float score = logits.data().get({i, c});
				if (score > best_score) {
					best_score = score;
					best_class = c;
				}
			}
			size_t true_class = static_cast<size_t>(std::round(y_batch.get({i, 0})));
			if (best_class == true_class) correct++;
			total++;
		}
	}

	std::printf("test accuracy: %.2f%% (%zu/%zu)\n",
	            100.0f * static_cast<float>(correct) / static_cast<float>(total), correct, total);
	return 0;
}
