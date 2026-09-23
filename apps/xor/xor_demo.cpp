/**
 * xor_demo.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a small network on the XOR problem using only nn_core
 */

#include "nn/nn/sequential.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/loss.hpp"
#include "nn/optim/sgd.hpp"

#include <cstdio>
#include <memory>

using namespace nn;

int main() {
	Tensor x_data({4, 2}, std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1});
	Tensor y_data({4, 1}, std::vector<float>{0, 1, 1, 0});
	Value x = Value::leaf(x_data, false);
	Value y = Value::leaf(y_data, false);

	Sequential model;
	model.add(std::make_unique<Linear>(2, 8));
	model.add(std::make_unique<ReLU>());
	model.add(std::make_unique<Linear>(8, 1));

	SGD optimizer(model.parameters(), 0.1f);

	const int epochs = 2000;
	for (int epoch = 0; epoch < epochs; epoch++) {
		Value pred = model.forward(x);
		Value loss = mse_loss(pred, y);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		if (epoch % 200 == 0) {
			std::printf("epoch %d loss %.6f\n", epoch, loss.data().get({0}));
		}
	}

	Value final_pred = model.forward(x);
	std::printf("final predictions:\n");
	for (size_t i = 0; i < 4; i++) {
		std::printf("  [%.0f %.0f] -> %.4f (expected %.0f)\n", x.data().get({i, 0}), x.data().get({i, 1}),
		            final_pred.data().get({i, 0}), y.data().get({i, 0}));
	}
	return 0;
}
