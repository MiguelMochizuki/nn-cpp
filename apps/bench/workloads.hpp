/**
 * workloads.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Seeded synthetic training workloads for the benchmark harness
 */

#pragma once

#include "nn/autograd/value.hpp"
#include "nn/nn/module.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/optimizer.hpp"
#include "nn/tensor.hpp"

#include <memory>
#include <string>
#include <vector>

namespace bench {

/**
 * One benchmark workload: a model, its optimizer and a single fixed batch
 *
 * Members:
 * string name: workload name
 * shared_ptr<Sequential> model: network under test
 * shared_ptr<Optimizer> optimizer: optimizer bound to the model's parameters
 * Tensor x, y: one batch of inputs and targets
 * bool classification: true -> cross_entropy_loss with y of shape {N}; false -> mse_loss with y shaped like the output
 */
struct Workload {
	std::string name;
	std::shared_ptr<nn::Sequential> model;
	std::shared_ptr<nn::Optimizer> optimizer;
	nn::Tensor x;
	nn::Tensor y;
	bool classification;
};

/**
 * Builds a workload with every parameter and the data drawn from a seeded generator
 *
 * Parameters:
 * string name: one of workload_names()
 * unsigned seed: seed for parameters, inputs and the teacher
 *
 * Returns Workload: ready to train; throws std::invalid_argument on an unknown name
 */
Workload make_workload(const std::string& name, unsigned seed);

/**
 * Overwrites every parameter with uniform draws in [-1/sqrt(fan_in), 1/sqrt(fan_in)] from
 * mt19937(seed); rank-1 parameters (biases) are set to zero. Replaces the library's unseeded init
 *
 * Parameters:
 * Module& model: model whose parameters are overwritten
 * unsigned seed: generator seed
 *
 * Returns void
 */
void reseed_parameters(nn::Module& model, unsigned seed);

/**
 * Names of all available workloads
 *
 * Parameters: none
 *
 * Returns vector<string>: xor, mlp_wide, cnn_mnist, cnn_large, mlp_adam, cnn_fer
 */
std::vector<std::string> workload_names();

}  // namespace bench
