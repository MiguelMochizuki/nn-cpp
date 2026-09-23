/**
 * sequential.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Container that chains layers in order
 */

#include "nn/nn/sequential.hpp"

namespace nn {

void Sequential::add(std::unique_ptr<Module> layer) { layers_.push_back(std::move(layer)); }

Value Sequential::forward(const Value& input) {
	Value current = input;
	for (auto& layer : layers_) {
		current = layer->forward(current);
	}
	return current;
}

std::vector<Value> Sequential::parameters() const {
	std::vector<Value> all_params;
	for (const auto& layer : layers_) {
		std::vector<Value> layer_params = layer->parameters();
		all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
	}
	return all_params;
}

}  // namespace nn
