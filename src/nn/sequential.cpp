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

std::vector<std::pair<std::string, Value>> Sequential::named_parameters() const {
	std::vector<std::pair<std::string, Value>> all_params;
	for (size_t i = 0; i < layers_.size(); i++) {
		for (auto& entry : layers_[i]->named_parameters()) {
			all_params.emplace_back(std::to_string(i) + "." + entry.first, entry.second);
		}
	}
	return all_params;
}

std::vector<std::pair<std::string, Tensor*>> Sequential::named_buffers() const {
	std::vector<std::pair<std::string, Tensor*>> all;
	for (size_t i = 0; i < layers_.size(); i++) {
		for (auto& entry : layers_[i]->named_buffers()) {
			all.emplace_back(std::to_string(i) + "." + entry.first, entry.second);
		}
	}
	return all;
}

void Sequential::train(bool mode) {
	for (auto& layer : layers_) {
		layer->train(mode);
	}
	training_ = mode;
}

}  // namespace nn
