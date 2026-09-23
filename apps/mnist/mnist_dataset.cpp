/**
 * mnist_dataset.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Dataset implementation loading MNIST from a pair of IDX files
 */

#include "mnist_dataset.hpp"

#include <algorithm>

namespace nn {

MnistDataset::MnistDataset(Tensor images, Tensor labels) : images_(std::move(images)), labels_(std::move(labels)) {}

MnistDataset MnistDataset::load(const std::string& images_path, const std::string& labels_path) {
	return MnistDataset(load_idx_ubyte(images_path), load_idx_ubyte(labels_path));
}

size_t MnistDataset::size() const { return images_.shape()[0]; }

std::pair<Tensor, Tensor> MnistDataset::get(size_t index) const {
	size_t rows = images_.shape()[1];
	size_t cols = images_.shape()[2];

	Tensor image({1, rows, cols});
	const float* src = images_.data().data() + index * rows * cols;
	std::copy(src, src + rows * cols, image.data().data());

	Tensor label({1}, labels_.get({index}));
	return {image, label};
}

}  // namespace nn
