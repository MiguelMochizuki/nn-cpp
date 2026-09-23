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
	Tensor images = load_idx_ubyte(images_path);
	Tensor labels = load_idx_ubyte(labels_path);
	if (images.ndim() != 3 || labels.ndim() != 1 || images.shape()[0] != labels.shape()[0]) {
		throw IdxParseError();
	}
	return MnistDataset(std::move(images), std::move(labels));
}

size_t MnistDataset::size() const { return images_.shape()[0]; }

std::pair<Tensor, Tensor> MnistDataset::get(size_t index) const {
	if (index >= size()) {
		throw TensorIndexError();
	}

	size_t rows = images_.shape()[1];
	size_t cols = images_.shape()[2];

	Tensor image({1, rows, cols});
	const float* src = images_.data().data() + index * rows * cols;
	std::copy(src, src + rows * cols, image.data().data());

	Tensor label({}, labels_.get({index}));
	return {image, label};
}

}  // namespace nn
