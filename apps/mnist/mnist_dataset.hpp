/**
 * mnist_dataset.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Dataset implementation loading MNIST from a pair of IDX files
 */

#pragma once

#include "idx_dataset.hpp"
#include "nn/data/dataset.hpp"

namespace nn {

/*
 * MNIST split loaded from an images IDX file and a labels IDX file. Each sample's
 * image gets an explicit leading channel dimension of 1 (grayscale), matching the
 * {N, C, H, W} shape conv2d expects.
 *
 * Attributes:
 * Tensor images_: shape {N, H, W}, pixel values 0..255
 * Tensor labels_: shape {N}, class index 0..9 stored as a float
 */
class MnistDataset : public Dataset {
public:
	/* Public, load a full MNIST split from its image and label IDX files
	 *
	 * Parameters:
	 * string images_path: path to the images IDX file
	 * string labels_path: path to the labels IDX file
	 *
	 * Returns MnistDataset: dataset with images and labels loaded and paired
	 */
	static MnistDataset load(const std::string& images_path, const std::string& labels_path);

	size_t size() const override;
	std::pair<Tensor, Tensor> get(size_t index) const override;

private:
	MnistDataset(Tensor images, Tensor labels);
	Tensor images_;
	Tensor labels_;
};

}  // namespace nn
