/**
 * fer2013.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Dataset reading the FER-2013 facial-expression CSV (emotion,pixels,Usage)
 */

#pragma once

#include "nn/data/dataset.hpp"

#include <string>
#include <vector>

namespace nn {

/*
 * FER-2013 loaded fully into memory. Each pixels field holds image_size^2 space-separated
 * integers 0-255, scaled to [0, 1]. No data is bundled with the library.
 *
 * Memory: pixels are stored as float, about 4 bytes per pixel, so the full 35k-row, 48x48 file
 * takes roughly 320 MB. Parser limits: multi-line quoted fields and a UTF-8 BOM are not supported.
 *
 * Attributes:
 * size_t image_size_: side length of the square images
 * vector<float> pixels_: all images, row-major, image_size_^2 floats each
 * vector<float> labels_: emotion class index per image
 */
class Fer2013Dataset : public Dataset {
public:
	/* Public, parse a FER-2013 CSV
	 *
	 * Parameters:
	 * const std::string& csv_path: path to the CSV; columns are located by header name
	 * const std::string& usage: keep only rows whose Usage equals this ("Training", "PublicTest",
	 * "PrivateTest"); empty keeps every row
	 * size_t image_size: image side length, 48 for the standard file
	 *
	 * Returns Fer2013Dataset: none (constructor); throws std::runtime_error with the line number and
	 * reason for a missing file, missing column, wrong pixel count, pixel outside 0-255 or bad emotion
	 */
	explicit Fer2013Dataset(const std::string& csv_path, const std::string& usage = "", size_t image_size = 48);

	size_t size() const override { return labels_.size(); }

	/* Public, fetch one image and its class
	 *
	 * Parameters:
	 * size_t index: sample index, 0-based
	 *
	 * Returns pair<Tensor, Tensor>: image {1, image_size, image_size} in [0, 1], scalar class index;
	 * throws TensorIndexError if index >= size()
	 */
	std::pair<Tensor, Tensor> get(size_t index) const override;

	/* Public, number of samples per emotion class
	 *
	 * Parameters: none
	 *
	 * Returns vector<size_t>: 7 counts, index = emotion; usable to build class weights
	 */
	std::vector<size_t> class_counts() const;

private:
	size_t image_size_;
	std::vector<float> pixels_;
	std::vector<float> labels_;
};

}  // namespace nn
