/**
 * dataset.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Abstract interface for indexable, sized data sources
 */

#pragma once

#include "nn/tensor.hpp"

#include <utility>

namespace nn {

/*
 * Base interface any data source implements to be usable by DataLoader. No
 * assumption about task (classification/regression) or storage format, and no
 * concrete implementation ships in this library — users implement their own.
 *
 * Attributes: none (interface)
 */
class Dataset {
public:
	virtual ~Dataset() = default;

	/* Public, number of samples in this dataset
	 *
	 * Parameters: none
	 *
	 * Returns size_t: sample count
	 */
	virtual size_t size() const = 0;

	/* Public, fetch one sample
	 *
	 * Parameters:
	 * size_t index: sample index, 0-based
	 *
	 * Returns pair<Tensor, Tensor>: input tensor, target tensor
	 */
	virtual std::pair<Tensor, Tensor> get(size_t index) const = 0;
};

}  // namespace nn
