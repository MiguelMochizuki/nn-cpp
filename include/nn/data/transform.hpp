/**
 * transform.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Sample transforms (normalise, flip, shift, compose) and a Dataset wrapper that applies them
 */

#pragma once

#include "nn/data/dataset.hpp"

#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace nn {

/*
 * A function from one input sample to another. operator() is non-const because random
 * transforms advance their generator.
 *
 * Attributes: none (interface)
 */
class Transform {
public:
	virtual ~Transform() = default;

	/* Public, transform one sample
	 *
	 * Parameters:
	 * const Tensor& sample: input sample
	 *
	 * Returns Tensor: transformed sample
	 */
	virtual Tensor operator()(const Tensor& sample) = 0;
};

/*
 * Applies several transforms in order.
 *
 * Attributes:
 * vector<unique_ptr<Transform>> steps_: transforms, first applied first
 */
class Compose : public Transform {
public:
	/* Public, build a pipeline
	 *
	 * Parameters:
	 * vector<unique_ptr<Transform>> steps: transforms to apply in order
	 *
	 * Returns Compose: none (constructor); throws std::invalid_argument if any step is null
	 */
	explicit Compose(std::vector<std::unique_ptr<Transform>> steps);

	Tensor operator()(const Tensor& sample) override;

private:
	std::vector<std::unique_ptr<Transform>> steps_;
};

/*
 * (x - mean) / std with scalar statistics.
 *
 * Attributes:
 * float mean_, std_: statistics
 */
class Normalize : public Transform {
public:
	/* Public, build a normaliser
	 *
	 * Parameters:
	 * float mean: value subtracted from every element
	 * float std: divisor, must be > 0
	 *
	 * Returns Normalize: none (constructor); throws std::invalid_argument if std <= 0
	 */
	Normalize(float mean, float std);

	Tensor operator()(const Tensor& sample) override;

private:
	float mean_;
	float std_;
};

/*
 * Mirrors the last axis with probability p; a sample with a zero-length last axis is returned
 * unchanged. Owns derive_generator("transform.hflip").
 *
 * Attributes:
 * float p_: flip probability
 * mt19937 rng_: decision stream
 */
class RandomHorizontalFlip : public Transform {
public:
	/* Public, build a flipper
	 *
	 * Parameters:
	 * float p: probability of flipping, in [0, 1]
	 *
	 * Returns RandomHorizontalFlip: none (constructor); throws std::invalid_argument if p is outside [0, 1]
	 */
	explicit RandomHorizontalFlip(float p = 0.5f);

	Tensor operator()(const Tensor& sample) override;

private:
	float p_;
	std::mt19937 rng_;
};

/*
 * Translates a {C,H,W} sample by an integer in [-max, max] on each spatial axis, filling
 * with zeros. Owns derive_generator("transform.shift").
 *
 * Attributes:
 * size_t max_shift_: largest shift in pixels
 * mt19937 rng_: shift stream
 */
class RandomShift : public Transform {
public:
	/* Public, build a shifter
	 *
	 * Parameters:
	 * size_t max_shift: largest absolute shift per axis
	 *
	 * Returns RandomShift: none (constructor)
	 */
	explicit RandomShift(size_t max_shift);

	/* Public, shift one sample
	 *
	 * Parameters:
	 * const Tensor& sample: shape {C, H, W}
	 *
	 * Returns Tensor: same shape, shifted; throws TensorShapeError if the rank is not 3
	 */
	Tensor operator()(const Tensor& sample) override;

private:
	size_t max_shift_;
	std::mt19937 rng_;
};

/*
 * View of a Dataset whose inputs go through a transform; targets are untouched.
 * get() is not thread-safe because the transform may advance a generator; DataLoader
 * calls it serially.
 *
 * Attributes:
 * const Dataset* dataset_: underlying dataset, not owned
 * shared_ptr<Transform> transform_: applied to inputs
 */
class TransformedDataset : public Dataset {
public:
	/* Public, wrap a dataset
	 *
	 * Parameters:
	 * const Dataset& dataset: underlying dataset; must outlive this object
	 * shared_ptr<Transform> input_transform: transform applied to each input, non-null
	 *
	 * Returns TransformedDataset: none (constructor); throws std::invalid_argument if the transform is null
	 */
	TransformedDataset(const Dataset& dataset, std::shared_ptr<Transform> input_transform);

	size_t size() const override { return dataset_->size(); }

	std::pair<Tensor, Tensor> get(size_t index) const override;

private:
	const Dataset* dataset_;
	std::shared_ptr<Transform> transform_;
};

/**
 * Mean and population standard deviation over every element of every input
 *
 * Parameters:
 * const Dataset& dataset: dataset to scan, non-empty
 *
 * Returns pair<float, float>: mean, std; throws std::invalid_argument if the dataset is empty
 */
std::pair<float, float> compute_mean_std(const Dataset& dataset);

}  // namespace nn
