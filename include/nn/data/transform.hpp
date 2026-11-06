/**
 * transform.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Sample transforms (normalise, flip, shift, crop, affine, ten-crop, compose) and a Dataset wrapper that applies them
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
 * Crops a {C,H,W} sample to a square at a uniformly random offset. Owns
 * derive_generator("transform.crop").
 *
 * Attributes:
 * size_t size_: side of the crop in pixels
 * mt19937 rng_: offset stream
 */
class RandomCrop : public Transform {
public:
	/* Public, build a cropper
	 *
	 * Parameters:
	 * size_t size: side of the square crop, > 0
	 *
	 * Returns RandomCrop: none (constructor)
	 */
	explicit RandomCrop(size_t size);

	/* Public, crop one sample
	 *
	 * Parameters:
	 * const Tensor& sample: shape {C, H, W}
	 *
	 * Returns Tensor: shape {C, size, size}; throws TensorShapeError if the rank is not 3,
	 * std::invalid_argument if size is 0 or exceeds H or W
	 */
	Tensor operator()(const Tensor& sample) override;

private:
	size_t size_;
	std::mt19937 rng_;
};

/*
 * With probability p each, independently: rotates about the image centre, translates by a
 * fraction of the height and width, and scales about the centre. The drawn components are
 * composed into one inverse mapping and resampled bilinearly with zeros outside the source.
 * If every component is the identity the sample is returned as an exact copy. Owns
 * derive_generator("transform.affine").
 *
 * Attributes:
 * float max_degrees_, max_shift_, max_scale_, p_: ranges and per-component probability
 * mt19937 rng_: parameter stream
 */
class RandomAffine : public Transform {
public:
	/* Public, build an affine augmenter
	 *
	 * Parameters:
	 * float max_degrees: rotation drawn uniformly from [-max_degrees, max_degrees], finite and >= 0
	 * float max_shift: translation fraction drawn from [-max_shift, max_shift], in [0, 1]
	 * float max_scale: scale drawn from [1 - max_scale, 1 + max_scale], in [0, 1)
	 * float p: probability of applying each component, in [0, 1]
	 *
	 * Returns RandomAffine: none (constructor); throws std::invalid_argument for an out-of-range argument
	 */
	RandomAffine(float max_degrees, float max_shift, float max_scale, float p = 0.5f);

	/* Public, transform one sample
	 *
	 * Parameters:
	 * const Tensor& sample: shape {C, H, W}
	 *
	 * Returns Tensor: same shape; throws TensorShapeError if the rank is not 3
	 */
	Tensor operator()(const Tensor& sample) override;

private:
	float max_degrees_;
	float max_shift_;
	float max_scale_;
	float p_;
	std::mt19937 rng_;
};

/*
 * With probability p, sets one random rectangle of a {C,H,W} sample to a constant in every
 * channel. Up to 10 placement attempts each draw a target area fraction uniform in
 * [min_area, max_area] (up to rounding of h and w, the erased area can exceed max_area * H * W by a few
 * pixels at extreme ratios) and an aspect ratio h/w log-uniform in [min_ratio, max_ratio], giving
 * h = round(sqrt(area * ratio)) and w = round(sqrt(area / ratio)) with area = fraction * H * W;
 * the first attempt with 1 <= h <= H and 1 <= w <= W is placed uniformly inside the image. If
 * none fits the sample is returned unchanged. Every call draws the same 41 numbers (the
 * decision plus four per attempt, all drawn up front) whatever p is and whether or not an
 * attempt fits, so the stream advances identically. Owns derive_generator("transform.erasing").
 *
 * Attributes:
 * float p_, min_area_, max_area_, min_ratio_, max_ratio_, value_: parameters as constructed
 * mt19937 rng_: parameter stream
 */
class RandomErasing : public Transform {
public:
	/* Public, build an eraser
	 *
	 * Parameters:
	 * float p: probability of erasing, in [0, 1]
	 * float min_area: smallest erased fraction of the image, > 0
	 * float max_area: largest erased fraction, in [min_area, 1]
	 * float min_ratio: smallest aspect ratio h/w, > 0
	 * float max_ratio: largest aspect ratio, >= min_ratio
	 * float value: fill value
	 *
	 * Returns RandomErasing: none (constructor); throws std::invalid_argument for an out-of-range argument
	 */
	explicit RandomErasing(float p = 0.5f, float min_area = 0.02f, float max_area = 0.33f, float min_ratio = 0.3f,
	                       float max_ratio = 3.3f, float value = 0.0f);

	/* Public, erase a rectangle of one sample
	 *
	 * Parameters:
	 * const Tensor& sample: shape {C, H, W}
	 *
	 * Returns Tensor: same shape; throws TensorShapeError if the rank is not 3
	 */
	Tensor operator()(const Tensor& sample) override;

private:
	float p_;
	float min_area_;
	float max_area_;
	float min_ratio_;
	float max_ratio_;
	float value_;
	std::mt19937 rng_;
};

/**
 * Ten test-time views of a {C,H,W} sample: the top-left, top-right, bottom-left, bottom-right
 * and centre crops, then the horizontal mirror of each in the same order. The centre offset is
 * floor((H - size) / 2) and likewise for W.
 *
 * Parameters:
 * const Tensor& sample: shape {C, H, W}
 * size_t size: side of each square crop, > 0
 *
 * Returns vector<Tensor>: ten tensors {C, size, size}; throws TensorShapeError if the rank is
 * not 3, std::invalid_argument if size is 0 or exceeds H or W
 */
std::vector<Tensor> ten_crop(const Tensor& sample, size_t size);

/**
 * ten_crop applied to every sample of a batch; view k stacks view k of each sample.
 *
 * Parameters:
 * const Tensor& batch: shape {N, C, H, W}
 * size_t size: side of each square crop, > 0
 *
 * Returns vector<Tensor>: ten tensors {N, C, size, size}; throws TensorShapeError if the rank
 * is not 4, std::invalid_argument if size is 0 or exceeds H or W
 */
std::vector<Tensor> ten_crop_batch(const Tensor& batch, size_t size);

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
