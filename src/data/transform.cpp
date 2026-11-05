/**
 * transform.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Sample transforms (normalise, flip, shift, compose) and a Dataset wrapper that applies them
 */

#include "nn/data/transform.hpp"
#include "nn/random.hpp"
#include "shuffle.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nn {

namespace {

// Uniform in [0, 1) from 24 raw bits, so decisions do not depend on the standard library's distribution classes
float unit(std::mt19937& rng) { return static_cast<float>(rng() >> 8) / 16777216.0f; }

}  // namespace

Compose::Compose(std::vector<std::unique_ptr<Transform>> steps) : steps_(std::move(steps)) {
	for (const auto& step : steps_) {
		if (!step) {
			throw std::invalid_argument("Compose: step is null");
		}
	}
}

Tensor Compose::operator()(const Tensor& sample) {
	Tensor out = sample;
	for (auto& step : steps_) {
		out = (*step)(out);
	}
	return out;
}

Normalize::Normalize(float mean, float std) : mean_(mean), std_(std) {
	if (!(std > 0.0f)) {
		throw std::invalid_argument("Normalize: std must be > 0");
	}
}

Tensor Normalize::operator()(const Tensor& sample) {
	Tensor out = sample;
	for (float& v : out.data()) {
		v = (v - mean_) / std_;
	}
	return out;
}

RandomHorizontalFlip::RandomHorizontalFlip(float p) : p_(p), rng_(derive_generator("transform.hflip")) {
	if (!(p >= 0.0f && p <= 1.0f)) {
		throw std::invalid_argument("RandomHorizontalFlip: p must be in [0, 1]");
	}
}

Tensor RandomHorizontalFlip::operator()(const Tensor& sample) {
	// always draw so the stream advances once per sample regardless of p
	bool flip = unit(rng_) < p_;
	Tensor out = sample;
	if (!flip || sample.shape().empty() || sample.shape().back() == 0) {
		return out;
	}
	size_t width = sample.shape().back();
	for (size_t row = 0; row < sample.size() / width; row++) {
		std::reverse(out.data().begin() + row * width, out.data().begin() + (row + 1) * width);
	}
	return out;
}

RandomShift::RandomShift(size_t max_shift) : max_shift_(max_shift), rng_(derive_generator("transform.shift")) {}

Tensor RandomShift::operator()(const Tensor& sample) {
	if (sample.shape().size() != 3) {
		throw TensorShapeError();
	}
	std::uint32_t span = static_cast<std::uint32_t>(2 * max_shift_ + 1);
	long dy = static_cast<long>(bounded_draw(rng_, span)) - static_cast<long>(max_shift_);
	long dx = static_cast<long>(bounded_draw(rng_, span)) - static_cast<long>(max_shift_);
	long C = static_cast<long>(sample.shape()[0]);
	long H = static_cast<long>(sample.shape()[1]);
	long W = static_cast<long>(sample.shape()[2]);
	Tensor out(sample.shape(), 0.0f);
	for (long c = 0; c < C; c++) {
		for (long y = 0; y < H; y++) {
			long sy = y - dy;
			if (sy < 0 || sy >= H) continue;
			for (long x = 0; x < W; x++) {
				long sx = x - dx;
				if (sx < 0 || sx >= W) continue;
				out.data()[(c * H + y) * W + x] = sample.data()[(c * H + sy) * W + sx];
			}
		}
	}
	return out;
}

TransformedDataset::TransformedDataset(const Dataset& dataset, std::shared_ptr<Transform> input_transform)
	: dataset_(&dataset), transform_(std::move(input_transform)) {
	if (!transform_) {
		throw std::invalid_argument("TransformedDataset: transform is null");
	}
}

std::pair<Tensor, Tensor> TransformedDataset::get(size_t index) const {
	auto [x, y] = dataset_->get(index);
	return {(*transform_)(x), y};
}

std::pair<float, float> compute_mean_std(const Dataset& dataset) {
	// Welford's update, in double, so large datasets do not lose precision
	double count = 0.0;
	double mean = 0.0;
	double m2 = 0.0;
	for (size_t i = 0; i < dataset.size(); i++) {
		auto sample = dataset.get(i);  // named, so the tensor outlives the loop
		for (float v : sample.first.data()) {
			count += 1.0;
			double delta = v - mean;
			mean += delta / count;
			m2 += delta * (v - mean);
		}
	}
	if (count == 0.0) {
		throw std::invalid_argument("compute_mean_std: dataset has no elements");
	}
	return {static_cast<float>(mean), static_cast<float>(std::sqrt(m2 / count))};
}

}  // namespace nn
