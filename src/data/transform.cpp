/**
 * transform.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Sample transforms (normalise, flip, shift, crop, affine, erasing, ten-crop, compose) and a Dataset wrapper that applies them
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

RandomCrop::RandomCrop(size_t size) : size_(size), rng_(derive_generator("transform.crop")) {
	if (size == 0) throw std::invalid_argument("RandomCrop: size must be > 0");
}

namespace {

// Copies the size x size window at (oy, ox) of every channel of one {C,H,W} sample, optionally mirrored
Tensor crop_window(const Tensor& sample, size_t oy, size_t ox, size_t size, bool mirror) {
	size_t C = sample.shape()[0], H = sample.shape()[1], W = sample.shape()[2];
	Tensor out({C, size, size}, 0.0f);
	for (size_t c = 0; c < C; c++)
		for (size_t y = 0; y < size; y++)
			for (size_t x = 0; x < size; x++)
				out.data()[(c * size + y) * size + x] = sample.data()[(c * H + oy + y) * W + ox + (mirror ? size - 1 - x : x)];
	return out;
}

void check_crop(const Tensor& sample, size_t rank, size_t size) {
	if (sample.shape().size() != rank) {
		throw TensorShapeError();
	}
	if (size == 0) {
		throw std::invalid_argument("crop: size must be > 0");
	}
	if (size > sample.shape()[rank - 2] || size > sample.shape()[rank - 1]) {
		throw std::invalid_argument("crop: size exceeds the image");
	}
}

}  // namespace

Tensor RandomCrop::operator()(const Tensor& sample) {
	check_crop(sample, 3, size_);
	std::uint32_t oy = bounded_draw(rng_, static_cast<std::uint32_t>(sample.shape()[1] - size_ + 1));
	std::uint32_t ox = bounded_draw(rng_, static_cast<std::uint32_t>(sample.shape()[2] - size_ + 1));
	return crop_window(sample, oy, ox, size_, false);
}

RandomAffine::RandomAffine(float max_degrees, float max_shift, float max_scale, float p)
	: max_degrees_(max_degrees), max_shift_(max_shift), max_scale_(max_scale), p_(p), rng_(derive_generator("transform.affine")) {
	if (!(max_degrees >= 0.0f && std::isfinite(max_degrees))) throw std::invalid_argument("RandomAffine: max_degrees must be finite and >= 0");
	if (!(max_shift >= 0.0f && max_shift <= 1.0f)) throw std::invalid_argument("RandomAffine: max_shift must be in [0, 1]");
	if (!(max_scale >= 0.0f && max_scale < 1.0f)) throw std::invalid_argument("RandomAffine: max_scale must be in [0, 1)");
	if (!(p >= 0.0f && p <= 1.0f)) throw std::invalid_argument("RandomAffine: p must be in [0, 1]");
}

Tensor RandomAffine::operator()(const Tensor& sample) {
	if (sample.shape().size() != 3) {
		throw TensorShapeError();
	}
	constexpr double kPi = 3.14159265358979323846;
	// always draw all seven numbers so the stream advances identically whatever p is
	bool do_rot = unit(rng_) < p_, do_shift = unit(rng_) < p_, do_scale = unit(rng_) < p_;
	double u_rot = unit(rng_), u_ty = unit(rng_), u_tx = unit(rng_), u_scale = unit(rng_);
	double degrees = do_rot ? (2.0 * u_rot - 1.0) * max_degrees_ : 0.0;
	double ty = do_shift ? (2.0 * u_ty - 1.0) * max_shift_ : 0.0;
	double tx = do_shift ? (2.0 * u_tx - 1.0) * max_shift_ : 0.0;
	double scale = do_scale ? 1.0 + (2.0 * u_scale - 1.0) * max_scale_ : 1.0;
	if (degrees == 0.0 && ty == 0.0 && tx == 0.0 && scale == 1.0) {
		return sample;
	}
	size_t C = sample.shape()[0], H = sample.shape()[1], W = sample.shape()[2];
	double cs = std::cos(degrees * kPi / 180.0), sn = std::sin(degrees * kPi / 180.0);
	double cy = (static_cast<double>(H) - 1.0) / 2.0, cx = (static_cast<double>(W) - 1.0) / 2.0;
	Tensor out(sample.shape(), 0.0f);
	for (size_t y = 0; y < H; y++) {
		for (size_t x = 0; x < W; x++) {
			// inverse of: rotate and scale about the centre, then translate
			double dy = (static_cast<double>(y) - cy - ty * static_cast<double>(H)) / scale;
			double dx = (static_cast<double>(x) - cx - tx * static_cast<double>(W)) / scale;
			double sy = cy + sn * dx + cs * dy;
			double sx = cx + cs * dx - sn * dy;
			double fy = std::floor(sy), fx = std::floor(sx);
			double wy = sy - fy, wx = sx - fx;
			for (int j = 0; j < 2; j++) {
				for (int i = 0; i < 2; i++) {
					double iy = fy + j, ix = fx + i;
					if (iy < 0.0 || iy > static_cast<double>(H) - 1.0 || ix < 0.0 || ix > static_cast<double>(W) - 1.0) continue;
					double weight = (j ? wy : 1.0 - wy) * (i ? wx : 1.0 - wx);
					for (size_t c = 0; c < C; c++) {
						out.data()[(c * H + y) * W + x] +=
							static_cast<float>(weight * sample.data()[(c * H + static_cast<size_t>(iy)) * W + static_cast<size_t>(ix)]);
					}
				}
			}
		}
	}
	return out;
}

RandomErasing::RandomErasing(float p, float min_area, float max_area, float min_ratio, float max_ratio, float value)
	: p_(p), min_area_(min_area), max_area_(max_area), min_ratio_(min_ratio), max_ratio_(max_ratio), value_(value),
	  rng_(derive_generator("transform.erasing")) {
	if (!(p >= 0.0f && p <= 1.0f)) throw std::invalid_argument("RandomErasing: p must be in [0, 1]");
	if (!(min_area > 0.0f && min_area <= max_area && max_area <= 1.0f))
		throw std::invalid_argument("RandomErasing: need 0 < min_area <= max_area <= 1");
	if (!(min_ratio > 0.0f && min_ratio <= max_ratio && std::isfinite(max_ratio)))
		throw std::invalid_argument("RandomErasing: need 0 < min_ratio <= max_ratio, finite");
}

Tensor RandomErasing::operator()(const Tensor& sample) {
	if (sample.shape().size() != 3) {
		throw TensorShapeError();
	}
	constexpr int kAttempts = 10;
	// draw everything up front: 1 decision + 4 numbers per attempt, whatever p is and whether anything fits
	bool erase = unit(rng_) < p_;
	double u[kAttempts][4];
	for (auto& attempt : u)
		for (double& v : attempt) v = unit(rng_);
	Tensor out = sample;
	if (!erase) {
		return out;
	}
	size_t C = sample.shape()[0], H = sample.shape()[1], W = sample.shape()[2];
	double log_lo = std::log(static_cast<double>(min_ratio_)), log_hi = std::log(static_cast<double>(max_ratio_));
	for (const auto& attempt : u) {
		double area = (static_cast<double>(min_area_) + attempt[0] * (static_cast<double>(max_area_) - min_area_)) * static_cast<double>(H) * static_cast<double>(W);
		double ratio = std::exp(log_lo + attempt[1] * (log_hi - log_lo));
		double hd = std::round(std::sqrt(area * ratio)), wd = std::round(std::sqrt(area / ratio));
		if (!(hd >= 1.0 && hd <= static_cast<double>(H) && wd >= 1.0 && wd <= static_cast<double>(W))) continue;
		size_t h = static_cast<size_t>(hd), w = static_cast<size_t>(wd);
		size_t y0 = std::min(static_cast<size_t>(attempt[2] * static_cast<double>(H - h + 1)), H - h);
		size_t x0 = std::min(static_cast<size_t>(attempt[3] * static_cast<double>(W - w + 1)), W - w);
		for (size_t c = 0; c < C; c++)
			for (size_t y = y0; y < y0 + h; y++)
				for (size_t x = x0; x < x0 + w; x++) out.data()[(c * H + y) * W + x] = value_;
		break;
	}
	return out;
}

std::vector<Tensor> ten_crop(const Tensor& sample, size_t size) {
	check_crop(sample, 3, size);
	size_t dy = sample.shape()[1] - size, dx = sample.shape()[2] - size;
	size_t offsets[5][2] = {{0, 0}, {0, dx}, {dy, 0}, {dy, dx}, {dy / 2, dx / 2}};
	std::vector<Tensor> views;
	for (bool mirror : {false, true})
		for (auto& o : offsets) views.push_back(crop_window(sample, o[0], o[1], size, mirror));
	return views;
}

std::vector<Tensor> ten_crop_batch(const Tensor& batch, size_t size) {
	check_crop(batch, 4, size);
	size_t N = batch.shape()[0], C = batch.shape()[1], H = batch.shape()[2], W = batch.shape()[3];
	std::vector<Tensor> views(10, Tensor({N, C, size, size}, 0.0f));
	size_t in_len = C * H * W, out_len = C * size * size;
	for (size_t n = 0; n < N; n++) {
		Tensor sample({C, H, W}, std::vector<float>(batch.data().begin() + n * in_len, batch.data().begin() + (n + 1) * in_len));
		auto crops = ten_crop(sample, size);
		for (size_t v = 0; v < 10; v++) std::copy(crops[v].data().begin(), crops[v].data().end(), views[v].data().begin() + n * out_len);
	}
	return views;
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
