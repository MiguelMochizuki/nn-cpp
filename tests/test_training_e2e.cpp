/**
 * test_training_e2e.cpp
 * Author: Miguel Mochizuki Silva
 * Description: End-to-end training test on a synthetic "which quadrant is bright" image task
 */

#include "doctest/doctest.h"
#include "nn/checkpoint.hpp"
#include "nn/data/dataloader.hpp"
#include "nn/data/subset.hpp"
#include "nn/data/tensor_dataset.hpp"
#include "nn/data/transform.hpp"
#include "nn/kernels/parallel.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/dropout.hpp"
#include "nn/nn/flatten.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/clip.hpp"
#include "nn/optim/scheduler.hpp"
#include "nn/random.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace nn;

namespace {

const size_t SIDE = 16, CLASSES = 3;

// Fixes the library thread count for one test and restores it afterwards
struct ThreadScope {
	size_t before = kernels::num_threads();
	ThreadScope() { kernels::set_num_threads(2); }
	~ThreadScope() { kernels::set_num_threads(before); }
};

// Class k brightens quadrant k (0 top-left, 1 top-right, 2 bottom-left) over Gaussian noise
TensorDataset make_data(size_t n) {
	std::mt19937 gen(123);
	std::normal_distribution<float> noise(0.0f, 0.3f);
	Tensor x({n, 1, SIDE, SIDE}), y({n});
	for (size_t i = 0; i < n; i++) {
		size_t k = i % CLASSES;
		y.data()[i] = static_cast<float>(k);
		size_t y0 = (k / 2) * (SIDE / 2), x0 = (k % 2) * (SIDE / 2);
		for (size_t r = 0; r < SIDE; r++) {
			for (size_t c = 0; c < SIDE; c++) {
				bool lit = r >= y0 && r < y0 + SIDE / 2 && c >= x0 && c < x0 + SIDE / 2;
				x.data()[i * SIDE * SIDE + r * SIDE + c] = noise(gen) + (lit ? 1.0f : 0.0f);
			}
		}
	}
	return TensorDataset(x, y);
}

std::unique_ptr<Sequential> make_model(float dropout) {
	auto m = std::make_unique<Sequential>();
	m->add(std::make_unique<Conv2d>(1, 8, 3, 1, 1));
	m->add(std::make_unique<BatchNorm2d>(8));
	m->add(std::make_unique<ReLU>());
	m->add(std::make_unique<MaxPool2d>(2, 2));
	m->add(std::make_unique<Conv2d>(8, 16, 3, 1, 1));
	m->add(std::make_unique<BatchNorm2d>(16));
	m->add(std::make_unique<ReLU>());
	m->add(std::make_unique<MaxPool2d>(2, 2));
	m->add(std::make_unique<Flatten>());
	m->add(std::make_unique<Dropout>(dropout));
	m->add(std::make_unique<Linear>(16 * 4 * 4, CLASSES));
	return m;
}

// One optimisation step: loss, backward, clip, AdamW, scheduler
void train_step(Sequential& model, AdamW& opt, LRScheduler& sched, const Value& x, const Tensor& y) {
	Value loss = cross_entropy_loss(model.forward(x), y, Tensor({CLASSES}, 1.0f));
	opt.zero_grad();
	loss.backward();
	clip_grad_norm(model.parameters(), 1.0f);
	opt.step();
	sched.step();
}

std::vector<float> flat_weights(const Sequential& model) {
	std::vector<float> w;
	for (Value p : model.parameters()) w.insert(w.end(), p.data().data().begin(), p.data().data().end());
	return w;
}

// Trains from manual_seed(seed) through random_split, transforms, BN, dropout, scheduler and clipping
float train_run(unsigned seed, size_t epochs, std::vector<float>* weights) {
	manual_seed(seed);
	TensorDataset data = make_data(300);
	std::vector<Subset> parts = random_split(data, {0.8, 0.2}, seed);
	auto [mean, sd] = compute_mean_std(parts[0]);
	std::vector<std::unique_ptr<Transform>> aug;
	aug.push_back(std::make_unique<RandomShift>(1));
	aug.push_back(std::make_unique<Normalize>(mean, sd));
	TransformedDataset train_set(parts[0], std::make_shared<Compose>(std::move(aug)));
	TransformedDataset val_set(parts[1], std::make_shared<Normalize>(mean, sd));

	auto model = make_model(0.2f);
	AdamW opt(model->parameters(), 3e-3f);
	DataLoader train_loader(train_set, 32, true), val_loader(val_set, 32, false);
	size_t total = epochs * train_loader.num_batches(), warm = total / 10;
	LinearWarmup sched(opt, std::make_unique<CosineAnnealingLR>(opt, total - warm), warm);
	for (size_t e = 0; e < epochs; e++) {
		model->train();
		train_loader.reset_epoch();
		for (size_t b = 0; b < train_loader.num_batches(); b++) {
			auto [x, y] = train_loader.get_batch(b);
			train_step(*model, opt, sched, x, y);
		}
	}
	model->eval();
	size_t correct = 0, n = 0;
	for (size_t b = 0; b < val_loader.num_batches(); b++) {
		auto [x, y] = val_loader.get_batch(b);
		Value logits = model->forward(x);
		for (size_t i = 0; i < y.size(); i++) {
			size_t best = 0;
			for (size_t k = 1; k < CLASSES; k++) {
				if (logits.data().get({i, k}) > logits.data().get({i, best})) best = k;
			}
			correct += static_cast<float>(best) == y.data()[i];
			n++;
		}
	}
	if (weights) *weights = flat_weights(*model);
	return static_cast<float>(correct) / static_cast<float>(n);
}

}  // namespace

TEST_CASE("e2e: CNN stack learns the bright-quadrant task and is deterministic") {
	ThreadScope threads;
	std::vector<float> w1, w2;
	float acc = train_run(5, 8, &w1);
	CHECK(acc > 0.9f);
	train_run(5, 8, &w2);
	CHECK(w1 == w2);
}

TEST_CASE("e2e: resuming from a checkpoint continues identically") {
	ThreadScope threads;
	const std::string path = (std::filesystem::temp_directory_path() / "nn_e2e_resume.ckpt").string();
	TensorDataset data = make_data(64);
	auto batch = [&](size_t b) {
		Tensor x({16, 1, SIDE, SIDE}), y({16});
		for (size_t i = 0; i < 16; i++) {
			auto [xi, yi] = data.get(b * 16 + i);
			std::copy(xi.data().begin(), xi.data().end(), x.data().begin() + i * SIDE * SIDE);
			y.data()[i] = yi.data()[0];
		}
		return std::make_pair(Value::leaf(x), y);
	};
	// Dropout p = 0 keeps the run free of random state that a checkpoint does not hold
	auto build = [](std::unique_ptr<Sequential>& m, std::unique_ptr<AdamW>& o, std::unique_ptr<LinearWarmup>& s) {
		manual_seed(3);
		m = make_model(0.0f);
		o = std::make_unique<AdamW>(m->parameters(), 3e-3f);
		s = std::make_unique<LinearWarmup>(*o, std::make_unique<CosineAnnealingLR>(*o, 20), 4);
	};
	std::unique_ptr<Sequential> ma, mb;
	std::unique_ptr<AdamW> oa, ob;
	std::unique_ptr<LinearWarmup> sa, sb;
	build(ma, oa, sa);
	for (size_t i = 0; i < 3; i++) { auto [x, y] = batch(i); train_step(*ma, *oa, *sa, x, y); }
	save_checkpoint(path, *ma, oa.get(), static_cast<int64_t>(sa->step_count()));
	float lr_at_save = oa->lr();
	for (size_t i = 3; i < 5; i++) { auto [x, y] = batch(i % 4); train_step(*ma, *oa, *sa, x, y); }

	build(mb, ob, sb);
	int64_t step = load_checkpoint(path, *mb, ob.get());
	CHECK(ob->lr() == lr_at_save);
	sb->set_step(static_cast<size_t>(step));
	CHECK(ob->lr() == lr_at_save);
	for (size_t i = 3; i < 5; i++) { auto [x, y] = batch(i % 4); train_step(*mb, *ob, *sb, x, y); }
	CHECK(flat_weights(*ma) == flat_weights(*mb));
	CHECK(oa->lr() == ob->lr());
	std::filesystem::remove(path);
}
