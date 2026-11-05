/**
 * train_fer2013.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a small CNN on the FER-2013 CSV and reports validation accuracy per epoch
 */

#include "nn/checkpoint.hpp"
#include "nn/data/dataloader.hpp"
#include "nn/data/fer2013.hpp"
#include "nn/data/transform.hpp"
#include "nn/metrics.hpp"
#include "nn/nn/activation.hpp"
#include "nn/nn/batchnorm.hpp"
#include "nn/nn/conv2d.hpp"
#include "nn/nn/dropout.hpp"
#include "nn/nn/linear.hpp"
#include "nn/nn/loss.hpp"
#include "nn/nn/pool2d.hpp"
#include "nn/nn/sequential.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/clip.hpp"
#include "nn/optim/scheduler.hpp"
#include "nn/random.hpp"

#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;

/**
 * Prints the command-line usage to stderr
 *
 * Parameters:
 * const char* prog: program name
 *
 * Returns void
 */
static void usage(const char* prog) {
	std::fprintf(stderr, "usage: %s <fer2013.csv> [--epochs N] [--batch N] [--lr X] [--seed N] [--out ckpt]\n", prog);
	std::fprintf(stderr, "  csv columns: emotion,pixels,Usage (Training rows train, PublicTest rows validate)\n");
	std::fprintf(stderr, "  defaults: --epochs 20 --batch 64 --lr 1e-3 --seed 1 --out fer2013.ckpt\n");
}

/**
 * Builds Conv-BN-ReLU blocks with max pooling, global average pooling, dropout and a 7-way classifier
 *
 * Parameters: none
 *
 * Returns Sequential: the untrained model
 */
static std::unique_ptr<Sequential> build_model() {
	auto m = std::make_unique<Sequential>();
	size_t in = 1;
	for (size_t out : {16, 32, 64}) {
		m->add(std::make_unique<Conv2d>(in, out, 3, 1, 1));
		m->add(std::make_unique<BatchNorm2d>(out));
		m->add(std::make_unique<ReLU>());
		if (out != 64) m->add(std::make_unique<MaxPool2d>(2, 2));
		in = out;
	}
	m->add(std::make_unique<GlobalAvgPool2d>());
	m->add(std::make_unique<Dropout>(0.3f));
	m->add(std::make_unique<Linear>(64, 7));
	return m;
}

/**
 * Predicted and true classes over a dataset, with the model in eval mode
 *
 * Parameters:
 * Sequential& model: network to evaluate
 * DataLoader& loader: unshuffled loader over the evaluation set
 * std::vector<size_t>& y_true: receives the true class of every sample
 * std::vector<size_t>& y_pred: receives the predicted class of every sample
 *
 * Returns void: none
 */
static void predict(Sequential& model, DataLoader& loader, std::vector<size_t>& y_true, std::vector<size_t>& y_pred) {
	model.eval();
	NoGradGuard no_grad;
	y_true.clear();
	y_pred.clear();
	for (size_t b = 0; b < loader.num_batches(); b++) {
		auto [x, y] = loader.get_batch(b);
		std::vector<size_t> p = argmax_rows(model.forward(x).data());
		std::vector<size_t> t = to_labels(y);
		y_pred.insert(y_pred.end(), p.begin(), p.end());
		y_true.insert(y_true.end(), t.begin(), t.end());
	}
}

/**
 * Trains the FER-2013 CNN, printing loss and validation accuracy per epoch and saving a checkpoint
 *
 * Parameters:
 * int argc: Argument count
 * char** argv: argv[1] is the CSV path, followed by optional --epochs, --batch, --lr, --seed, --out
 *
 * Returns int: 0 on success, 1 on bad usage, 2 on a runtime failure
 */
int main(int argc, char** argv) {
	if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
		usage(argv[0]);
		return 1;
	}
	std::string csv = argv[1], out_path = "fer2013.ckpt";
	size_t epochs = 20, batch = 64;
	float lr = 1e-3f;
	unsigned long long seed = 1;
	try {
		for (int i = 2; i < argc; i++) {
			std::string a = argv[i];
			if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
			std::string v = argv[++i];
			size_t used = 0;
			if (a != "--out" && !v.empty() && v[0] == '-' && a != "--lr") throw std::invalid_argument("bad value for " + a + ": " + v);
			if (a == "--epochs") epochs = std::stoul(v, &used);
			else if (a == "--batch") batch = std::stoul(v, &used);
			else if (a == "--lr") lr = std::stof(v, &used);
			else if (a == "--seed") seed = std::stoull(v, &used);
			else if (a == "--out") { out_path = v; used = v.size(); }
			else throw std::invalid_argument("unknown option " + a);
			if (used != v.size()) throw std::invalid_argument("bad value for " + a + ": " + v);
		}
		if (epochs == 0 || batch == 0 || !(lr > 0.0f)) throw std::invalid_argument("--epochs, --batch and --lr must be positive");
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		usage(argv[0]);
		return 1;
	}

	try {
		manual_seed(seed);
		Fer2013Dataset train_raw(csv, "Training"), val_raw(csv, "PublicTest");
		if (train_raw.size() == 0 || val_raw.size() == 0) {
			std::fprintf(stderr, "error: need Training and PublicTest rows in %s\n", csv.c_str());
			return 2;
		}
		auto [mean, sd] = compute_mean_std(train_raw);
		if (!(sd > 0.0f)) sd = 1.0f;

		// Augmentation only on Training rows; both sets are normalised with the Training statistics
		std::vector<std::unique_ptr<Transform>> aug;
		aug.push_back(std::make_unique<RandomHorizontalFlip>(0.5f));
		aug.push_back(std::make_unique<RandomShift>(3));
		aug.push_back(std::make_unique<Normalize>(mean, sd));
		TransformedDataset train_set(train_raw, std::make_shared<Compose>(std::move(aug)));
		TransformedDataset val_set(val_raw, std::make_shared<Normalize>(mean, sd));

		// Inverse-frequency class weights; absent classes get weight 0
		std::vector<size_t> counts = train_raw.class_counts();
		Tensor weights({counts.size()});
		for (size_t c = 0; c < counts.size(); c++) {
			weights.data()[c] = counts[c] ? static_cast<float>(train_raw.size()) / (counts.size() * static_cast<float>(counts[c])) : 0.0f;
		}

		auto model = build_model();
		AdamW opt(model->parameters(), lr);
		DataLoader train_loader(train_set, batch, true), val_loader(val_set, batch, false);
		size_t total_steps = epochs * train_loader.num_batches();
		size_t warmup = total_steps / 10;
		LinearWarmup sched(opt, std::make_unique<CosineAnnealingLR>(opt, total_steps - warmup), warmup);

		std::vector<size_t> y_true, y_pred;
		for (size_t epoch = 0; epoch < epochs; epoch++) {
			model->train();
			train_loader.reset_epoch();
			float epoch_loss = 0.0f;
			for (size_t b = 0; b < train_loader.num_batches(); b++) {
				auto [x, y] = train_loader.get_batch(b);
				Value loss = cross_entropy_loss(model->forward(x), y, weights);
				opt.zero_grad();
				loss.backward();
				clip_grad_norm(model->parameters(), 1.0f);
				opt.step();
				sched.step();
				epoch_loss += loss.data().get({0});
			}
			predict(*model, val_loader, y_true, y_pred);
			std::printf("epoch %zu loss %.4f val accuracy %.2f%% macro-F1 %.4f\n", epoch,
			            epoch_loss / static_cast<float>(train_loader.num_batches()),
			            100.0f * accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, 7, Average::Macro));
			std::fflush(stdout);
		}
		std::printf("\nvalidation report after the last epoch:\n%s", classification_report(
		    y_true, y_pred, 7, {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"}).c_str());
		save_checkpoint(out_path, *model, &opt, static_cast<int64_t>(total_steps));
		std::printf("saved checkpoint to %s\n", out_path.c_str());
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		return 2;
	}
	return 0;
}
