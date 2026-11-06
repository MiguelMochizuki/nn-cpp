/**
 * train_fer2013.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a CNN on the FER-2013 CSV, tracks validation macro-F1 per epoch and reports the best epoch on the test rows
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

#include <chrono>
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
	std::fprintf(stderr, "          [--balance sampler|loss|none] [--smoothing X]\n");
	std::fprintf(stderr, "  csv columns: emotion,pixels,Usage (Training rows train, PublicTest rows pick the best epoch,\n");
	std::fprintf(stderr, "  PrivateTest rows give the final report)\n");
	std::fprintf(stderr, "  --balance: sampler draws classes equally often (default), loss weights the loss by class\n");
	std::fprintf(stderr, "  --out receives the best epoch by validation macro-F1, not the final model\n");
	std::fprintf(stderr, "  defaults: --epochs 100 --batch 64 --lr 1e-3 --seed 1 --balance sampler --smoothing 0.1 --out fer2013.ckpt\n");
}

/**
 * Builds four Conv-BN-ReLU blocks (32, 64, 128, 256 channels) with max pooling and feature dropout between
 * them, global average pooling, and a dense head with dropout before the 7-way classifier
 *
 * Parameters: none
 *
 * Returns Sequential: the untrained model
 */
static std::unique_ptr<Sequential> build_model() {
	auto m = std::make_unique<Sequential>();
	size_t in = 1;
	for (size_t out : {32, 64, 128, 256}) {
		m->add(std::make_unique<Conv2d>(in, out, 3, 1, 1));
		m->add(std::make_unique<BatchNorm2d>(out));
		m->add(std::make_unique<ReLU>());
		if (out != 256) {
			m->add(std::make_unique<MaxPool2d>(2, 2));
			m->add(std::make_unique<Dropout>(0.25f));
		}
		in = out;
	}
	m->add(std::make_unique<GlobalAvgPool2d>());
	m->add(std::make_unique<Dropout>(0.4f));
	m->add(std::make_unique<Linear>(256, 128));
	m->add(std::make_unique<ReLU>());
	m->add(std::make_unique<Dropout>(0.3f));
	m->add(std::make_unique<Linear>(128, 7));
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
 * Trains the FER-2013 CNN, printing loss, validation accuracy and macro-F1 per epoch, keeping the best epoch's
 * checkpoint, and finishing with a classification report and confusion matrix on the PrivateTest rows
 *
 * Parameters:
 * int argc: Argument count
 * char** argv: argv[1] is the CSV path, followed by optional --epochs, --batch, --lr, --seed, --out, --balance, --smoothing
 *
 * Returns int: 0 on success, 1 on bad usage, 2 on a runtime failure
 */
int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: progress lines appear as printed, even into a file (a sized _IOLBF is rejected on Windows)
	if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
		usage(argv[0]);
		return 1;
	}
	std::string csv = argv[1], out_path = "fer2013.ckpt";
	size_t epochs = 100, batch = 64;
	float lr = 1e-3f, smoothing = 0.1f;
	std::string balance = "sampler";
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
			else if (a == "--balance") { balance = v; used = v.size(); }
			else if (a == "--smoothing") smoothing = std::stof(v, &used);
			else throw std::invalid_argument("unknown option " + a);
			if (used != v.size()) throw std::invalid_argument("bad value for " + a + ": " + v);
		}
		if (epochs == 0 || batch == 0 || !(lr > 0.0f)) throw std::invalid_argument("--epochs, --batch and --lr must be positive");
		if (balance != "sampler" && balance != "loss" && balance != "none") throw std::invalid_argument("--balance must be sampler, loss or none");
		if (!(smoothing >= 0.0f && smoothing < 1.0f)) throw std::invalid_argument("--smoothing must be in [0, 1)");
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		usage(argv[0]);
		return 1;
	}

	try {
		manual_seed(seed);
		Fer2013Dataset train_raw(csv, "Training"), val_raw(csv, "PublicTest"), test_raw(csv, "PrivateTest");
		if (train_raw.size() == 0 || val_raw.size() == 0 || test_raw.size() == 0) {
			std::fprintf(stderr, "error: need Training, PublicTest and PrivateTest rows in %s\n", csv.c_str());
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
		TransformedDataset test_set(test_raw, std::make_shared<Normalize>(mean, sd));

		// Class imbalance: draw the classes equally often (sampler), weight the loss (loss) or neither (none).
		// Only one of the two is used, since applying both would correct the imbalance twice.
		Tensor weights({7}, 1.0f);
		if (balance == "loss") {
			std::vector<size_t> counts = train_raw.class_counts();
			for (size_t c = 0; c < counts.size(); c++) {
				weights.data()[c] = counts[c] ? static_cast<float>(train_raw.size()) / (counts.size() * static_cast<float>(counts[c])) : 0.0f;
			}
		}
		const bool sampler = balance == "sampler";
		const std::vector<std::string> names = {"Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"};

		auto model = build_model();
		AdamW opt(model->parameters(), lr, 0.9f, 0.999f, 1e-8f, 5e-2f);
		DataLoader train_loader = sampler ? DataLoader(train_set, batch, balanced_sample_weights(train_raw.labels(), 7))
		                                  : DataLoader(train_set, batch, true);
		DataLoader val_loader(val_set, batch, false), test_loader(test_set, batch, false);
		size_t total_steps = epochs * train_loader.num_batches();
		size_t warmup = total_steps / 10;
		LinearWarmup sched(opt, std::make_unique<CosineAnnealingLR>(opt, total_steps - warmup), warmup);
		std::printf("train %zu  validation (PublicTest) %zu  test (PrivateTest) %zu  balance %s  epochs %zu\n",
		            train_raw.size(), val_raw.size(), test_raw.size(), balance.c_str(), epochs);

		std::vector<size_t> y_true, y_pred;
		float best_f1 = -1.0f;
		size_t best_epoch = 0;
		const auto t_start = std::chrono::steady_clock::now();
		for (size_t epoch = 0; epoch < epochs; epoch++) {
			model->train();
			train_loader.reset_epoch();
			float epoch_loss = 0.0f;
			for (size_t b = 0; b < train_loader.num_batches(); b++) {
				auto [x, y] = train_loader.get_batch(b);
				Value loss = cross_entropy_loss(model->forward(x), y, weights, smoothing);
				opt.zero_grad();
				loss.backward();
				clip_grad_norm(model->parameters(), 1.0f);
				opt.step();
				sched.step();
				epoch_loss += loss.data().get({0});
			}
			predict(*model, val_loader, y_true, y_pred);
			const float f1 = f1_score(y_true, y_pred, 7, Average::Macro);
			const bool best = f1 > best_f1;
			if (best) {
				best_f1 = f1;
				best_epoch = epoch;
				save_checkpoint(out_path, *model, &opt, static_cast<int64_t>((epoch + 1) * train_loader.num_batches()));
			}
			const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
			std::printf("epoch %zu loss %.4f val accuracy %.2f%% macro-F1 %.4f%s  [%.0f s]\n", epoch,
			            epoch_loss / static_cast<float>(train_loader.num_batches()),
			            100.0f * accuracy_score(y_true, y_pred), f1, best ? " *" : "", elapsed);
		}

		// The best epoch by validation macro-F1 is scored once on the held-out PrivateTest rows
		load_checkpoint(out_path, *model, nullptr);
		predict(*model, test_loader, y_true, y_pred);
		std::printf("\nbest epoch %zu (validation macro-F1 %.4f), checkpoint %s\n", best_epoch, best_f1, out_path.c_str());
		std::printf("\ntest report (PrivateTest):\n%s", classification_report(y_true, y_pred, 7, names).c_str());
		std::printf("\nconfusion matrix (rows: true class, columns: predicted):\n%10s", "");
		for (const std::string& n : names) std::printf(" %8.8s", n.c_str());
		std::printf("\n");
		const auto cm = confusion_matrix(y_true, y_pred, 7);
		for (size_t t = 0; t < 7; t++) {
			std::printf("%10s", names[t].c_str());
			for (size_t p = 0; p < 7; p++) std::printf(" %8zu", cm[t][p]);
			std::printf("\n");
		}
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		return 2;
	}
	return 0;
}
