/**
 * train_transformer.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Trains a small decoder-only Transformer on generated string tasks (reverse, copy, add) and reports
 * exact-match accuracy of greedy generation on held-out samples
 */

#include "nn/autograd/ops.hpp"
#include "nn/data/text.hpp"
#include "nn/nn/transformer.hpp"
#include "nn/optim/adamw.hpp"
#include "nn/optim/clip.hpp"
#include "nn/optim/scheduler.hpp"
#include "nn/random.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <set>
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
	std::fprintf(stderr, "usage: %s [--task reverse|copy|add] [--steps N] [--batch N] [--lr X] [--seed N] [--dim N] [--layers N]\n", prog);
	std::fprintf(stderr, "  defaults: --task reverse --steps 3000 --batch 64 --lr 3e-3 --seed 1 --dim 64 --layers 2 (4 heads)\n");
}

/**
 * Generates one sample string of the task; every sample of a task has the same length
 *
 * Parameters:
 * const std::string& task: reverse, copy or add
 * std::mt19937& rng: random source
 *
 * Returns std::string: "<w>=<reverse(w)>", "<w>=<w>" or "<a>+<b>=<c>"
 */
static std::string make_sample(const std::string& task, std::mt19937& rng) {
	if (task == "add") {
		const int a = static_cast<int>(rng() % 100), b = static_cast<int>(rng() % 100), c = a + b;
		char buf[16];
		std::snprintf(buf, sizeof buf, "%02d+%02d=%03d", a, b, c);
		return buf;
	}
	std::string w;
	for (int i = 0; i < 6; i++) w += static_cast<char>('a' + rng() % 26);
	std::string ans = w;
	if (task == "reverse") ans.assign(w.rbegin(), w.rend());
	return w + "=" + ans;
}

/**
 * Builds a batch of samples as model inputs and shifted targets, with targets before the answer ignored
 *
 * Parameters:
 * const CharVocab& vocab: symbol ids
 * const std::vector<std::string>& s: equal-length samples
 *
 * Returns pair<Tensor, Tensor>: input ids {N, L-1}; targets {N, L-1} with -100 on prompt positions
 */
static std::pair<Tensor, Tensor> make_batch(const CharVocab& vocab, const std::vector<std::string>& s) {
	const size_t n = s.size(), t = s[0].size() - 1, eq = s[0].find('=');
	Tensor x({n, t}), y({n, t});
	for (size_t i = 0; i < n; i++) {
		std::vector<float> ids = vocab.encode(s[i]);
		for (size_t j = 0; j < t; j++) {
			x.data()[i * t + j] = ids[j];
			y.data()[i * t + j] = j < eq ? -100.0f : ids[j + 1];
		}
	}
	return {x, y};
}

/**
 * Exact-match accuracy of greedy generation: the prompt through '=' is fed and the answer is generated one
 * token at a time by re-running the model on the growing prefix (no cache)
 *
 * Parameters:
 * Transformer& model: network to evaluate (switched to eval mode, left in eval mode)
 * const CharVocab& vocab: symbol ids
 * const std::vector<std::string>& samples: held-out samples
 *
 * Returns float: fraction of samples whose generated answer equals the true one
 */
static float exact_match(Transformer& model, const CharVocab& vocab, const std::vector<std::string>& samples) {
	model.eval();
	NoGradGuard no_grad;
	const size_t n = samples.size(), len = samples[0].size(), eq = samples[0].find('=');
	const size_t v = vocab.size();
	std::vector<std::vector<float>> seq(n);
	for (size_t i = 0; i < n; i++) seq[i] = vocab.encode(samples[i].substr(0, eq + 1));
	for (size_t t = eq + 1; t < len; t++) {
		Tensor x({n, t});
		for (size_t i = 0; i < n; i++) {
			for (size_t j = 0; j < t; j++) x.data()[i * t + j] = seq[i][j];
		}
		Tensor logits = model.forward(Value::leaf(x)).data();
		for (size_t i = 0; i < n; i++) {
			const float* row = &logits.data()[(i * t + t - 1) * v];
			size_t best = 0;
			for (size_t c = 1; c < v; c++) {
				if (row[c] > row[best]) best = c;
			}
			seq[i].push_back(static_cast<float>(best));
		}
	}
	size_t ok = 0;
	for (size_t i = 0; i < n; i++) ok += vocab.decode(seq[i]) == samples[i];
	return static_cast<float>(ok) / static_cast<float>(n);
}

/**
 * Trains the Transformer on a generated task, printing loss and held-out exact-match accuracy every 100 steps
 *
 * Parameters:
 * int argc: Argument count
 * char** argv: optional --task, --steps, --batch, --lr, --seed, --dim, --layers
 *
 * Returns int: 0 on success, 1 on bad usage, 2 on a runtime failure
 */
int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: progress lines appear as printed (a sized _IOLBF is rejected on Windows)
	std::string task = "reverse";
	size_t steps = 3000, batch = 64, dim = 64, layers = 2;
	float lr = 3e-3f;
	unsigned long long seed = 1;
	try {
		for (int i = 1; i < argc; i++) {
			std::string a = argv[i];
			if (a == "--help" || a == "-h") {
				usage(argv[0]);
				return 1;
			}
			if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
			std::string v = argv[++i];
			size_t used = 0;
			if (a != "--task" && !v.empty() && v[0] == '-') throw std::invalid_argument("bad value for " + a + ": " + v);
			if (a == "--task") { task = v; used = v.size(); }
			else if (a == "--steps") steps = std::stoul(v, &used);
			else if (a == "--batch") batch = std::stoul(v, &used);
			else if (a == "--lr") lr = std::stof(v, &used);
			else if (a == "--seed") seed = std::stoull(v, &used);
			else if (a == "--dim") dim = std::stoul(v, &used);
			else if (a == "--layers") layers = std::stoul(v, &used);
			else throw std::invalid_argument("unknown option " + a);
			if (used != v.size()) throw std::invalid_argument("bad value for " + a + ": " + v);
		}
		if (steps == 0 || batch == 0 || dim == 0 || layers == 0 || !(lr > 0.0f)) throw std::invalid_argument("--steps, --batch, --lr, --dim and --layers must be positive");
		if (dim % 4 != 0) throw std::invalid_argument("--dim must be divisible by 4 (heads)");
		if (task != "reverse" && task != "copy" && task != "add") throw std::invalid_argument("--task must be reverse, copy or add");
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		usage(argv[0]);
		return 1;
	}

	try {
		manual_seed(seed);
		std::mt19937 rng = derive_generator("train_transformer_data");
		CharVocab vocab("abcdefghijklmnopqrstuvwxyz0123456789+=");
		const size_t len = make_sample(task, rng).size();
		TransformerConfig cfg{vocab.size(), dim, 4, layers, len - 1};
		Transformer model(cfg);
		AdamW opt(model.parameters(), lr, 0.9f, 0.999f, 1e-8f, 1e-2f);
		const size_t warmup = steps / 10;
		LinearWarmup sched(opt, std::make_unique<CosineAnnealingLR>(opt, steps - warmup > 0 ? steps - warmup : 1), warmup);

		// 200 distinct held-out samples; training rejects them (add has only 10000 pairs, so this matters there)
		std::vector<std::string> held;
		std::set<std::string> held_set;
		while (held.size() < 200) {
			std::string s = make_sample(task, rng);
			if (held_set.insert(s).second) held.push_back(s);
		}
		std::printf("task %s  sample length %zu  vocab %zu  steps %zu  batch %zu\n", task.c_str(), len, vocab.size(), steps, batch);

		const auto t_start = std::chrono::steady_clock::now();
		float acc = 0.0f;
		for (size_t step = 1; step <= steps; step++) {
			model.train();
			std::vector<std::string> s(batch);
			for (std::string& x : s) {
				do x = make_sample(task, rng);
				while (held_set.count(x));
			}
			auto [x, y] = make_batch(vocab, s);
			Value loss = sequence_cross_entropy(model.forward(Value::leaf(x)), y);
			opt.zero_grad();
			loss.backward();
			clip_grad_norm(model.parameters(), 1.0f);
			opt.step();
			sched.step();
			if (step % 100 == 0) {
				acc = exact_match(model, vocab, held);
				const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
				std::printf("step %zu loss %.4f accuracy %.3f  [%.0f s]\n", step, loss.data().get({0}), acc, elapsed);
			}
		}
		if (steps % 100 != 0) acc = exact_match(model, vocab, held);
		std::printf("final accuracy %.3f\n", acc);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "error: %s\n", e.what());
		return 2;
	}
	return 0;
}
