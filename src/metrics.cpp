/**
 * metrics.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Classification metrics in the style of scikit-learn: confusion matrix, precision, recall, F1 and a text report
 */

#include "nn/metrics.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace nn {

namespace {

void check_pair(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes) {
	if (y_true.size() != y_pred.size()) throw std::invalid_argument("metrics: y_true and y_pred differ in length");
	for (size_t i = 0; i < y_true.size(); i++) {
		if (y_true[i] >= num_classes || y_pred[i] >= num_classes) {
			throw std::invalid_argument("metrics: label out of range for num_classes");
		}
	}
}

float ratio(double num, double den, float zero_division) {
	return den > 0.0 ? static_cast<float>(num / den) : zero_division;
}

// Mean of per-class values: unweighted, or weighted by support
float combine(const std::vector<float>& values, const std::vector<size_t>& support, Average average) {
	double sum = 0.0, weight = 0.0;
	for (size_t c = 0; c < values.size(); c++) {
		const double w = average == Average::Weighted ? static_cast<double>(support[c]) : 1.0;
		sum += w * values[c];
		weight += w;
	}
	return weight > 0.0 ? static_cast<float>(sum / weight) : 0.0f;
}

enum class Which { Precision, Recall, F1 };

float score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
            Average average, float zero_division, Which which) {
	if (average == Average::Micro) {
		check_pair(y_true, y_pred, num_classes);
		// every wrong prediction is one false positive and one false negative, so all three coincide
		return accuracy_score(y_true, y_pred);
	}
	ClassScores s = precision_recall_f1(y_true, y_pred, num_classes, zero_division);
	const std::vector<float>& v = which == Which::Precision ? s.precision : which == Which::Recall ? s.recall : s.f1;
	return combine(v, s.support, average);
}

}  // namespace

std::vector<size_t> to_labels(const Tensor& ids) {
	std::vector<size_t> out(ids.size());
	for (size_t i = 0; i < out.size(); i++) {
		const float v = ids.data()[i];
		if (!(v >= 0.0f) || v != std::floor(v)) throw std::invalid_argument("to_labels: not a non-negative integer");
		out[i] = static_cast<size_t>(v);
	}
	return out;
}

std::vector<size_t> argmax_rows(const Tensor& logits) {
	if (logits.shape().size() != 2 || logits.shape()[1] == 0) throw std::invalid_argument("argmax_rows: expected shape {N, C}");
	const size_t n = logits.shape()[0], c = logits.shape()[1];
	const std::vector<float>& d = logits.data();
	std::vector<size_t> out(n, 0);
	for (size_t i = 0; i < n; i++) {
		for (size_t k = 1; k < c; k++) {
			if (d[i * c + k] > d[i * c + out[i]]) out[i] = k;
		}
	}
	return out;
}

std::vector<std::vector<size_t>> confusion_matrix(const std::vector<size_t>& y_true,
                                                  const std::vector<size_t>& y_pred, size_t num_classes) {
	check_pair(y_true, y_pred, num_classes);
	std::vector<std::vector<size_t>> m(num_classes, std::vector<size_t>(num_classes, 0));
	for (size_t i = 0; i < y_true.size(); i++) m[y_true[i]][y_pred[i]]++;
	return m;
}

float accuracy_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred) {
	if (y_true.size() != y_pred.size()) throw std::invalid_argument("metrics: y_true and y_pred differ in length");
	if (y_true.empty()) return 0.0f;
	size_t correct = 0;
	for (size_t i = 0; i < y_true.size(); i++) correct += y_true[i] == y_pred[i];
	return static_cast<float>(static_cast<double>(correct) / static_cast<double>(y_true.size()));
}

ClassScores precision_recall_f1(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred,
                                size_t num_classes, float zero_division) {
	const auto m = confusion_matrix(y_true, y_pred, num_classes);
	ClassScores s;
	s.precision.resize(num_classes);
	s.recall.resize(num_classes);
	s.f1.resize(num_classes);
	s.support.resize(num_classes);
	for (size_t c = 0; c < num_classes; c++) {
		double predicted = 0.0, actual = 0.0;
		for (size_t k = 0; k < num_classes; k++) {
			predicted += static_cast<double>(m[k][c]);
			actual += static_cast<double>(m[c][k]);
		}
		const double tp = static_cast<double>(m[c][c]);
		s.precision[c] = ratio(tp, predicted, zero_division);
		s.recall[c] = ratio(tp, actual, zero_division);
		s.support[c] = static_cast<size_t>(actual);
		// F1 is undefined only when precision and recall both have a zero denominator, or both are zero
		s.f1[c] = predicted + actual > 0.0 ? static_cast<float>(2.0 * tp / (predicted + actual)) : zero_division;
	}
	return s;
}

float precision_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
                      Average average, float zero_division) {
	return score(y_true, y_pred, num_classes, average, zero_division, Which::Precision);
}

float recall_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
                   Average average, float zero_division) {
	return score(y_true, y_pred, num_classes, average, zero_division, Which::Recall);
}

float f1_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
               Average average, float zero_division) {
	return score(y_true, y_pred, num_classes, average, zero_division, Which::F1);
}

std::string classification_report(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred,
                                  size_t num_classes, const std::vector<std::string>& class_names, int digits,
                                  float zero_division) {
	if (!class_names.empty() && class_names.size() != num_classes) {
		throw std::invalid_argument("classification_report: class_names must have one entry per class");
	}
	const ClassScores s = precision_recall_f1(y_true, y_pred, num_classes, zero_division);
	std::vector<std::string> names(num_classes);
	size_t width = std::string("weighted avg").size();
	for (size_t c = 0; c < num_classes; c++) {
		names[c] = class_names.empty() ? std::to_string(c) : class_names[c];
		if (names[c].size() > width) width = names[c].size();
	}
	size_t total = 0;
	for (size_t n : s.support) total += n;

	std::string out;
	char buf[256];
	auto row = [&](const std::string& name, float p, float r, float f, size_t n) {
		std::snprintf(buf, sizeof buf, "%*s  %9.*f %9.*f %9.*f %9zu\n", static_cast<int>(width), name.c_str(), digits, p,
		              digits, r, digits, f, n);
		out += buf;
	};
	std::snprintf(buf, sizeof buf, "%*s  %9s %9s %9s %9s\n\n", static_cast<int>(width), "", "precision", "recall",
	              "f1-score", "support");
	out += buf;
	for (size_t c = 0; c < num_classes; c++) row(names[c], s.precision[c], s.recall[c], s.f1[c], s.support[c]);
	out += "\n";
	std::snprintf(buf, sizeof buf, "%*s  %20s%9.*f %9zu\n", static_cast<int>(width), "accuracy", "", digits,
	              accuracy_score(y_true, y_pred), total);
	out += buf;
	row("macro avg", combine(s.precision, s.support, Average::Macro), combine(s.recall, s.support, Average::Macro),
	    combine(s.f1, s.support, Average::Macro), total);
	row("weighted avg", combine(s.precision, s.support, Average::Weighted),
	    combine(s.recall, s.support, Average::Weighted), combine(s.f1, s.support, Average::Weighted), total);
	return out;
}

}  // namespace nn
