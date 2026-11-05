/**
 * test_metrics.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Unit tests for the classification metrics
 */

#include "doctest/doctest.h"

#include "nn/metrics.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace nn;
using Labels = std::vector<size_t>;

// The example from the scikit-learn documentation of classification_report
static const Labels kTrue = {0, 1, 2, 2, 2};
static const Labels kPred = {0, 0, 2, 2, 1};

TEST_CASE("confusion_matrix has true classes in rows and predictions in columns") {
	auto m = confusion_matrix(kTrue, kPred, 3);
	CHECK(m[0] == Labels{1, 0, 0});
	CHECK(m[1] == Labels{1, 0, 0});
	CHECK(m[2] == Labels{0, 1, 2});
}

TEST_CASE("per-class precision, recall, F1 and support match scikit-learn") {
	ClassScores s = precision_recall_f1(kTrue, kPred, 3);
	CHECK(s.precision[0] == doctest::Approx(0.5f));
	CHECK(s.precision[1] == doctest::Approx(0.0f));
	CHECK(s.precision[2] == doctest::Approx(1.0f));
	CHECK(s.recall[0] == doctest::Approx(1.0f));
	CHECK(s.recall[1] == doctest::Approx(0.0f));
	CHECK(s.recall[2] == doctest::Approx(2.0f / 3.0f));
	CHECK(s.f1[0] == doctest::Approx(2.0f / 3.0f));
	CHECK(s.f1[1] == doctest::Approx(0.0f));
	CHECK(s.f1[2] == doctest::Approx(0.8f));
	CHECK(s.support == Labels{1, 1, 3});
}

TEST_CASE("averages match scikit-learn") {
	CHECK(accuracy_score(kTrue, kPred) == doctest::Approx(0.6f));
	CHECK(precision_score(kTrue, kPred, 3, Average::Macro) == doctest::Approx(0.5f));
	CHECK(recall_score(kTrue, kPred, 3, Average::Macro) == doctest::Approx(0.5556f).epsilon(1e-3));
	CHECK(f1_score(kTrue, kPred, 3, Average::Macro) == doctest::Approx(0.4889f).epsilon(1e-3));
	CHECK(precision_score(kTrue, kPred, 3, Average::Weighted) == doctest::Approx(0.7f));
	CHECK(recall_score(kTrue, kPred, 3, Average::Weighted) == doctest::Approx(0.6f));
	CHECK(f1_score(kTrue, kPred, 3, Average::Weighted) == doctest::Approx(0.6133f).epsilon(1e-3));
	for (float v : {precision_score(kTrue, kPred, 3, Average::Micro), recall_score(kTrue, kPred, 3, Average::Micro),
	                f1_score(kTrue, kPred, 3, Average::Micro)}) {
		CHECK(v == doctest::Approx(0.6f));
	}
}

TEST_CASE("a class that is never predicted or absent takes zero_division") {
	Labels t = {0, 0, 1, 1}, p = {0, 0, 0, 0};  // class 1 is never predicted; class 2 does not occur
	ClassScores s = precision_recall_f1(t, p, 3);
	CHECK(s.precision[1] == 0.0f);
	CHECK(s.precision[2] == 0.0f);
	CHECK(s.recall[2] == 0.0f);
	CHECK(s.f1[2] == 0.0f);
	ClassScores one = precision_recall_f1(t, p, 3, 1.0f);
	CHECK(one.precision[1] == 1.0f);
	CHECK(one.recall[2] == 1.0f);
	CHECK(one.f1[2] == 1.0f);
	CHECK(one.precision[0] == doctest::Approx(0.5f));  // defined scores are unaffected
	// Macro counts the absent class, Weighted gives it weight zero
	CHECK(recall_score(t, p, 3, Average::Macro) == doctest::Approx(1.0f / 3.0f));
	CHECK(recall_score(t, p, 3, Average::Weighted) == doctest::Approx(0.5f));
}

TEST_CASE("metrics on empty input and single samples") {
	CHECK(accuracy_score({}, {}) == 0.0f);
	CHECK(f1_score({}, {}, 2, Average::Macro) == 0.0f);
	CHECK(accuracy_score({3}, {3}) == 1.0f);
	CHECK(f1_score({0}, {0}, 1, Average::Weighted) == doctest::Approx(1.0f));
}

TEST_CASE("invalid input throws") {
	CHECK_THROWS_AS(confusion_matrix({0, 1}, {0}, 2), std::invalid_argument);
	CHECK_THROWS_AS(confusion_matrix({0, 2}, {0, 1}, 2), std::invalid_argument);
	CHECK_THROWS_AS(confusion_matrix({0, 1}, {0, 5}, 2), std::invalid_argument);
	CHECK_THROWS_AS(accuracy_score({0}, {0, 1}), std::invalid_argument);
	CHECK_THROWS_AS(classification_report(kTrue, kPred, 3, {"a", "b"}), std::invalid_argument);
}

TEST_CASE("to_labels and argmax_rows") {
	CHECK(to_labels(Tensor({3}, std::vector<float>{0, 2, 5})) == Labels{0, 2, 5});
	CHECK_THROWS_AS(to_labels(Tensor({1}, std::vector<float>{1.5f})), std::invalid_argument);
	CHECK_THROWS_AS(to_labels(Tensor({1}, std::vector<float>{-1.0f})), std::invalid_argument);
	Tensor logits({3, 3}, std::vector<float>{0.1f, 0.9f, 0.0f, 2.0f, 1.0f, 1.0f, 0.5f, 0.5f, 0.5f});
	CHECK(argmax_rows(logits) == Labels{1, 0, 0});  // the first of a tie wins
	CHECK_THROWS_AS(argmax_rows(Tensor({3}, std::vector<float>{1, 2, 3})), std::invalid_argument);
}

TEST_CASE("classification_report reproduces the scikit-learn layout") {
	std::string r = classification_report(kTrue, kPred, 3, {"class 0", "class 1", "class 2"});
	const std::string expected =
	    "              precision    recall  f1-score   support\n"
	    "\n"
	    "     class 0       0.50      1.00      0.67         1\n"
	    "     class 1       0.00      0.00      0.00         1\n"
	    "     class 2       1.00      0.67      0.80         3\n"
	    "\n"
	    "    accuracy                           0.60         5\n"
	    "   macro avg       0.50      0.56      0.49         5\n"
	    "weighted avg       0.70      0.60      0.61         5\n";
	CHECK(r == expected);
}
