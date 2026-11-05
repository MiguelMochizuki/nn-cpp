/**
 * metrics.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Classification metrics in the style of scikit-learn: confusion matrix, precision, recall, F1 and a text report
 */

#pragma once

#include "nn/tensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nn {

/** How per-class scores are combined into one number */
enum class Average {
	Micro,     // count all samples together (equals accuracy for single-label problems)
	Macro,     // unweighted mean over the classes
	Weighted,  // mean over the classes weighted by their support
};

/** Per-class scores, one entry per class id 0 .. num_classes - 1 */
struct ClassScores {
	std::vector<float> precision;
	std::vector<float> recall;
	std::vector<float> f1;
	std::vector<size_t> support;  // number of true samples of each class
};

/**
 * Converts a tensor of class ids (floats holding integers, as the DataLoader yields them) to labels
 *
 * Parameters:
 * const Tensor& ids: tensor whose elements are non-negative integers
 *
 * Returns std::vector<size_t>: one label per element; throws std::invalid_argument for a non-integral or negative value
 */
std::vector<size_t> to_labels(const Tensor& ids);

/**
 * Index of the largest value in each row of a {N, C} tensor (the first one wins a tie)
 *
 * Parameters:
 * const Tensor& logits: scores of shape {N, C} with C >= 1
 *
 * Returns std::vector<size_t>: N predicted classes; throws std::invalid_argument if logits is not 2-D
 */
std::vector<size_t> argmax_rows(const Tensor& logits);

/**
 * Confusion matrix: entry [t][p] counts samples of true class t predicted as class p
 *
 * Parameters:
 * const std::vector<size_t>& y_true: true labels
 * const std::vector<size_t>& y_pred: predicted labels, same length
 * size_t num_classes: number of classes; every label must be below it
 *
 * Returns std::vector<std::vector<size_t>>: num_classes x num_classes counts; throws std::invalid_argument on a length mismatch or an out-of-range label
 */
std::vector<std::vector<size_t>> confusion_matrix(const std::vector<size_t>& y_true,
                                                  const std::vector<size_t>& y_pred, size_t num_classes);

/**
 * Fraction of samples predicted correctly
 *
 * Parameters:
 * const std::vector<size_t>& y_true: true labels
 * const std::vector<size_t>& y_pred: predicted labels, same length
 *
 * Returns float: accuracy in [0, 1]; 0 for empty input; throws std::invalid_argument on a length mismatch
 */
float accuracy_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred);

/**
 * Precision, recall, F1 and support of every class. A score whose denominator is zero (precision of a class
 * never predicted, recall of a class absent from y_true, F1 of either) takes the value zero_division.
 *
 * Parameters:
 * const std::vector<size_t>& y_true: true labels
 * const std::vector<size_t>& y_pred: predicted labels, same length
 * size_t num_classes: number of classes; every label must be below it
 * float zero_division: value used when a denominator is zero (scikit-learn's default is 0)
 *
 * Returns ClassScores: vectors of length num_classes; throws std::invalid_argument as confusion_matrix does
 */
ClassScores precision_recall_f1(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred,
                                size_t num_classes, float zero_division = 0.0f);

/**
 * Precision combined over the classes as chosen by average (see Average)
 *
 * Parameters:
 * const std::vector<size_t>& y_true: true labels
 * const std::vector<size_t>& y_pred: predicted labels, same length
 * size_t num_classes: number of classes
 * Average average: how to combine the classes
 * float zero_division: value used when a denominator is zero
 *
 * Returns float: the averaged score; Macro averages over all num_classes classes, including absent ones
 */
float precision_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
                      Average average = Average::Macro, float zero_division = 0.0f);

/**
 * Recall combined over the classes as chosen by average
 *
 * Parameters: as precision_score
 *
 * Returns float: the averaged score
 */
float recall_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
                   Average average = Average::Macro, float zero_division = 0.0f);

/**
 * F1 (harmonic mean of precision and recall) combined over the classes as chosen by average. Macro and
 * Weighted average the per-class F1 values; Micro is the F1 of the pooled counts.
 *
 * Parameters: as precision_score
 *
 * Returns float: the averaged score
 */
float f1_score(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred, size_t num_classes,
               Average average = Average::Macro, float zero_division = 0.0f);

/**
 * Text table with precision, recall, F1 and support per class, then accuracy, macro and weighted averages,
 * laid out like scikit-learn's classification_report
 *
 * Parameters:
 * const std::vector<size_t>& y_true: true labels
 * const std::vector<size_t>& y_pred: predicted labels, same length
 * size_t num_classes: number of classes
 * const std::vector<std::string>& class_names: row names; empty means "0", "1", ...; otherwise one per class
 * int digits: decimals shown for the scores
 * float zero_division: value used when a denominator is zero
 *
 * Returns std::string: the report, ending with a newline; throws std::invalid_argument if class_names has the wrong size
 */
std::string classification_report(const std::vector<size_t>& y_true, const std::vector<size_t>& y_pred,
                                  size_t num_classes, const std::vector<std::string>& class_names = {},
                                  int digits = 2, float zero_division = 0.0f);

}  // namespace nn
