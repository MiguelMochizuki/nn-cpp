/**
 * plateau.hpp
 * Author: Miguel Mochizuki Silva
 * Description: ReduceLROnPlateau, a scheduler that lowers the lr when a monitored metric stalls
 */

#pragma once

#include "nn/optim/optimizer.hpp"

#include <cstddef>

namespace nn {

/* Whether a lower (Min, e.g. loss) or a higher (Max, e.g. accuracy) metric is better. */
enum class PlateauMode { Min, Max };

/*
 * Multiplies the optimizer's lr by factor once the metric has not improved for patience epochs
 * (PyTorch ReduceLROnPlateau with threshold_mode='rel'). An epoch improves when
 * metric < best * (1 - threshold) in Min mode, or metric > best * (1 + threshold) in Max mode.
 * After a reduction the bad-epoch counter resets and the next cooldown epochs are not counted.
 * The lr never rises and never goes below min_lr; if a reduction would change the lr by less
 * than 1e-8 (PyTorch's eps) the lr is left untouched, though the counter and cooldown still
 * restart. A NaN metric is a bad epoch and never becomes best(). Holds a reference to the
 * optimizer, which must outlive the scheduler, and keeps no copy of its lr. Construction does
 * not change the optimizer's lr.
 *
 * Attributes:
 * Optimizer& opt_: optimizer whose lr is driven
 * PlateauMode mode_: direction of improvement
 * float factor_: lr multiplier on a reduction
 * size_t patience_: bad epochs tolerated before a reduction
 * float threshold_: relative improvement margin
 * size_t cooldown_: epochs after a reduction during which bad epochs are not counted
 * float min_lr_: lower bound for the lr
 * float best_: best metric so far
 * size_t bad_epochs_: current run of non-improving epochs
 * size_t cooldown_left_: remaining cooldown epochs
 */
class ReduceLROnPlateau {
public:
	/* Public, attach to an optimizer
	 *
	 * Parameters:
	 * Optimizer& opt: optimizer to drive; must outlive this scheduler
	 * PlateauMode mode: Min if lower metrics are better, Max if higher are
	 * float factor: lr multiplier on a reduction, in (0, 1)
	 * size_t patience: consecutive non-improving epochs before a reduction
	 * float threshold: relative improvement margin, non-negative
	 * size_t cooldown: epochs after a reduction in which bad epochs are not counted
	 * float min_lr: lower bound for the lr, non-negative
	 *
	 * Returns ReduceLROnPlateau: none (constructor); throws std::invalid_argument if factor is
	 * outside (0, 1) or threshold or min_lr is negative (or NaN)
	 */
	ReduceLROnPlateau(Optimizer& opt, PlateauMode mode = PlateauMode::Min, float factor = 0.1f,
	                  size_t patience = 10, float threshold = 1e-4f, size_t cooldown = 0,
	                  float min_lr = 0.0f);

	/* Public, record one epoch's metric and reduce the lr if it has plateaued
	 *
	 * Parameters:
	 * float metric: the monitored value for this epoch; NaN counts as a bad epoch
	 *
	 * Returns void: none
	 */
	void step(float metric);

	size_t num_bad_epochs() const { return bad_epochs_; }

	/* Public, best metric so far
	 *
	 * Parameters: none
	 *
	 * Returns float: +inf (Min) or -inf (Max) before any improving step
	 */
	float best() const { return best_; }

private:
	Optimizer& opt_;
	PlateauMode mode_;
	float factor_;
	size_t patience_;
	float threshold_;
	size_t cooldown_;
	float min_lr_;
	float best_;
	size_t bad_epochs_ = 0;
	size_t cooldown_left_ = 0;
};

}
