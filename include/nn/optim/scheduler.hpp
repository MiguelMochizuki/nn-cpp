/**
 * scheduler.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Learning-rate schedulers that drive Optimizer::set_lr
 */

#pragma once

#include "nn/optim/optimizer.hpp"

#include <cstddef>
#include <memory>

namespace nn {

/*
 * Base class for schedules. The learning rate is a pure function of the step count, computed
 * from the base lr read from the optimizer at construction. Holds a reference to the optimizer,
 * which must outlive the scheduler. Construction does not change the optimizer's lr.
 *
 * Attributes:
 * Optimizer& opt_: optimizer whose lr is driven
 * size_t step_: number of step() calls so far
 * float base_lr_: optimizer lr at construction
 */
class LRScheduler {
public:
	/* Public, attach to an optimizer and capture its current lr as the base lr
	 *
	 * Parameters:
	 * Optimizer& opt: optimizer to drive; must support lr() and outlive this scheduler
	 *
	 * Returns LRScheduler: none (constructor)
	 */
	explicit LRScheduler(Optimizer& opt);

	virtual ~LRScheduler() = default;

	/* Public, advance one step and set the optimizer's lr to compute(step_count())
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void step();

	/* Public, jump to step n and apply its lr, for resuming training
	 *
	 * Parameters:
	 * size_t n: step count to resume at
	 *
	 * Returns void: none
	 */
	void set_step(size_t n);

	size_t step_count() const { return step_; }
	float base_lr() const { return base_lr_; }

protected:
	/* Protected, learning rate at a given step
	 *
	 * Parameters:
	 * size_t step: step count
	 *
	 * Returns float: the lr for that step
	 */
	virtual float compute(size_t step) const = 0;

	Optimizer& opt_;
	size_t step_ = 0;
	float base_lr_;

	friend class LinearWarmup;
};

/* Multiplies the lr by gamma every step_size steps: base * gamma^(step / step_size). */
class StepLR : public LRScheduler {
public:
	/* Public, construct a step-decay schedule
	 *
	 * Parameters:
	 * Optimizer& opt: optimizer to drive
	 * size_t step_size: steps between decays; 0 throws std::invalid_argument
	 * float gamma: decay factor, default 0.1
	 *
	 * Returns StepLR: none (constructor)
	 */
	StepLR(Optimizer& opt, size_t step_size, float gamma = 0.1f);

protected:
	float compute(size_t step) const override;

private:
	size_t step_size_;
	float gamma_;
};

/* Cosine decay from the base lr to eta_min over t_max steps, then constant at eta_min. */
class CosineAnnealingLR : public LRScheduler {
public:
	/* Public, construct a cosine schedule
	 *
	 * Parameters:
	 * Optimizer& opt: optimizer to drive
	 * size_t t_max: steps to reach eta_min; 0 throws std::invalid_argument
	 * float eta_min: final lr, default 0
	 *
	 * Returns CosineAnnealingLR: none (constructor)
	 */
	CosineAnnealingLR(Optimizer& opt, size_t t_max, float eta_min = 0.0f);

protected:
	float compute(size_t step) const override;

private:
	size_t t_max_;
	float eta_min_;
};

/* Linear ramp from 0 to the base lr over warmup_steps, then the inner schedule shifted by warmup_steps. */
class LinearWarmup : public LRScheduler {
public:
	/* Public, construct a warmup in front of another schedule; sets the optimizer's lr to the
	 * step 0 value (0) immediately so the first update uses the warmed-up lr
	 *
	 * Parameters:
	 * Optimizer& opt: optimizer to drive; must be the one the inner scheduler drives
	 * unique_ptr<LRScheduler> inner: schedule used after warmup; must not be null
	 * size_t warmup_steps: length of the ramp
	 *
	 * Returns LinearWarmup: none (constructor)
	 */
	LinearWarmup(Optimizer& opt, std::unique_ptr<LRScheduler> inner, size_t warmup_steps);

protected:
	float compute(size_t step) const override;

private:
	std::unique_ptr<LRScheduler> inner_;
	size_t warmup_steps_;
};

}  // namespace nn
