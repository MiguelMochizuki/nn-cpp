/**
 * optimizer.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Base interface for parameter update rules
 */

#pragma once

#include "nn/autograd/value.hpp"

#include <string>
#include <utility>
#include <vector>

namespace nn {

/* Named tensors describing an optimizer's resumable state (hyperparameters, buffers). */
using OptimizerState = std::vector<std::pair<std::string, Tensor>>;

/*
 * Base interface for optimizers. Holds the parameter set it updates; operates
 * directly on Value data/grad, outside the autograd graph.
 *
 * Attributes:
 * vector<Value> params_: parameters this optimizer updates
 */
class Optimizer {
public:
	/* Public, construct an optimizer over a fixed parameter set
	 *
	 * Parameters:
	 * vector<Value> params: parameters to update on each step()
	 *
	 * Returns Optimizer: none (constructor)
	 */
	explicit Optimizer(std::vector<Value> params);

	virtual ~Optimizer() = default;

	/* Public, apply one update step to every parameter using its current grad
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	virtual void step() = 0;

	/* Public, read the current learning rate
	 *
	 * Parameters: none
	 *
	 * Returns float: the learning rate; the base class throws std::logic_error
	 */
	virtual float lr() const;

	/* Public, change the learning rate used by the next step()
	 *
	 * Parameters:
	 * float lr: new learning rate
	 *
	 * Returns void: none; the base class throws std::logic_error
	 */
	virtual void set_lr(float lr);

	/* Public, reset every tracked parameter's grad to zero
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_grad();

	/* Public, snapshot the state needed to resume training
	 *
	 * Parameters: none
	 *
	 * Returns OptimizerState: named tensors; empty for a stateless optimizer
	 */
	virtual OptimizerState state() const { return {}; }

	/* Public, restore a snapshot produced by state() of the same optimizer type over a
	 * parameter set with the same shapes; all-or-nothing
	 *
	 * Parameters:
	 * OptimizerState state: snapshot to restore
	 *
	 * Returns void: none; throws std::runtime_error naming missing, unexpected or
	 * shape-mismatched entries, leaving this optimizer unchanged
	 */
	virtual void load_state(const OptimizerState& state);

protected:
	/* Protected, throw std::runtime_error unless got has exactly the names and shapes of expected */
	static void check_state(const OptimizerState& got, const OptimizerState& expected);
	/* Protected, copy of snapshot with an "lr" entry (of current_lr) appended when it has none, so v1.0.0 snapshots load */
	static OptimizerState with_lr(const OptimizerState& snapshot, float current_lr);

	std::vector<Value> params_;
};

}  // namespace nn
