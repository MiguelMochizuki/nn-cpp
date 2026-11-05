/**
 * value.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Reference-counted autograd graph node handle over Tensor
 */

#pragma once

#include "nn/tensor.hpp"

#include <memory>
#include <vector>
#include <functional>
#include <exception>

namespace nn {

/*
 * Thrown when backward() is called on a non-scalar Value.
 *
 * Attributes: none
 */
class AutogradShapeError : public std::exception {
public:
	const char* what() const noexcept override {
		return "backward() requires a scalar Value (size 1).";
	}
};

class Value;

/*
 * RAII scope that disables graph building on the current thread: while one is alive,
 * Value::node returns a plain leaf with no parents and no backward rule, so evaluating a
 * model for inference keeps no autograd graph. Guards nest; the previous mode is restored
 * on destruction.
 *
 * Attributes:
 * bool previous_: grad mode in effect before this guard
 */
class NoGradGuard {
public:
	NoGradGuard();
	~NoGradGuard();
	NoGradGuard(const NoGradGuard&) = delete;
	NoGradGuard& operator=(const NoGradGuard&) = delete;

	/* Public, whether ops currently build the autograd graph on this thread
	 *
	 * Parameters: none
	 *
	 * Returns bool: true unless a NoGradGuard is alive
	 */
	static bool grad_enabled();

private:
	bool previous_;
};

/*
 * Internal graph node state, jointly owned by every Value handle referencing it.
 * Edges point one direction only, child to parent, so the graph structure itself
 * cannot form a cycle; see nn::Value::node for the one rule that must hold to
 * keep it that way.
 *
 * Attributes:
 * Tensor data: forward value
 * Tensor grad: accumulated dL/d(data), same shape as data (a non-leaf node holds an empty placeholder until backward() sizes it)
 * bool requires_grad: whether this node participates in backward
 * vector<Value> parents: inputs this node was computed from
 * function<void(const Tensor&)> backward_fn: given dL/d(this.data), accumulate into each parent's grad
 */
struct NodeImpl {
	Tensor data;
	Tensor grad;
	bool requires_grad;
	std::vector<Value> parents;
	std::function<void(const Tensor&)> backward_fn;

	NodeImpl(Tensor d, bool rg, bool allocate_grad = true);
	~NodeImpl();
};

/*
 * Reference-counted handle to a graph node. Copyable, cheap (shared_ptr copy).
 * Two Values wrapping the same NodeImpl refer to the same tensor/grad/graph position.
 *
 * Attributes:
 * shared_ptr<NodeImpl> impl_: shared graph node state
 */
class Value {
public:
	/* Public, wrap raw data as a leaf node with no parents
	 *
	 * Parameters:
	 * Tensor data: initial tensor value
	 * bool requires_grad: track gradient if true
	 *
	 * Returns Value: new leaf handle
	 */
	static Value leaf(Tensor data, bool requires_grad = false);

	/* Public, construct a non-leaf node from parents and a backward rule; used only by
	 * autograd op implementations, never directly by layer/application code. backward_fn
	 * must capture only parents and saved Tensors, never the Value this call returns —
	 * capturing it would create a shared_ptr reference cycle
	 *
	 * Parameters:
	 * Tensor data: forward result
	 * vector<Value> parents: inputs this node was computed from
	 * function<void(const Tensor&)> backward_fn: local backward rule
	 *
	 * Returns Value: new non-leaf handle, requires_grad() true
	 */
	static Value node(Tensor data, std::vector<Value> parents, std::function<void(const Tensor&)> backward_fn);

	/* Public, forward value, read-write
	 *
	 * Parameters: none
	 *
	 * Returns Tensor&: reference to the node's data
	 */
	Tensor& data() const&;

	/* Public, forward value of a temporary Value, by copy: a reference would dangle once the
	 * temporary (e.g. model.forward(x)) and its node are destroyed at the end of the expression
	 *
	 * Parameters: none
	 *
	 * Returns Tensor: copy of the node's data
	 */
	Tensor data() &&;

	/* Public, accumulated gradient, read-write
	 *
	 * Parameters: none
	 *
	 * Returns Tensor&: reference to the node's grad. For a non-leaf node it is
	 * an empty placeholder (shape {0}) until backward() reaches that node
	 */
	Tensor& grad() const&;

	/* Public, accumulated gradient of a temporary Value, by copy; see data() &&
	 *
	 * Parameters: none
	 *
	 * Returns Tensor: copy of the node's grad
	 */
	Tensor grad() &&;

	/* Public, whether this node participates in backward
	 *
	 * Parameters: none
	 *
	 * Returns bool: requires_grad flag
	 */
	bool requires_grad() const;

	/* Public, identity of the underlying graph node, for use as a visited-set key
	 *
	 * Parameters: none
	 *
	 * Returns NodeImpl*: raw pointer to the shared node state
	 */
	NodeImpl* impl() const;

	/* Public, run backward pass from this node
	 *
	 * Parameters: none
	 *
	 * Returns void: seeds this node's grad to 1, then runs every reachable node's
	 * backward_fn in reverse topological order; throws AutogradShapeError if this
	 * Value is not scalar (size() != 1)
	 */
	void backward() const;

	/* Public, reset this node's own grad to zero
	 *
	 * Parameters: none
	 *
	 * Returns void: none
	 */
	void zero_grad() const;

	/* Public, count of currently-live graph nodes, for leak testing
	 *
	 * Parameters: none
	 *
	 * Returns size_t: number of NodeImpl instances currently allocated
	 */
	static size_t live_node_count();

private:
	explicit Value(std::shared_ptr<NodeImpl> impl);
	std::shared_ptr<NodeImpl> impl_;
};

}  // namespace nn
