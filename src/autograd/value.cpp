/**
 * value.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Reference-counted autograd graph node handle over Tensor
 */

#include "nn/autograd/value.hpp"

#include <algorithm>
#include <unordered_set>

namespace nn {

namespace {
size_t g_live_node_count = 0;
}

NodeImpl::NodeImpl(Tensor d, bool rg, bool allocate_grad)
	: data(std::move(d)), grad(allocate_grad ? data.shape() : std::vector<size_t>{0}, 0.0f), requires_grad(rg) {
	g_live_node_count++;
}

NodeImpl::~NodeImpl() { g_live_node_count--; }

Value::Value(std::shared_ptr<NodeImpl> impl) : impl_(std::move(impl)) {}

Value Value::leaf(Tensor data, bool requires_grad) {
	return Value(std::make_shared<NodeImpl>(std::move(data), requires_grad));
}

namespace {
thread_local bool g_grad_enabled = true;
}

NoGradGuard::NoGradGuard() : previous_(g_grad_enabled) { g_grad_enabled = false; }
NoGradGuard::~NoGradGuard() { g_grad_enabled = previous_; }
bool NoGradGuard::grad_enabled() { return g_grad_enabled; }

Value Value::node(Tensor data, std::vector<Value> parents, std::function<void(const Tensor&)> backward_fn) {
	if (!g_grad_enabled) return leaf(std::move(data), false);
	auto impl = std::make_shared<NodeImpl>(std::move(data), true, false);
	impl->parents = std::move(parents);
	impl->backward_fn = std::move(backward_fn);
	return Value(impl);
}

Tensor& Value::data() const& { return impl_->data; }
Tensor Value::data() && { return impl_->data; }
Tensor& Value::grad() const& {
	// A gradient nobody stored into yet stands for zeros; make that real for a reader or a partial writer
	if (impl_->grad_fresh) {
		impl_->grad.zero_();
		impl_->grad_fresh = false;
	}
	return impl_->grad;
}

Tensor Value::grad() && { return static_cast<const Value&>(*this).grad(); }

Tensor& Value::grad_accum(bool& first) const {
	first = impl_->grad_fresh;
	impl_->grad_fresh = false;
	return impl_->grad;
}
bool Value::requires_grad() const { return impl_->requires_grad; }
bool Value::has_grad() const { return impl_->grad.shape() == impl_->data.shape(); }
NodeImpl* Value::impl() const { return impl_.get(); }

void Value::zero_grad() const {
	if (impl_->grad.shape() != impl_->data.shape()) {
		impl_->grad = Tensor(impl_->data.shape(), 0.0f);
	} else {
		impl_->grad.zero_();
	}
	impl_->grad_fresh = false;
}

size_t Value::live_node_count() { return g_live_node_count; }

void Value::backward() const {
	if (impl_->data.size() != 1) {
		throw AutogradShapeError();
	}

	std::vector<Value> topo_order;
	// Iterative post-order DFS. A node is expanded once and appended once every parent has been
	// appended, so it always lands after all of its parents, however its consumers are nested.
	std::unordered_set<NodeImpl*> expanded;
	std::unordered_set<NodeImpl*> done;
	std::vector<Value> stack;

	stack.push_back(*this);
	while (!stack.empty()) {
		Value v = stack.back();
		if (done.count(v.impl())) {
			stack.pop_back();
			continue;
		}
		if (expanded.count(v.impl())) {
			stack.pop_back();
			done.insert(v.impl());
			topo_order.push_back(v);
			continue;
		}
		expanded.insert(v.impl());
		for (const Value& p : v.impl()->parents) {
			if (!expanded.count(p.impl())) stack.push_back(p);
		}
	}

	for (const Value& v : topo_order) {
		if (v.impl()->backward_fn) {
			// Not zeroed here: the first backward rule that reaches the node stores into it (see grad_accum)
			if (v.impl()->grad.shape() != v.impl()->data.shape()) {
				v.impl()->grad = Tensor::empty(v.impl()->data.shape());
			}
			v.impl()->grad_fresh = true;
		}
	}
	if (impl_->grad.shape() != impl_->data.shape()) {
		impl_->grad = Tensor(impl_->data.shape(), 1.0f);
	} else {
		std::fill(impl_->grad.data().begin(), impl_->grad.data().end(), 1.0f);
	}
	impl_->grad_fresh = false;

	for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
		if (it->impl()->backward_fn) {
			it->impl()->backward_fn(it->grad());
		}
	}
}

}  // namespace nn
