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

NodeImpl::NodeImpl(Tensor d, bool rg) : data(d), grad(Tensor(d.shape(), 0.0f)), requires_grad(rg) {
	g_live_node_count++;
}

NodeImpl::~NodeImpl() { g_live_node_count--; }

Value::Value(std::shared_ptr<NodeImpl> impl) : impl_(std::move(impl)) {}

Value Value::leaf(Tensor data, bool requires_grad) {
	return Value(std::make_shared<NodeImpl>(std::move(data), requires_grad));
}

Value Value::node(Tensor data, std::vector<Value> parents, std::function<void(const Tensor&)> backward_fn) {
	auto impl = std::make_shared<NodeImpl>(std::move(data), true);
	impl->parents = std::move(parents);
	impl->backward_fn = std::move(backward_fn);
	return Value(impl);
}

Tensor& Value::data() const { return impl_->data; }
Tensor& Value::grad() const { return impl_->grad; }
bool Value::requires_grad() const { return impl_->requires_grad; }
NodeImpl* Value::impl() const { return impl_.get(); }

void Value::zero_grad() const {
	if (impl_->grad.shape() != impl_->data.shape()) {
		impl_->grad = Tensor(impl_->data.shape(), 0.0f);
	} else {
		impl_->grad.zero_();
	}
}

size_t Value::live_node_count() { return g_live_node_count; }

void Value::backward() const {
	if (impl_->data.size() != 1) {
		throw AutogradShapeError();
	}

	std::vector<Value> topo_order;
	std::unordered_set<NodeImpl*> visited;
	std::vector<Value> stack;
	std::unordered_set<NodeImpl*> pushed;

	stack.push_back(*this);
	pushed.insert(impl());
	while (!stack.empty()) {
		Value v = stack.back();
		if (visited.count(v.impl())) {
			stack.pop_back();
			topo_order.push_back(v);
			continue;
		}
		visited.insert(v.impl());
		for (const Value& p : v.impl()->parents) {
			if (!pushed.count(p.impl())) {
				pushed.insert(p.impl());
				stack.push_back(p);
			}
		}
	}

	for (const Value& v : topo_order) {
		if (v.impl()->backward_fn) {
			if (v.impl()->grad.shape() != v.impl()->data.shape()) {
				v.impl()->grad = Tensor(v.impl()->data.shape(), 0.0f);
			} else {
				v.impl()->grad.zero_();
			}
		}
	}
	if (impl_->grad.shape() != impl_->data.shape()) {
		impl_->grad = Tensor(impl_->data.shape(), 1.0f);
	} else {
		std::fill(impl_->grad.data().begin(), impl_->grad.data().end(), 1.0f);
	}

	for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
		if (it->impl()->backward_fn) {
			it->impl()->backward_fn(it->grad());
		}
	}
}

}  // namespace nn
