/**
 * gradcheck.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Finite-difference gradient checking utility for autograd op tests
 */

#pragma once

#include "nn/autograd/value.hpp"

#include <cmath>
#include <functional>

/* Public, compare analytic backward() gradient against a numerical finite-difference
 * estimate, element by element
 *
 * Parameters:
 * function<Value(Value)> f: builds a scalar output Value from one input Value
 * Value input: the Value to perturb and check gradients for; must have requires_grad() true
 * float eps: finite-difference step size
 * float tol: maximum allowed absolute difference between analytic and numeric gradient
 *
 * Returns bool: true if every element of input.grad() matches its numerical estimate
 * within tol after calling f(input).backward()
 */
inline bool gradcheck(std::function<nn::Value(nn::Value)> f, nn::Value input, float eps = 1e-3f, float tol = 5e-2f) {
	input.zero_grad();
	nn::Value out = f(input);
	out.backward();
	nn::Tensor analytic = input.grad();

	for (size_t i = 0; i < input.data().size(); i++) {
		float orig = input.data().data()[i];

		input.data().data()[i] = orig + eps;
		float loss_plus = f(input).data().data()[0];

		input.data().data()[i] = orig - eps;
		float loss_minus = f(input).data().data()[0];

		input.data().data()[i] = orig;

		float numeric = (loss_plus - loss_minus) / (2.0f * eps);
		if (std::fabs(numeric - analytic.data()[i]) > tol) {
			return false;
		}
	}
	return true;
}
