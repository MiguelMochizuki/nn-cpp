/**
 * vmath.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Elementwise single-precision math kernels (tanh, exp) with one result per input value
 */

#pragma once

#include <cstddef>

namespace nn {
namespace kernels {

/**
 * exp(x) by range reduction and a degree-5 polynomial, built only from correctly rounded
 * operations (no FMA, no libm), so it returns the same bits on every build and for every position
 * of x in an array. Within 2 ulp of std::exp on [-87.3, 88.3]; below that range it returns 0
 * (the true value is under FLT_MIN), above it infinity, and NaN is returned unchanged.
 *
 * Parameters:
 * float x: Argument
 *
 * Returns float: Approximation of e^x
 */
float exp_one(float x);

/**
 * tanh(x) as (e^2x - 1) / (e^2x + 1) on exp_one, saturating to +-1 beyond |x| = 9. The absolute
 * error against std::tanh stays below 2e-7 (the relative error grows near 0, where the result is
 * a difference of nearly equal numbers). Same bits on every build; NaN is returned unchanged.
 *
 * Parameters:
 * float x: Argument
 *
 * Returns float: Approximation of tanh(x)
 */
float tanh_one(float x);

/**
 * Replaces every element by exp_one of itself. The AVX2 build works on 8 floats at a time and
 * returns exactly the bits of exp_one for each element, so the result does not depend on how an
 * array is split between threads.
 *
 * Parameters:
 * float* x: Elements, overwritten
 * size_t n: Element count
 *
 * Returns void
 */
void exp_inplace(float* x, size_t n);

/**
 * Replaces every element by tanh_one of itself; same guarantees as exp_inplace
 *
 * Parameters:
 * float* x: Elements, overwritten
 * size_t n: Element count
 *
 * Returns void
 */
void tanh_inplace(float* x, size_t n);

}  // namespace kernels
}  // namespace nn
