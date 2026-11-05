/**
 * clip.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Gradient clipping by global norm and by value
 */

#pragma once

#include "nn/autograd/value.hpp"

#include <vector>

namespace nn {

/**
 * Rescales all gradients together when their global L2 norm exceeds max_norm. Parameters
 * without a gradient are skipped.
 *
 * Parameters:
 * const std::vector<Value>& params: parameters whose grads are clipped in place
 * float max_norm: largest allowed total norm
 *
 * Returns float: the total L2 norm before clipping
 */
float clip_grad_norm(const std::vector<Value>& params, float max_norm);

/**
 * Clamps every gradient element to [-clip, clip]. Parameters without a gradient are skipped.
 *
 * Parameters:
 * const std::vector<Value>& params: parameters whose grads are clipped in place
 * float clip: largest allowed magnitude
 *
 * Returns void: none
 */
void clip_grad_value(const std::vector<Value>& params, float clip);

}  // namespace nn
