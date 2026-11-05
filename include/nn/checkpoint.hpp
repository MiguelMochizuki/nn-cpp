/**
 * checkpoint.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Portable binary checkpoints of a Module's parameters and an Optimizer's state
 */

#pragma once

#include "nn/nn/module.hpp"
#include "nn/optim/optimizer.hpp"

#include <cstdint>
#include <string>

namespace nn {

/* Public, write the model's named parameters, the optimizer's state (if given) and a
 * caller-defined step counter to a little-endian binary file. The file is written to
 * path + ".tmp" and renamed over path, so a failed save never destroys an existing checkpoint
 *
 * Parameters:
 * string path: destination file
 * Module model: model whose named_parameters() and then named_buffers() are saved as
 *   "model.<name>"; a buffer name equal to a parameter name throws
 * Optimizer optimizer: optional; its state() is saved as "optim.<name>"
 * int64_t step: caller-defined progress counter (epoch, iteration, ...)
 *
 * Returns void: none; throws std::runtime_error if the file cannot be written
 */
void save_checkpoint(const std::string& path, const Module& model, const Optimizer* optimizer = nullptr,
                     int64_t step = 0);

/* Public, restore a checkpoint into a model built with the same architecture. The whole file
 * is validated against model (and optimizer) before anything is modified, buffers included:
 * a buffer the file lacks fails, naming it, as does any unexpected model.* entry. Layers without
 * parameters leave no trace in the file, so swapping e.g. ReLU for Tanh is not detected
 *
 * Parameters:
 * string path: source file
 * Module model: model whose parameters are overwritten in place
 * Optimizer optimizer: optional; if given, its state is restored too (including lr and
 *   momentum); the file must contain optimizer state. If null, optimizer entries are ignored
 *
 * Returns int64_t: the step stored by save_checkpoint; throws std::runtime_error if the file
 * is missing, malformed, or does not match the model/optimizer (message names the entries)
 */
int64_t load_checkpoint(const std::string& path, Module& model, Optimizer* optimizer = nullptr);

}  // namespace nn
