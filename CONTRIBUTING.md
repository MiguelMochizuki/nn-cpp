# Contributing

## Build and test

```
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
```

Binaries land in `build/apps/` and `build/tests/`. New library sources go under `src/` and `include/nn/` and in the `nn_core` list in `CMakeLists.txt`.

## Documentation Conventions

### C++

Every file starts with a header comment, and every function, class and struct gets a block comment. Indent with tabs. The comment goes on the declaration in the header; the definition in the `.cpp` needs none unless it is a helper the header does not declare.

```cpp
/**
 * file_name.hpp
 * Author: Miguel Mochizuki Silva
 * Description: One line on what the file holds
 */

/**
 * What the function does, in a sentence
 *
 * Parameters:
 * const Tensor& a: Left operand
 * float scale: What the value means
 *
 * Returns Tensor: What the result is
 */
Tensor scaled(const Tensor& a, float scale);

/**
 * What the type is for
 *
 * Attributes:
 * size_t rows: Number of rows
 */
```

Trivial getters need no block. Comments inside functions say why, not what.
