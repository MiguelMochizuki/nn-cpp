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

### Python

Google-style docstrings. Every module gets a docstring; add `Args:`/`Returns:` (and `Raises:` where it applies) to functions with non-obvious logic. Trivial helpers need none. Type hints carry the types, the docstring carries the meaning.

```python
def resize_image(image: Image, width: int, height: int, keep_aspect: bool = True) -> Image:
    """Resize an image to the given dimensions.

    Args:
        image: The source image to resize.
        width: Target width in pixels.
        height: Target height in pixels.
        keep_aspect: If True, preserve the original aspect ratio and
            letterbox rather than stretch.

    Returns:
        A new Image at the requested size.

    Raises:
        ValueError: If width or height is not positive.
    """
```

Module docstring:

```python
"""Utilities for resizing and format-converting images."""
```
