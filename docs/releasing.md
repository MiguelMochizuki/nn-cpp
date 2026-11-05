# Releasing

Versions follow Semantic Versioning: breaking API change, major; new feature, minor; fix, patch. Tags are `vX.Y.Z`.

1. Branch from main, for example `release/vX.Y.Z`.
2. Set the new version in `CMakeLists.txt`: `project(nn_cpp VERSION X.Y.Z LANGUAGES CXX)`. The workflow reads the version from that line, so keep its format.
3. In `CHANGELOG.md`, add a `## [X.Y.Z] - YYYY-MM-DD` section for the changes since the last release.
4. Check locally:
   ```bash
   cmake -S . -B build && cmake --build build -j
   ctest --test-dir build --output-on-failure
   cmake --install build --prefix /tmp/nn-prefix
   ```
5. Open a pull request and wait for the `Build and Release` workflow to pass on all five platforms. Merge it to main.
6. Tag the merge commit on main and push the tag:
   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "nn-cpp X.Y.Z"
   git push origin vX.Y.Z
   ```

## What the workflow does

On a `v*` tag, `.github/workflows/build-and-release.yml`:

- fails if the tag is not equal to `v` plus the CMake project version;
- builds Release with `NN_BUILD_APPS=ON` and `NN_NATIVE=OFF` and runs `ctest` on Linux x86-64, Linux ARM64, Windows x86-64, Windows ARM64 and macOS (universal, x86_64 and arm64);
- runs `cmake --install` into a staging directory and packages it as `nn-cpp-X.Y.Z-<os>-<arch>.tar.gz` (`.zip` on Windows), with the platforms `linux-x86_64`, `linux-arm64`, `windows-x86_64`, `windows-arm64` and `macos-universal`;
- creates the GitHub release with those five packages, a source archive `nn-cpp-X.Y.Z-source.tar.gz` and generated release notes.

Pushes to main, pull requests and manual runs build, test and upload the packages as workflow artifacts, but publish nothing. If the tag check fails, fix the version, delete the tag locally and on origin, and tag again.

## Release notes

Generated notes list the merged changes. Add these standing notes by editing the release if they still apply: MSVC builds on x86 use the unpacked `sgemm` (no `__SSE2__`/`__AVX2__`/`__FMA__` without `/arch` flags), which is correct but slower than the GCC/Clang x86-64 builds. A package is built without `-march=native`.
