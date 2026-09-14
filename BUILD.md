# Build

## Requirements

- CMake 3.24+
- CUDA Toolkit (`nvcc` on PATH) — for the CUDA targets
- An NVIDIA GPU and driver — to run either backend
- A SYCL compiler with the CUDA backend enabled — for the SYCL targets
- CLI11 — not packaged on most distros, build and install it yourself:

      git clone https://github.com/CLIUtils/CLI11.git
      cmake -B CLI11/build -S CLI11 -DCLI11_BUILD_TESTS=OFF
      cmake --build CLI11/build --target install

## SYCL compiler

The `icpx` that ships with oneAPI does not have a CUDA backend — it only
carries the Level Zero and OpenCL adapters. You need a build of `intel/llvm`
(the project `icpx` itself is built from) configured with `--cuda`:

    git clone https://github.com/intel/llvm.git
    cd llvm
    python buildbot/configure.py --cuda
    python buildbot/compile.py

That produces `build/bin/clang++`. Confirm the CUDA backend is actually
there before pointing the project at it:

    build/bin/sycl-ls
    # should print a [cuda:gpu] line for your GPU

If it only lists OpenCL/CPU, see "oneAPI gets in the way" below before
assuming the build is broken.

## Configuring

    cmake -B build
    cmake --build build

CUDA and SYCL targets are each built automatically if their compiler is
found, and skipped with a warning otherwise — same as any normal CMake
`check_language()` detection. `SYCL_CUDA_ROOT` (default
`~/projects/llvm/build`) is where the SYCL side looks for `bin/clang++`;
override it if yours lives elsewhere:

    cmake -B build -DSYCL_CUDA_ROOT=/path/to/intel-llvm/build

Each kernel has its own `CMakeLists.txt` under `src/<kernel>/`, building
`<kernel>_<variant>_cuda` and `<kernel>_<variant>_sycl` into `build/`.
Either target is skipped if its compiler isn't available, so a missing
executable usually just means that variant isn't implemented yet (see
`CLAUDE.md`'s kernel backlog), not a build failure.

## Running

No environment activation needed. The SYCL binaries carry an rpath to the
compiler's own runtime libs, so they resolve the right `libsycl.so` and CUDA
adapter regardless of shell state, including a shell with oneAPI's
`setvars.sh` sourced.

### oneAPI gets in the way

Running `sycl-ls` or `clang++` *directly* (outside this build) in a shell
that has also sourced oneAPI's `setvars.sh` will silently hide your GPU:
oneAPI's `LD_LIBRARY_PATH` entries shadow your build's `libsycl.so`, and
oneAPI's own runtime has no CUDA adapter. Source your build's env script
*after* oneAPI's, or just don't mix the two in the same shell.

### CUDA_ERROR_UNSUPPORTED_PTX_VERSION

Means your driver is older than the CUDA toolkit's PTX ISA version. This
project targets your GPU's real architecture directly instead of embedding
PTX (`CMAKE_CUDA_ARCHITECTURES=native` for CUDA, `nvptx-arch` output for
SYCL), which avoids this by default. If you still hit it, check that
`nvidia-smi` and `nvptx-arch` actually resolve on your machine — CMake
falls back to less specific defaults when they don't.

## Recommended setup

Keep the oneAPI install and your `intel/llvm --cuda` build separate and
don't try to make `icpx` work for this project — it structurally can't.
Build `intel/llvm` once, point `-DSYCL_CUDA_ROOT` at it if it's not at the
default path, and forget about shell setup entirely: the rpath handling
exists so you never need to activate anything just to build or run.
