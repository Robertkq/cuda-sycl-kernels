# cuda-sycl-kernels

Common GPU kernels written in both CUDA and SYCL. Each one has a naive and an optimized version, and all of them are benchmarked on the same GPU.

## Results

Median bandwidth in GB/s on an RTX 3060, whose theoretical peak is 360 GB/s:

| Kernel | CUDA naive | CUDA optimized | SYCL naive | SYCL optimized |
|---|---:|---:|---:|---:|
| [vector_add](src/vector_add) | 333.6 | 333.7 | 333.4 | 333.5 |
| [reduction](src/reduction) | 15.3 | 113.1 | 14.2 | 112.7 |
| [transpose](src/transpose) | 53.6 | 164.6 | 53.6 | 164.6 |

## Kernels

- [vector_add](src/vector_add): element-wise sum of two vectors
- [reduction](src/reduction): sum of all elements of a vector
- [transpose](src/transpose): transpose of a square matrix

Planned next: GEMM, histogram, stencil, prefix sum, bitonic sort.

Each kernel has `cuda/` and `sycl/` folders containing `naive.cpp` and `optimized.cpp`.

## Measurement

- Kernel time is measured on the GPU, using CUDA events and SYCL event profiling.
- Each benchmark does 5 warm-up runs and 50 timed runs, and reports the median. This was done in 3 rounds, and the tables use the median of the three. Results varied by up to 12.2% between rounds.
- Bandwidth is the number of bytes the kernel has to read and write, divided by its time.
- All results use the default input size of 2²⁷ elements (set with `--count`).
- Setup: RTX 3060 12 GB, driver 615.71.09, CUDA 13.4, DPC++ 7.2.0 (intel/llvm with the CUDA backend), measured on 2026-09-24.

## Build and run

```sh
cmake -B build && cmake --build build
./build/transpose-optimized-cuda -i 50 -w 5
```

Binaries are named `<kernel>-<variant>-<lang>`. Use `--help` for the options. Setting up the toolchain is described in [BUILD.md](BUILD.md).

## License

[MIT](LICENSE)
