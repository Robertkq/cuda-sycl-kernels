# cuda-sycl-kernels

The same GPU kernels written twice, once in **CUDA** and once in **SYCL**, and each of those twice again: a **naive** version and an **optimized** one. They're all benchmarked on the same card. The aim isn't the fastest possible kernel. It's to see what each optimization actually buys, and where the two languages really differ.

## Results

Median bandwidth on an RTX 3060, against its 360 GB/s theoretical peak:

```text
                          0                                360 GB/s
                          |                                  |
vector_add naive     cuda ███████████████████████████████▋      316.1
           naive     sycl ███████████████████████████████▊      317.4
           optimized cuda ███████████████████████████████▋      316.5
           optimized sycl ███████████████████████████████▊      318.0

reduction  naive     cuda █▌                                     14.6
           naive     sycl █▍                                     13.5
           optimized cuda ██████████▊                           106.9
           optimized sycl ██████████▋                           106.5

transpose  naive     cuda █████▏                                 51.5
           naive     sycl █████▏                                 51.6
           optimized cuda ███████████████▊                      157.7
           optimized sycl ███████████████▊                      157.7
```

| Kernel | CUDA naive → optimized | SYCL naive → optimized | Speedup (CUDA · SYCL) |
|---|---:|---:|---:|
| [vector_add](src/vector_add) | 316.1 → 316.5 GB/s | 317.4 → 318.0 GB/s | 1.00× · 1.00× |
| [reduction](src/reduction) | 14.6 → 106.9 GB/s | 13.5 → 106.5 GB/s | 7.26× · 7.83× |
| [transpose](src/transpose) | 51.5 → 157.7 GB/s | 51.6 → 157.7 GB/s | 3.06× · 3.05× |

A few things stand out:

- **The access pattern matters more than the arithmetic.** Transpose does no math and still gets 3.06× faster, just from changing which thread reads and writes which element.
- **Not every optimization pays off.** Vectorized `vector_add` gains +0.1%. The naive version was already at 88% of peak, and a DRAM-bound kernel can't go faster by issuing fewer instructions.
- **CUDA vs SYCL makes no speed difference once the kernels are optimized.** Every optimized pair lands within 0.5% of each other. The differences are in how the code is written, and each kernel's README has a table of them.

## Kernels

| # | Kernel | What it's about | |
|---|---|---|---|
| 1 | [vector_add](src/vector_add) | the bandwidth ceiling | ✅ |
| 2 | [reduction](src/reduction) | shared memory, barriers, tree reduction | ✅ |
| 3 | [transpose](src/transpose) | coalescing, tiling, bank conflicts | ✅ |
| 4 | GEMM | tiling, occupancy trade-offs | |
| 5 | histogram | atomics, local-then-global reduction | |
| 6 | stencil / convolution | halo handling | |
| 7 | prefix sum / scan | | |
| 8 | bitonic sort | heavy synchronization | |

Each kernel directory holds `cuda/` and `sycl/`, each containing `naive.cpp` and `optimized.cpp` plus a README that explains what changed and why.

## How it's measured

- **Timed on the GPU**: CUDA events and SYCL queue profiling (`command_start` → `command_end`). No host clock is involved.
- **5 warm-up runs, then 50 timed runs, reporting the median.** Results are verified on the host every 10th run.
- **Bandwidth** = bytes the kernel must move ÷ median time. It's always compared against the card's peak, never reported as raw milliseconds alone.
- **Size**: 134,217,728 floats (2²⁷, 512 MiB per buffer).

Hardware and software: RTX 3060 12 GB · driver 615.71.09 · CUDA 13.4 · DPC++ 7.2.0 (intel/llvm, CUDA backend). Measured on 2026-09-24.

## Build & run

```sh
cmake -B build && cmake --build build

./build/transpose-optimized-cuda              # defaults: 2^27 elements, 10 runs
./build/transpose-optimized-sycl -i 50 -w 5   # the settings used above
./build/reduction-naive-cuda --json           # write the summary to reduction-naive-cuda.json
./scripts/sweep.py --kernel transpose         # every variant across a range of sizes
```

Binaries are named `<kernel>-<variant>-<lang>`. Setting up the toolchain, and especially a SYCL compiler with a CUDA backend, is covered in [BUILD.md](BUILD.md).

## License

[MIT](LICENSE)
