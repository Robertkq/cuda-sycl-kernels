# reduction

Sums a 1D float vector into a single value.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 14.7 GB/s (36.641 ms) | 13.6 GB/s (39.542 ms) |
| Optimized | 107.7 GB/s (5.026 ms) | 107.3 GB/s (5.044 ms) |
| Speedup | 7.29× | 7.84× |

Measured with 2²⁷ elements. Bandwidth counts one read per element. The optimized version also counts writing and reading its partial sums (one per block), so the speedup is computed from time.
