# reduction

Sums a 1D float vector into a single value.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 15.3 GB/s (35.172 ms) | 14.2 GB/s (37.839 ms) |
| Optimized | 113.1 GB/s (4.783 ms) | 112.7 GB/s (4.801 ms) |
| Speedup | 7.35× | 7.88× |

Measured with 2²⁷ elements. Bandwidth counts one read per element. The optimized version also counts writing and reading its partial sums (one per block), so the speedup is computed from time.
