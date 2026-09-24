# saxpy

Computes `out[i] = a * x[i] + y[i]` for two 1D float vectors `x`, `y` and a scalar `a`. The name comes from BLAS: Single-precision A·X Plus Y.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 316.8 GB/s (5.083 ms) | 317.5 GB/s (5.073 ms) |
| Optimized | 317.5 GB/s (5.072 ms) | 317.2 GB/s (5.078 ms) |
| Speedup | 1.00× | 1.00× |

Measured with 2²⁷ elements. Bandwidth counts two reads and one write per element.
