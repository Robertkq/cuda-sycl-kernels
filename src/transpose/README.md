# transpose

Transposes an N × N float matrix: `output[c][r] = input[r][c]`.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 53.6 GB/s (20.028 ms) | 53.6 GB/s (20.031 ms) |
| Optimized | 164.6 GB/s (6.522 ms) | 164.6 GB/s (6.523 ms) |
| Speedup | 3.07× | 3.07× |

Measured with N = 11585 (the largest square that fits in 2²⁷ elements). Bandwidth counts one read and one write per element.
