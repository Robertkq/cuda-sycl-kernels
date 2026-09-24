# transpose

Transposes an N × N float matrix: `output[c][r] = input[r][c]`.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 51.7 GB/s (20.782 ms) | 51.7 GB/s (20.770 ms) |
| Optimized | 157.8 GB/s (6.803 ms) | 157.9 GB/s (6.798 ms) |
| Speedup | 3.05× | 3.06× |

Measured with N = 11585 (the largest square that fits in 2²⁷ elements). Bandwidth counts one read and one write per element.
