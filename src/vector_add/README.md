# vector_add

Adds two 1D float vectors element by element: `out[i] = lhs[i] + rhs[i]`.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | 333.6 GB/s (4.828 ms) | 333.4 GB/s (4.831 ms) |
| Optimized | 333.7 GB/s (4.827 ms) | 333.5 GB/s (4.829 ms) |
| Speedup | 1.00× | 1.00× |

Measured with 2²⁷ elements. Bandwidth counts two reads and one write per element.
