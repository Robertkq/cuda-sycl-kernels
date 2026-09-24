# vector_add

The baseline: two loads, one add and one store per element. There's no reuse and nothing to rearrange, so it measures how close a simple kernel gets to the card's memory bandwidth. Every other kernel here is compared against that ceiling.

Implementations: [CUDA](cuda/README.md) · [SYCL](sycl/README.md)

## CUDA vs SYCL

| Variant | CUDA | SYCL | SYCL vs CUDA |
|---|---:|---:|---:|
| Naive | 316.1 GB/s · 5.096 ms | 317.4 GB/s · 5.074 ms | +0.4% |
| Optimized | 316.5 GB/s · 5.088 ms | 318.0 GB/s · 5.064 ms | +0.5% |
| **Speedup** | **1.00×** | **1.00×** | |

Bandwidth counts 3 × 4 bytes per element (read `lhs` and `rhs`, write `out`). *SYCL vs CUDA* compares median times; positive means SYCL is faster.

## Takeaway

Both languages reach about 88% of the 360 GB/s peak **with the naive kernel**. Vectorizing to 16-byte accesses changes nothing measurable, because the kernel is limited by DRAM, not by how many instructions it issues. Its roughly 316.1 GB/s is the practical ceiling that the other kernels are measured against.

## Where the languages differ

| | CUDA | SYCL |
|---|---|---|
| 4-wide type | `float4`, built-in | `sycl::vec<float, 4>` |
| Alignment for 16-byte access | `cudaMalloc` is 256-byte aligned | needs `aligned_alloc_device(16, …)` |
| Launch | grid rounded up + bounds check | `range<1>(count)`: exact size, no check |
