# vector_add · SYCL

`out[i] = lhs[i] + rhs[i]` over 134,217,728 floats.

## Naive

[naive.cpp](naive.cpp): a plain `parallel_for` over `range<1>(count)`, one work-item per element. No work-groups, local memory or barriers are needed, so it uses the simple `range`/`id` form. The runtime picks the group size, and there's no bounds check because the range is exactly `count`.

## Optimized

[optimized.cpp](optimized.cpp)

- **`sycl::vec<float, 4>` loads and stores.** Each work-item moves 16 bytes instead of 4, so a quarter of the work-items and memory instructions move the same data. The element count must be a multiple of 4.
- **`aligned_alloc_device(16, …)`.** `malloc_device<float>` only promises alignment for a `float` (4 bytes), and a 16-byte vector load needs 16.

**Result: +0.2%, which is within noise.** The naive kernel already runs at 88% of peak. The bottleneck is DRAM, not instruction count, so issuing fewer instructions doesn't help.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 5.074 ms | 317.4 GB/s | 88% |
| Optimized | 5.064 ms | 318.0 GB/s | 88% |
| **Speedup** | **1.00×** | | |
