# vector_add · CUDA

`out[i] = lhs[i] + rhs[i]` over 134,217,728 floats.

## Naive

[naive.cpp](naive.cpp): one thread per element, 256 threads per block, a bounds check for the last partial block. Neighbouring threads touch neighbouring floats, so every warp load and store is already coalesced.

## Optimized

[optimized.cpp](optimized.cpp)

- **`float4` loads and stores.** Each thread moves 16 bytes instead of 4, so a quarter of the threads and memory instructions move the same data. The element count must be a multiple of 4. `cudaMalloc` already returns 256-byte-aligned memory.

**Result: +0.1%, which is within noise.** The naive kernel already runs at 88% of peak. The bottleneck is DRAM, not instruction count, so issuing fewer instructions doesn't help.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 5.096 ms | 316.1 GB/s | 88% |
| Optimized | 5.088 ms | 316.5 GB/s | 88% |
| **Speedup** | **1.00×** | | |
