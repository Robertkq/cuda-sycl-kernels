# reduction

This kernel turns 134,217,728 numbers into one. Every thread depends on every other thread's result, which makes it the first kernel here where threads have to cooperate through shared or local memory and barriers.

Implementations: [CUDA](cuda/README.md) · [SYCL](sycl/README.md)

## CUDA vs SYCL

| Variant | CUDA | SYCL | SYCL vs CUDA |
|---|---:|---:|---:|
| Naive | 14.6 GB/s · 36.727 ms | 13.5 GB/s · 39.790 ms | -7.7% |
| Optimized | 106.9 GB/s · 5.062 ms | 106.5 GB/s · 5.080 ms | -0.4% |
| **Speedup** | **7.26×** | **7.83×** | |

Bandwidth counts one read per element, plus the partial-sum traffic for the optimized variant. *SYCL vs CUDA* compares median times; positive means SYCL is faster.

## Takeaway

Replacing 256 atomics on one shared address with an 8-step tree gives 7.26× in CUDA and 7.83× in SYCL. Both optimized versions land within 0.4% of each other. The naive SYCL kernel is 7.7% slower than CUDA, probably because of how the float atomic on local memory gets compiled. That hasn't been investigated yet.

Headroom: at about 30% of peak there's plenty left. Possible next steps: do the first add while loading, handle several elements per thread, and use warp or sub-group reductions for the last 32 values.

## Where the languages differ

| | CUDA | SYCL |
|---|---|---|
| Float atomic | `atomicAdd(float*, float)` | `atomic_ref<float, order, scope, address_space>`: you state the memory order, scope and address space explicitly |
| Ordering two kernels | same stream, runs in order automatically | out-of-order queue by default, needs `depends_on` |
| Built-in group reduction | none in the language: `__shfl_down_sync` or CUB's `BlockReduce` | `sycl::reduce_over_group`, part of the standard |
