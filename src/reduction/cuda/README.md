# reduction (CUDA)

## Naive

A single kernel with blocks of 256 threads. Each thread `atomicAdd`s its element into one float in shared memory. Thread 0 then `atomicAdd`s the block's total into the global result.

## Optimized

Two kernels, and no atomics:

1. Each block loads 256 elements into shared memory and sums them with a tree reduction. Each step halves the number of active threads, so it takes 8 steps. The block writes out one partial sum.
2. A single block sums the partial sums the same way.

It's 7.35× faster than naive.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 35.172 ms | 15.3 GB/s | 4% |
| Optimized | 4.783 ms | 113.1 GB/s | 31% |
