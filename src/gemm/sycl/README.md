# gemm (SYCL)

## Naive

One work-item per output element, over a 2D `nd_range` with 16 × 16 work-groups. Each work-item loops over `inner` and reads A and B directly from global memory.

## Optimized

A 2D `nd_range` with 16 × 16 work-groups, each computing one 64 × 64 block of C:

- For each 64-wide chunk of `inner`, the work-group loads a 64 × 64 tile of A and of B into two `local_accessor`s. Each work-item loads a 4 × 4 piece of each tile. Values outside the matrix are loaded as 0, so sizes that aren't multiples of 64 work.
- After `group_barrier`, each work-item computes a 4 × 4 block of outputs. For each of the 64 steps it reads 4 values from its rows of tile A and 4 from its columns of tile B into registers, then does 16 multiply-adds.
- A second `group_barrier` stops the next chunk's loads from overwriting a tile that other work-items are still reading.

It's 3.34× faster than naive.

## Results

| Variant | Time | GFLOP/s | % of peak (12.7 TFLOP/s) |
|---|---:|---:|---:|
| Naive | 188.132 ms | 730.5 | 6% |
| Optimized | 56.275 ms | 2442.3 | 19% |
