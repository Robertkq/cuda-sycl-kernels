# reduction (SYCL)

## Naive

A single kernel over an `nd_range<1>` with work-groups of 256. Each work-item adds its element to one float in local memory, using an `atomic_ref`. Work-item 0 then adds the group's total to the global result, again with an `atomic_ref`.

## Optimized

Two kernels, and no atomics:

1. Each work-group loads 256 elements into local memory and sums them with a tree reduction. Each step halves the number of active work-items, so it takes 8 steps. The group writes out one partial sum.
2. A single work-group sums the partial sums the same way. The second kernel is submitted with `depends_on` on the first, because the queue is out-of-order.

It's 7.88× faster than naive.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 37.839 ms | 14.2 GB/s | 4% |
| Optimized | 4.801 ms | 112.7 GB/s | 31% |
