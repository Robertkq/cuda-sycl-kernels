# saxpy (SYCL)

## Naive

A `parallel_for` over `range<1>(count)`, one work-item per element. The runtime chooses the work-group size.

## Optimized

Each work-item reads and writes a `sycl::vec<float, 4>` instead of a single float, so a quarter as many work-items are launched. The buffers are allocated with `aligned_alloc_device(16, …)` so that the 16-byte accesses are aligned.

There's no measurable difference. The naive version already runs at 88% of peak bandwidth, so memory is the limit.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 5.073 ms | 317.5 GB/s | 88% |
| Optimized | 5.078 ms | 317.2 GB/s | 88% |
