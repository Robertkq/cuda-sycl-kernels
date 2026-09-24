# vector_add (CUDA)

## Naive

One thread per element, in blocks of 256 threads.

## Optimized

Each thread reads and writes a `float4` (4 floats) instead of a single float, so a quarter as many threads are launched.

There's no measurable difference. The naive version already runs at 93% of peak bandwidth, so memory is the limit.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 4.828 ms | 333.6 GB/s | 93% |
| Optimized | 4.827 ms | 333.7 GB/s | 93% |
