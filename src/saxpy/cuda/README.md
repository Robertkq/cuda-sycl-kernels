# saxpy (CUDA)

## Naive

One thread per element, in blocks of 256 threads.

## Optimized

Each thread reads and writes a `float4` (4 floats) instead of a single float, so a quarter as many threads are launched.

There's no measurable difference. The naive version already runs at 88% of peak bandwidth, so memory is the limit.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 5.083 ms | 316.8 GB/s | 88% |
| Optimized | 5.072 ms | 317.5 GB/s | 88% |
