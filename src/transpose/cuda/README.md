# transpose (CUDA)

## Naive

One thread per element, on a 1D grid of 256-thread blocks. Reads from the input are coalesced. Writes are not, because neighbouring threads write to different rows of the output.

## Optimized

Blocks of 16 × 16 threads, each handling one 16 × 16 tile:

- A block reads its tile into shared memory row by row. After `__syncthreads()`, it writes the tile to the transposed position in the output, again row by row. Both the reads and the writes are coalesced.
- The shared tile is 16 × 17. The extra column avoids bank conflicts when the tile is read by column. On this GPU it makes no measurable difference: 157.8 GB/s without it.

It's 3.05× faster than naive.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 20.782 ms | 51.7 GB/s | 14% |
| Optimized | 6.803 ms | 157.8 GB/s | 44% |
