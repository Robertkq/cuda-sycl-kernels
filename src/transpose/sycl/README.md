# transpose (SYCL)

## Naive

One work-item per element, over an `nd_range<1>` with work-groups of 256. Reads from the input are coalesced. Writes are not, because neighbouring work-items write to different rows of the output.

## Optimized

A 2D `nd_range` with 16 × 16 work-groups, each handling one 16 × 16 tile:

- A work-group reads its tile into a `local_accessor` row by row. After `group_barrier`, it writes the tile to the transposed position in the output, again row by row. Both the reads and the writes are coalesced.
- The local tile is 16 × 17. The extra column avoids bank conflicts when the tile is read by column. On this GPU it makes no measurable difference: 164.7 GB/s without it.

It's 3.07× faster than naive.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 20.031 ms | 53.6 GB/s | 15% |
| Optimized | 6.523 ms | 164.6 GB/s | 46% |
