# transpose (SYCL)

## Naive

One work-item per element, over an `nd_range<1>` with work-groups of 256. Reads from the input are coalesced. Writes are not, because neighbouring work-items write to different rows of the output.

## Optimized

A 2D `nd_range` with 16 × 16 work-groups, each handling one 16 × 16 tile:

- A work-group reads its tile into a `local_accessor` row by row. After `group_barrier`, it writes the tile to the transposed position in the output, again row by row. Both the reads and the writes are coalesced.
- The local tile is 16 × 17. The extra column avoids bank conflicts when the tile is read by column. On this GPU it makes no measurable difference: 157.8 GB/s without it.

It's 3.06× faster than naive.

## Results

| Variant | Time | Bandwidth | % of peak (360 GB/s) |
|---|---:|---:|---:|
| Naive | 20.770 ms | 51.7 GB/s | 14% |
| Optimized | 6.798 ms | 157.9 GB/s | 44% |
