# transpose · SYCL

`output[c][r] = input[r][c]` for an 11585 × 11585 float matrix (⌊√134,217,728⌋ per side).

## Naive

[naive.cpp](naive.cpp): an `nd_range<1>` with 256-item work-groups and one work-item per element. The flat id is split into `rowIndex` and `colIndex`. Reads walk along an input row, so they're coalesced. Writes land in a different output row for each work-item, so one sub-group store touches 32 separate 32-byte sectors and uses only 4 bytes of each.

## Optimized

[optimized.cpp](optimized.cpp)

- **Local-memory tile (16 × 16).** Each work-group loads one tile into a `local_accessor` row by row, which is coalesced. After a `group_barrier`, it writes the tile to the mirrored tile position `(groupRow, groupCol)` → `(groupCol, groupRow)`, again row by row. The column-wise access now happens in local memory, which is on-chip and wastes no sectors.
- **+1 padding: `range<2>(16, 17)`.** Without it, every element of a tile column falls in the same bank or two of local memory's 32 banks, so the column read queues up. The extra float shifts each row by one bank.
- **2D `nd_range`.** The global range is 11600 × 11600 work-items (each side rounded up to a multiple of 16), with 16 × 16 work-groups. Dimension 0 is the row and dimension 1 is the column (the fast index). Bounds checks wrap the memory accesses instead of returning early, because every work-item must reach the barrier.

**Result: 3.05× faster**, at 44% of peak. Possible next step: a 32 × 32 tile with 32 × 8 work-items (4 elements per work-item), so that a full sub-group covers one 128-byte tile row.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 20.789 ms | 51.6 GB/s | 14% |
| Optimized | 6.808 ms | 157.7 GB/s | 44% |
| **Speedup** | **3.05×** | | |
