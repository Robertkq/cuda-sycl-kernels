# transpose · CUDA

`output[c][r] = input[r][c]` for an 11585 × 11585 float matrix (⌊√134,217,728⌋ per side).

## Naive

[naive.cpp](naive.cpp): a 1D launch with one thread per element. The flat index is split into `rowIndex` and `colIndex`. Reads walk along an input row, so they're coalesced. Writes land in a different output row for each thread, so one warp store touches 32 separate 32-byte sectors and uses only 4 bytes of each.

## Optimized

[optimized.cpp](optimized.cpp)

- **Shared-memory tile (16 × 16).** Each block loads one tile row by row, which is coalesced. After a barrier, it writes the tile to the mirrored tile position `(blockIdx.x, blockIdx.y)` → `(blockIdx.y, blockIdx.x)`, again row by row. The column-wise access now happens in shared memory, which is on-chip and wastes no sectors.
- **+1 padding: `tile[16][17]`.** Without it, every element of a tile column falls in the same bank or two of shared memory's 32 banks, so the column read queues up. The extra float shifts each row by one bank.
- **2D launch.** A grid of 725 × 725 blocks of 16 × 16 threads. `x` is the column (the fast index) and `y` is the row. Bounds checks wrap the memory accesses instead of returning early, because every thread must reach `__syncthreads()`.

**Result: 3.06× faster**, at 44% of peak. Possible next step: a 32 × 32 tile with 32 × 8 threads (4 elements per thread), so that a full warp covers one 128-byte tile row.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 20.830 ms | 51.5 GB/s | 14% |
| Optimized | 6.807 ms | 157.7 GB/s | 44% |
| **Speedup** | **3.06×** | | |
