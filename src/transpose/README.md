# transpose

This kernel does no arithmetic at all, and it's still the one where the access pattern matters most. Reading rows and writing columns means one side is always strided. A small tile in shared/local memory lets both global accesses run row by row.

Implementations: [CUDA](cuda/README.md) · [SYCL](sycl/README.md)

## CUDA vs SYCL

| Variant | CUDA | SYCL | SYCL vs CUDA |
|---|---:|---:|---:|
| Naive | 51.5 GB/s · 20.830 ms | 51.6 GB/s · 20.789 ms | +0.2% |
| Optimized | 157.7 GB/s · 6.807 ms | 157.7 GB/s · 6.808 ms | 0.0% |
| **Speedup** | **3.06×** | **3.05×** | |

Bandwidth counts one read and one write per element. *SYCL vs CUDA* compares median times; positive means SYCL is faster.

## Takeaway

Staging through a tile turns the strided writes into coalesced ones: 3.06× in CUDA and 3.05× in SYCL, with the same kernel structure in both. Transpose moves the same bytes as a copy, so vector_add's 316.1 GB/s is the ceiling it could reach. It sits at 157.7 GB/s now, so there's still room left.

## Where the languages differ

| | CUDA | SYCL |
|---|---|---|
| Fast index | `threadIdx.x` | `get_local_id(1)`: the **last** dimension is the contiguous one |
| Launch size | `grid` = number of blocks, `dim3(x, y)` | global range = total work-items, `range<2>(rows, cols)` |
| Global index | computed yourself: `blockIdx * blockDim + threadIdx` | `get_global_id(d)`, built in |
| Tile | `__shared__ float tile[16][17]` inside the kernel | `local_accessor<float, 2>` created on the handler, captured by the kernel |
| Barrier | `__syncthreads()` | `group_barrier(item.get_group())` |

The fast-index row is the easiest one to get wrong. If you swap dimensions 0 and 1 in SYCL, the kernel still gives the right answer, but neighbouring work-items walk down a column and the coalescing is gone.
