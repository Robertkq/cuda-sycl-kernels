# gemm

Multiplies two float matrices: `C = A × B`, with A of size rows × inner, B of size inner × cols, and C of size rows × cols. Each output `C[r][c]` is the dot product of row r of A and column c of B.

## How it's computed

- **Naive:** one work-item per output element. Each one reads a full row of A and a full column of B from global memory. Neighbouring work-items read the same values again, so most global reads are repeated.
- **Tiled:** C is split into 64 × 64 blocks, one per work-group. The work-group walks along `inner` in 64-wide chunks. For each chunk it copies a 64 × 64 tile of A and a 64 × 64 tile of B into local memory, waits on a barrier, then does the multiply-adds from local memory. Each value loaded from global memory is reused 64 times.
- **Register blocking:** inside a work-group, each work-item computes a 4 × 4 block of outputs instead of one. For each step along the chunk it reads 4 values of A and 4 values of B from local memory into registers and does 16 multiply-adds with them, so there are fewer local-memory reads per multiply-add.

The CUDA version follows the same steps: work-group → block, local memory → shared memory, `group_barrier` → `__syncthreads()`.

| Variant | CUDA | SYCL |
|---|---:|---:|
| Naive | — | 730.5 GFLOP/s (188.132 ms) |
| Optimized | — | 2442.3 GFLOP/s (56.275 ms) |
| Speedup | — | 3.34× |

Measured with rows = cols = inner = 4096 (the defaults), median of 3 alternating rounds of 10 iterations. FLOPs count one multiply and one add per inner step for each output.
