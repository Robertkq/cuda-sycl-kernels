# reduction · CUDA

Sum of 134,217,728 floats, all `1.0f`, so the expected result is exactly the element count.

## Naive

[naive.cpp](naive.cpp): one kernel. Each block zeroes a `__shared__` float, and every thread `atomicAdd`s its element into it. Thread 0 then `atomicAdd`s the block total into the global sum. All 256 threads of a block hit the **same shared address**, so their atomics run one after another.

## Optimized

[optimized.cpp](optimized.cpp): two passes, no atomics.

- **Tree reduction in shared memory.** Each block loads 256 elements into `sdata`. Each step halves the number of active threads (128, 64, … 1) and adds pairs, so the sum takes 8 steps instead of 256 serialized atomics.
- **Sequential addressing.** The active threads are the first `stride` ones and read `sdata[tid + stride]`. That means contiguous accesses with no bank conflicts, and whole warps go idle together instead of diverging, until `stride` < 32.
- **Second kernel instead of global atomics.** The first pass writes one partial sum per block (524,288 of them). A single 256-thread block then adds those up with a strided loop, followed by the same tree.

**Result: 7.26× faster.** At 106.9 GB/s (30% of peak), it's still well below vector_add's 316.1 GB/s. Half the threads are idle from the first tree step, and each thread handles only one element.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 36.727 ms | 14.6 GB/s | 4% |
| Optimized | 5.062 ms | 106.9 GB/s | 30% |
| **Speedup** | **7.26×** | | |

The optimized variant also counts writing and re-reading the partial sums, which is 0.8% more bytes. The speedup is based on time.
