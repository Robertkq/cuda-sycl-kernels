# reduction · SYCL

Sum of 134,217,728 floats, all `1.0f`, so the expected result is exactly the element count.

## Naive

[naive.cpp](naive.cpp): one kernel over an `nd_range<1>` with 256-item work-groups. Each group zeroes a one-float `local_accessor`, and every work-item adds its element to it through an `atomic_ref` (scope `work_group`, `local_space`). Work-item 0 then adds the group total to the global sum with a second `atomic_ref` (scope `device`, `global_space`). All 256 work-items hit the **same local address**, so their atomics run one after another.

## Optimized

[optimized.cpp](optimized.cpp): two passes, no atomics.

- **Tree reduction in local memory.** Each work-group loads 256 elements into `sdata`. Each step halves the number of active work-items (128, 64, … 1) and adds pairs, so the sum takes 8 steps instead of 256 serialized atomics.
- **Sequential addressing.** The active work-items are the first `stride` ones and read `sdata[localId + stride]`. That means contiguous accesses with no bank conflicts, and whole sub-groups go idle together instead of diverging, until `stride` < 32.
- **Second kernel instead of global atomics.** The first pass writes one partial sum per group (524,288 of them). A single 256-item group then adds those up with a strided loop, followed by the same tree. The second submit uses `depends_on(firstTreeReduction)`, because a default SYCL queue is out-of-order.

**Result: 7.83× faster.** At 106.5 GB/s (30% of peak), it's still well below vector_add's 317.4 GB/s. Half the work-items are idle from the first tree step, and each work-item handles only one element.

## Results

| Variant | Median time | Bandwidth | % of 360 GB/s peak |
|---|---:|---:|---:|
| Naive | 39.790 ms | 13.5 GB/s | 4% |
| Optimized | 5.080 ms | 106.5 GB/s | 30% |
| **Speedup** | **7.83×** | | |

The optimized variant also counts writing and re-reading the partial sums, which is 0.8% more bytes. The speedup is based on time. The optimized time runs from the first kernel's start to the second kernel's end.
