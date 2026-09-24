# cuda-sycl-kernels

## What this project is

A reference implementation of common GPU kernels in **both CUDA and SYCL**, benchmarked against each other. Every kernel exists in at least two forms: a naive/baseline implementation and an optimized one, so the benchmark results show the impact of each optimization, not just a single data point.

## Kernel list (planned, roughly in order of complexity)

1. Vector add / SAXPY — baseline, bandwidth-bound
2. Reduction (sum/max) — shared/local memory + barrier usage
3. Matrix transpose — naive vs. shared-memory-staged (coalescing)
4. Tiled matrix multiplication (GEMM) — tiling + occupancy trade-offs
5. Histogram — atomics, local-then-global reduction pattern
6. Stencil / convolution — halo handling
7. Prefix sum / scan
8. Bitonic sort — heaviest synchronization pattern

Not every kernel needs to be implemented for the project to be usable — treat this as a backlog, not a blocker.

## Structure expectations

- Keep CUDA and SYCL implementations of the same kernel structurally comparable where reasonable (similar file layout, similar variable naming for equivalent concepts) so the two implementations of a given kernel are easy to diff against each other.
- Each kernel has a naive and an optimized variant, clearly named/separated — not just a single "best" version — since the benchmark results are meant to compare them.
- Benchmarking uses proper GPU-side timing (CUDA events / SYCL queue profiling), not host-side wall-clock timers, and reports against a meaningful baseline (achieved bandwidth or FLOPs vs. device theoretical peak), not just raw milliseconds.
- Warm-up iterations before timed runs; enough repetitions for stable numbers.

## Conventions

- Prefer explicit, readable indexing math over clever one-liners, e.g. `int i = blockIdx.x * blockDim.x + threadIdx.x;` written plainly.
- In SYCL, default to the `nd_range`/`nd_item` form (not the plain `range`/`item` form) once a kernel needs local memory or barriers, to keep the CUDA grid/block ↔ SYCL global/local range mapping visible in the code rather than abstracted away.
- Comment *why* an optimization helps (coalescing, occupancy, bank conflicts, etc.), not just what the code does.
- No premature abstraction across kernels — some duplication between kernel implementations is fine and often clearer than a shared framework, given the project's comparative goal.

## When helping with this project

- If asked to modify a kernel, consider whether the naive/optimized pairing should be preserved or extended, not just edited in place.
- When explaining a performance result or suggesting an optimization, tie it to the underlying GPU execution-model reasoning (memory coalescing, occupancy, divergence, bank conflicts) rather than just stating "this is faster."
- Flag when a CUDA-specific idiom doesn't have a clean SYCL equivalent (or vice versa) rather than silently picking one language's approach — surfacing where the two languages genuinely differ is part of the point.
- Preserve the existing timing/warm-up methodology in benchmark code unless there's a specific reason to change it.
- GPU programming pulls in a lot of unfamiliar vocabulary fast. When new CUDA/SYCL API or types show up in code (e.g. `local_accessor`, `group_barrier`, `nd_range`, `atomic_ref`), define each one in a short, plain sentence the first time it's used, so the codebase stays legible without needing outside references.