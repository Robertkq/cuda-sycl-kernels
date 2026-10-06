# Paper plan

Working question: **how much of a CUDA optimization ladder carries over to SYCL on NVIDIA hardware?**

Each kernel exists in CUDA and SYCL, as a sequence of optimization steps, each measured on the same GPU. The paper compares the two languages step by step, and compares the final versions against the vendor libraries.

Possible venues: a technical report on arXiv first; [IWOCL & SYCLcon](https://www.iwocl.org/) if the results hold up.

## Where things stand (2026-10-06)

- saxpy, reduction, transpose: naive + optimized in both languages. CUDA and SYCL land within a few percent of each other.
- GEMM: SYCL naive, SYCL v1 (16 × 16 tiles, 1 × 4 outputs per work-item), and an optimized SYCL kernel written by Claude (not committed, used as a reference). No CUDA GEMM yet.
- Benchmark reports time, GB/s and GFLOP/s, plus JSON output. `scripts/sweep.py` runs every binary across sizes.

### GEMM ladder measured so far (SYCL, RTX 3060)

Each gain was measured against the previous step, with the variants run in alternating rounds within one session. Absolute numbers drift between sessions, so these are relative.

| Step | Gain |
|---|---|
| 2D register tiling (128 × 128 block, 8 × 8 per work-item), float4 loads, A transposed in local memory | ~7–8× over v1 |
| Work-item columns split into groups of 4 (bank conflicts) | +6–9% |
| Double buffering, BK = 8 | +2% (−10% with BK = 16: 153 registers) |
| Block swizzle, band height 4 | +3% |
| 256 × 128 block, 16 × 8 per work-item | +8% |
| Pad M, N and K separately instead of all to the block size | +4% at N = 11585 |

Final kernel against cuBLAS SGEMM (FP32, measured in alternating rounds): 1.04× at N = 4096, 0.82× at N = 8192, 0.88× at N = 11585, about 65% of the 12.7 TFLOP/s FP32 peak.

## Work, in order

Owner in brackets. Kernel code is the user's, so the learning stays with the person writing the paper. Tooling is Claude's.

Breadth comes before depth: GEMM optimization pauses until more kernels exist. Kernels that use features where CUDA and SYCL differ come first, since that's where the interesting results are.

1. **Tooling**
   - [ ] [Claude] Benchmark protocol script: alternating rounds, several rounds per binary, median + spread, one JSON per session
   - [ ] [Claude] Record GPU clocks and power during each run; check whether clocks can be locked (`nvidia-smi -lgc`, needs root)
   - [x] [Claude] Kernel-specific CLI options: a `Benchmark` constructor overload taking a callback that adds options through `BenchmarkOptions` before parsing (see `CLI_PLAN.md`)
2. **Histogram** [user]: naive + optimized, CUDA and SYCL. Atomics: `atomicAdd` vs `atomic_ref` with explicit memory order and scope.
3. **Scan** [user]: naive + optimized, CUDA and SYCL. Warp-level operations: `__shfl_*_sync` vs sub-group functions.
4. **Reduction v2** [user]: a third variant using warp shuffles (CUDA) and `reduce_over_group` (SYCL), which has no direct CUDA equivalent.
5. **GEMM ladder** [user]
   - [ ] SYCL ladder rebuilt as separate, committed steps (naive → v1 → 2D tiling → … → final)
   - [ ] CUDA version of each step, using the SYCL reference kernel as a map
6. **Stencil, bitonic sort** [user], if time allows.
7. **Profiling and baselines**
   - [ ] [user] Enable profiler counters once (driver option `NVreg_RestrictProfilingToAdminUsers=0`) so Nsight Compute can run without root
   - [ ] [Claude] Nsight Compute collection per kernel: bank conflicts, occupancy, DRAM bytes, achieved FLOP/s
   - [ ] [Claude] Library baselines: cuBLAS for GEMM, CUB for reduction and scan, `cudaMemcpy` for the bandwidth ceiling
8. **Results**
   - [ ] [Claude] Pipeline from benchmark JSON to tables and figures (extend the README generator)
   - [ ] [Claude] Roofline plot: every kernel and step against the memory and compute limits of the card
   - [ ] [user] Write the paper: method, results per kernel, where CUDA and SYCL differ and why

## More hardware

More GPUs would change the question from "SYCL vs CUDA" to "SYCL vs each vendor's native language", which is the case SYCL is made for. Options to look into (check what is currently offered, these programs change):

- **Intel Tiber AI Cloud** (formerly Intel Developer Cloud): has had a free tier with Intel GPUs and DPC++ preinstalled.
- **AMD Developer Cloud / AMD University Program**: AMD GPUs; SYCL via DPC++'s HIP backend or AdaptiveCpp, compared against HIP.
- **NVIDIA Academic Grant Program**: cloud credits or hardware; needs a university affiliation and a proposal.
- **Kaggle / Google Colab**: free T4 or P100 hours per week; fine for CUDA, awkward for building DPC++.
- **University HPC cluster**, if available.

## Rules for numbers in the paper

- Only compare variants measured in the same session, run in alternating rounds.
- Report the median and the spread, never a single run.
- State GPU, driver, CUDA and DPC++ versions, clocks, and matrix sizes with every table.
- FLOPs and bytes count only useful work, not padding.
- Every kernel passes verification at the sizes it is reported at.

## Open questions

- Which sizes to report: powers of two only, or also awkward sizes like 11585?
- Should non-square GEMM shapes be in scope?
- Is TF32 / tensor-core GEMM in scope? It's not FP32, so it would be a separate section at most.
- With more hardware: which native language to compare SYCL against on each vendor (CUDA, HIP, Level Zero / plain SYCL on Intel)?
