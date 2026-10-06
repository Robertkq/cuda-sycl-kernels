# Plan: kernel-specific CLI options and a fixed sweep

Status: **implemented** (2026-10-06). Differences from the original proposal are marked "Changed" below.

## Decisions

- Each kernel declares its own size options. `--count` is no longer a shared option.
- Vector kernels (saxpy, reduction) take `--count`. Matrix kernels (transpose, gemm) take `--rows` and `--cols`.
- saxpy gets `--alpha`.
- `sweep.py` stops discovering binaries. It has a fixed list of kernels and executables, and passes different arguments to each kind of kernel.
- GEMM's inner dimension is `--inner`. naive and v1 were generalized to any rows × cols × inner.
- Defaults are powers of two: transpose 8192 × 8192, GEMM 4096 × 4096 × 4096. Sweeps use powers of two only.

## 1. Benchmark class

**Shared options** (unchanged for every kernel): `-i/--iterations`, `-w/--warmups`, `-v/--verify`, `-t/--time-unit`, `--no-color`, `--json`.

**Removed:** `-c/--count` and `Benchmark::count()`.

**New constructor**, with a callback that adds kernel options before `argv` is parsed:

```cpp
Benchmark(int argc, char **argv, const std::string &hardware,
          const std::function<void(BenchmarkOptions &)> &addOptions);
```

Every kernel uses it, since every kernel now has at least one size option:

```cpp
uint32_t count = 1 << 27;
Benchmark bench(argc, argv, deviceName, [&](BenchmarkOptions &options) {
  options.add("-c,--count", count, "Number of elements");
});
```

`BenchmarkOptions::add` has overloads for `uint32_t`, `uint64_t` (both must be positive) and `float`.

**Changed:** the proposal passed `CLI::App &` to kernels and made CLI11 a public dependency. That crashed every CUDA binary before `main`: CUDA host code is compiled by GCC (through `nvcc`), the benchmark library by DPC++'s clang, and CLI11 is header-only, so the program ended up with two compilers' copies of CLI11's global objects. `BenchmarkOptions` keeps all CLI11 code inside the library, and CLI11 stays a private dependency.

- `--help` lists shared and kernel options together.

**JSON output:**
- removed: `"count"`
- added: `"params"`, every kernel option with its final value, e.g. `{"rows": 4096, "cols": 4096}` or `{"count": 134217728, "alpha": 2.0}`. `Benchmark` collects these from the parsed `CLI::App` itself, so kernels don't report them by hand.
- everything else stays as it is: times, GB/s, GFLOP/s.

`printInfo` (the header before a run) prints the kernel options under the shared ones.

## 2. Per kernel

Both CUDA and SYCL versions of each kernel get exactly the same options.

| Kernel | Options | Default | Notes |
|---|---|---|---|
| saxpy | `-c, --count` | 2²⁷ | the optimized variant still requires a multiple of 4 |
|  | `-a, --alpha` | 2.0 | verification stays exact (`alpha * 1 + 2` computed the same way on host and device) |
| reduction | `-c, --count` | 2²⁷ | |
| transpose | `--rows`, `--cols` | 8192 × 8192 | non-square works |
| gemm | `--rows`, `--cols`, `--inner` | 4096 each | `--inner` must stay below 342,392 so the verification stays exact (49 × inner < 2²⁴) |

**transpose, non-square:** the kernels already handle `rows ≠ cols`. The verification doesn't: it checks `output[row * cols + col]`, but the output is `cols × rows`. I'd fix the verification and test several non-square shapes.

**gemm, non-square:** naive and v1 now handle any shape: A is rows × inner, B is inner × cols, C is rows × cols. A test with a planted square-only bug (`cols` instead of `inner` as A's row width) passes square shapes and fails non-square ones, so non-square shapes belong in every check.

## 3. sweep.py

**From discovery to a fixed list.** The script names exactly the kernels, variants and languages it expects:

```python
KERNELS = {
    "saxpy":     {"shape": "vector"},
    "reduction": {"shape": "vector"},
    "transpose": {"shape": "matrix"},
    "gemm":      {"shape": "gemm"},
}
VARIANTS = ["naive", "optimized"]
LANGS = ["cuda", "sycl"]
```

The executable names follow from that: `<kernel>-<variant>-<lang>`.

- **A missing executable is an error**, listed by name before anything runs, instead of being silently skipped.
- `--kernel` takes exact names (`--kernel transpose,gemm`), no substring matching.
- Adding a kernel means adding one line to `KERNELS`.

**Arguments by shape:**

| Shape | Size means | Arguments per run | Default sizes |
|---|---|---|---|
| vector | element count | `--count S` | 2¹⁰ … 2²⁷ (as today) |
| matrix | side length | `--rows S --cols S` | 2⁵ … 2¹⁴ |
| gemm | side length | `--rows S --cols S --inner S` | 2⁵ … 2¹³ |

- `--sizes` stays, but its values mean whatever the selected shape means. It's rejected when the selected kernels would read it differently (element count vs side length); transpose and gemm together are fine, since both read it as side length.
- Iterations can be set per shape. GEMM needs fewer: at the default 200 iterations, naive GEMM at 4096 would take about 1 minute per point.

**Output JSON:** each run stores the `params` from the benchmark JSON instead of `requested_count`, plus an `x` value for plotting: element count for vectors, side length for matrices.

## 4. visualize_sweep.py

- x-axis from each run's `x` instead of `count`.
- Axis label by shape: "Element count" for vectors, "Side length (N × N)" for matrices.
- GEMM plotted in GFLOP/s by default; the others stay in GB/s.

## 5. Other places that mention `--count`

- README "Measurement" section: "set with `--count`" becomes "set with `--count` for vector kernels, `--rows`/`--cols` for matrix kernels".
- `src/transpose/README.md` explains N = 11585 as "the largest square that fits in 2²⁷ elements". That explanation goes away with `--rows`/`--cols`; the note becomes just the measured size (and changes if the default changes, question 3).
- `CLAUDE.md` and `BUILD.md` don't mention `--count`. BUILD.md's binary name pattern (`_` instead of `-`) is already wrong; I'd fix that while I'm there.
- `PAPER_PLAN.md`: tick the CLI item when it's done.

## Open questions (answered)

1. GEMM's third dimension: `--inner`.
2. Non-square GEMM kernels: generalized as part of this change.
3. Default sizes: powers of two (transpose 8192, GEMM 4096).
4. Sweep sizes: powers of two only, for now.
