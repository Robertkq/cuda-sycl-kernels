#!/usr/bin/env python3
"""Run a fixed set of benchmark executables across a range of problem sizes
and collect the results into one JSON file, nested by kernel/variant/lang.

The kernels, their executables and their arguments are listed below in
KERNELS and SHAPES. A listed executable that is missing from the build
directory is an error. Adding a kernel means adding it to KERNELS.

Each shape defines what a "size" means and which options it turns into:
  vector  element count       --count S
  matrix  side length         --rows S --cols S
  gemm    side length         --rows S --cols S --inner S

Runs are strictly sequential -- two benchmark binaries sharing a GPU at the
same time corrupts both runs' timing.

Usage:
  ./scripts/sweep.py                                # every kernel, default sizes
  ./scripts/sweep.py --kernel transpose,gemm        # only these kernels
  ./scripts/sweep.py --kernel saxpy --sizes 1024,1048576
  ./scripts/sweep.py --dry-run                      # show the plan, run nothing
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def powers_of_two(first, last):
    return [1 << e for e in range(first, last + 1)]


SHAPES = {
    "vector": {
        "size_means": "element count",
        "args": lambda s: ["--count", str(s)],
        "sizes": [1 << e for e in (10, 14, 16, 18, 20, 22, 24, 26, 27)],
        "x_label": "Element count",
        "metric": "median_gbps",
        "iterations": 200,
    },
    "matrix": {
        "size_means": "side length",
        "args": lambda s: ["--rows", str(s), "--cols", str(s)],
        "sizes": powers_of_two(5, 14),
        "x_label": "Side length (N x N)",
        "metric": "median_gbps",
        "iterations": 200,
    },
    "gemm": {
        "size_means": "side length",
        "args": lambda s: ["--rows", str(s), "--cols", str(s), "--inner", str(s)],
        "sizes": powers_of_two(5, 13),
        "x_label": "Side length (N x N x N)",
        "metric": "median_gflops",
        # naive GEMM takes over a second per run at 8192
        "iterations": 20,
    },
}

KERNELS = {
    "saxpy": {
        "shape": "vector",
        "executables": ["saxpy-naive-cuda", "saxpy-optimized-cuda",
                        "saxpy-naive-sycl", "saxpy-optimized-sycl"],
    },
    "reduction": {
        "shape": "vector",
        "executables": ["reduction-naive-cuda", "reduction-optimized-cuda",
                        "reduction-naive-sycl", "reduction-optimized-sycl"],
    },
    "transpose": {
        "shape": "matrix",
        "executables": ["transpose-naive-cuda", "transpose-optimized-cuda",
                        "transpose-naive-sycl", "transpose-optimized-sycl"],
    },
    "gemm": {
        "shape": "gemm",
        # no CUDA GEMM yet
        "executables": ["gemm-naive-sycl", "gemm-optimized-sycl"],
    },
}


def variant_and_lang(executable):
    # names are <kernel>-<variant>-<lang>
    _, variant, lang = executable.rsplit("-", 2)
    return variant, lang


def run_one(path: Path, shape: dict, size: int, iterations: int, warmups: int, verify: str):
    cmd = [str(path), *shape["args"](size),
           "-i", str(iterations), "-w", str(warmups), "-v", verify,
           "--json", "--no-color"]
    json_path = path.parent / f"{path.name}.json"
    json_path.unlink(missing_ok=True)
    proc = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or f"exit code {proc.returncode}"
    if not json_path.exists():
        return None, f"expected JSON file not found: {json_path}"
    try:
        return json.loads(json_path.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON in {json_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "build",
                        help="Directory containing the built benchmark binaries")
    parser.add_argument("--kernel", type=str, default=None,
                        help=f"Comma-separated kernel names (default: all of {', '.join(KERNELS)})")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated sizes, meaning depends on the kernels' shape; "
                             "only allowed when they mean the same for all selected kernels")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Timed runs per point (default: per shape)")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--verify", type=str, default="None", choices=["None", "Semi", "Full"],
                        help="Kept off by default -- verify iterations reallocate and copy the "
                             "full output buffer, which dominates timing and isn't representative "
                             "of steady-state kernel performance")
    parser.add_argument("--output", type=Path, default=Path("sweep_results.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="List the planned runs, execute nothing")
    args = parser.parse_args()

    names = list(KERNELS) if args.kernel is None else args.kernel.split(",")
    unknown = [n for n in names if n not in KERNELS]
    if unknown:
        print(f"Unknown kernel(s): {', '.join(unknown)} (known: {', '.join(KERNELS)})",
              file=sys.stderr)
        return 1

    meanings = {SHAPES[KERNELS[n]["shape"]]["size_means"] for n in names}
    if args.sizes is not None and len(meanings) > 1:
        print(f"--sizes is ambiguous for these kernels: it would mean "
              f"{' and '.join(sorted(meanings))}", file=sys.stderr)
        return 1
    custom_sizes = None if args.sizes is None else [int(s) for s in args.sizes.split(",")]

    missing = [exe for n in names for exe in KERNELS[n]["executables"]
               if not (args.build_dir / exe).is_file()]
    if missing:
        print(f"Missing executables in {args.build_dir}:", file=sys.stderr)
        for exe in missing:
            print(f"  {exe}", file=sys.stderr)
        return 1

    plan = []
    for n in names:
        shape = SHAPES[KERNELS[n]["shape"]]
        sizes = custom_sizes or shape["sizes"]
        iterations = args.iterations or shape["iterations"]
        for exe in KERNELS[n]["executables"]:
            for size in sizes:
                plan.append((n, exe, size, iterations))

    for n in names:
        shape_name = KERNELS[n]["shape"]
        sizes = custom_sizes or SHAPES[shape_name]["sizes"]
        print(f"{n:<10} {shape_name:<7} {len(KERNELS[n]['executables'])} executables, "
              f"sizes {sizes}", file=sys.stderr)
    if args.dry_run:
        print(f"Dry run: would execute {len(plan)} benchmark runs.", file=sys.stderr)
        return 0

    results = {}
    ok = 0
    for done, (n, exe, size, iterations) in enumerate(plan, start=1):
        shape_name = KERNELS[n]["shape"]
        shape = SHAPES[shape_name]
        print(f"[{done}/{len(plan)}] {exe} @ {size} ...", end=" ", file=sys.stderr, flush=True)
        start = time.monotonic()
        data, err = run_one(args.build_dir / exe, shape, size, iterations,
                            args.warmups, args.verify)
        elapsed = time.monotonic() - start
        if data is None:
            print(f"FAILED ({elapsed:.1f}s): {err}", file=sys.stderr)
            continue
        print(f"{elapsed:.1f}s, {shape['metric']} {data[shape['metric']]:.1f}", file=sys.stderr)

        variant, lang = variant_and_lang(exe)
        kernel = results.setdefault(n, {"shape": shape_name, "x_label": shape["x_label"],
                                        "metric": shape["metric"], "variants": {}})
        branch = (kernel["variants"].setdefault(variant, {})
                  .setdefault(lang, {"program": data["program"],
                                     "hardware": data["hardware"], "runs": []}))
        run = {k: v for k, v in data.items() if k not in ("program", "hardware")}
        run["x"] = size
        branch["runs"].append(run)
        ok += 1

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"\nWrote {ok}/{len(plan)} runs to {args.output}", file=sys.stderr)
    return 0 if ok == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
