#!/usr/bin/env python3
"""Sweep every benchmark binary in the build directory across a range of
problem sizes and collect the results into one JSON file, nested by
kernel/variant/lang.

Discovery is by naming convention, not a hardcoded list, so adding a new
kernel (reduction-naive-cuda, matrix_transpose-optimized-sycl, ...) needs no
change here -- it just needs to follow the same <kernel>-<variant>-<lang>
target naming CMakeLists.txt already uses, and accept -c/--count the same
way the vector_add binaries do.

Runs are strictly sequential -- two benchmark binaries sharing a GPU at the
same time corrupts both runs' timing.

Usage:
  ./scripts/sweep.py                              # sweep everything found
  ./scripts/sweep.py --kernel vector_add           # only that kernel
  ./scripts/sweep.py --dry-run                     # show the plan, run nothing
  ./scripts/sweep.py --sizes 1024,1048576,134217728
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BINARY_RE = re.compile(r"^(?P<kernel>.+)-(?P<variant>naive|optimized)-(?P<lang>cuda|sycl)$")

DEFAULT_SIZES = [1 << e for e in (10, 14, 16, 18, 20, 22, 24, 26, 27)]


def discover_binaries(build_dir: Path, kernel_filter: str | None):
    found = []
    for path in sorted(build_dir.iterdir()):
        if not path.is_file() or not path.stat().st_mode & 0o111:
            continue
        m = BINARY_RE.match(path.name)
        if not m:
            continue
        info = m.groupdict()
        if kernel_filter and kernel_filter not in info["kernel"]:
            continue
        found.append({"path": path, **info})
    return found


def run_one(binary: dict, size: int, iterations: int, warmups: int, verify: str):
    cmd = [
        str(binary["path"]),
        "-c", str(size),
        "-i", str(iterations),
        "-w", str(warmups),
        "-v", verify,
        "--json", "--no-color",
    ]
    binary_dir = binary["path"].parent
    json_path = binary_dir / f"{binary['path'].name}.json"

    proc = subprocess.run(cmd, cwd=binary_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip()

    if not json_path.exists():
        return None, f"expected JSON file not found: {json_path}"

    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as e:
        return None, f"invalid JSON in {json_path}: {e}"

    row = {"kernel": binary["kernel"], "variant": binary["variant"],
           "lang": binary["lang"], "requested_count": size}
    row.update(data)
    return row, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path,
                         default=Path(__file__).resolve().parent.parent / "build",
                         help="Directory containing the built benchmark binaries")
    parser.add_argument("--sizes", type=str, default=None,
                         help="Comma-separated element counts (default: powers of two, 2^10..2^27)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--verify", type=str, default="None", choices=["None", "Semi", "Full"],
                         help="Kept off by default -- verify iterations reallocate and copy the "
                              "full output buffer, which dominates timing and isn't representative "
                              "of steady-state kernel performance")
    parser.add_argument("--kernel", type=str, default=None,
                         help="Only sweep binaries whose kernel name contains this substring")
    parser.add_argument("--output", type=Path, default=Path("sweep_results.json"))
    parser.add_argument("--dry-run", action="store_true",
                         help="List discovered binaries and the planned runs, execute nothing")
    args = parser.parse_args()

    sizes = DEFAULT_SIZES if args.sizes is None else [int(s) for s in args.sizes.split(",")]

    binaries = discover_binaries(args.build_dir, args.kernel)
    if not binaries:
        print(f"No benchmark binaries found in {args.build_dir} "
              f"(expected names like vector_add-naive-cuda)", file=sys.stderr)
        return 1

    print(f"Discovered {len(binaries)} binaries in {args.build_dir}:", file=sys.stderr)
    for b in binaries:
        print(f"  {b['kernel']:<20} {b['variant']:<10} {b['lang']:<5} {b['path'].name}",
              file=sys.stderr)
    print(f"Sizes: {sizes}", file=sys.stderr)

    total = len(binaries) * len(sizes)
    if args.dry_run:
        print(f"Dry run: would execute {total} benchmark runs.", file=sys.stderr)
        return 0

    results = {}
    done = 0
    ok = 0
    for binary in binaries:
        for size in sizes:
            done += 1
            label = f"{binary['kernel']}/{binary['variant']}/{binary['lang']} @ {size}"
            print(f"[{done}/{total}] {label} ...", end=" ", file=sys.stderr, flush=True)
            start = time.monotonic()
            row, err = run_one(binary, size, args.iterations, args.warmups, args.verify)
            elapsed = time.monotonic() - start
            if row is None:
                print(f"SKIPPED ({elapsed:.1f}s): {err}", file=sys.stderr)
                continue
            print(f"{elapsed:.1f}s, median {row['median_gbps']} GB/s", file=sys.stderr)

            branch = (results.setdefault(row["kernel"], {})
                              .setdefault(row["variant"], {})
                              .setdefault(row["lang"], {"program": row["program"],
                                                         "hardware": row["hardware"],
                                                         "runs": []}))
            run = {k: v for k, v in row.items()
                   if k not in ("kernel", "variant", "lang", "program", "hardware")}
            branch["runs"].append(run)
            ok += 1

    with args.output.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"\nWrote {ok}/{total} runs to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
