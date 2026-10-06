#!/usr/bin/env python3
"""Visualize a sweep_results.json (see sweep.py): one subplot per kernel,
one line per variant/lang combination, against the kernel's size (element
count for vector kernels, side length for matrix kernels), plus a Markdown
report with a results table per kernel.

Each kernel is plotted with its own default metric (GB/s for memory-bound
kernels, GFLOP/s for GEMM) unless --metric overrides it for all.

Usage:
  ./scripts/visualize_sweep.py                          # reads sweep_results.json
  ./scripts/visualize_sweep.py --input my_sweep.json --plot-output p.png --md-output r.md
  ./scripts/visualize_sweep.py --metric mean_gbps
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

VARIANT_COLOR = {"naive": "tab:red", "optimized": "tab:blue"}
LANG_STYLE = {"cuda": "-", "sycl": "--"}
LANG_MARKER = {"cuda": "o", "sycl": "s"}


def plot(results, metric_override, output, show):
    kernels = sorted(results)
    fig, axes = plt.subplots(len(kernels), 1, figsize=(8, 5 * len(kernels)), squeeze=False)

    for ax, kernel in zip(axes[:, 0], kernels):
        info = results[kernel]
        metric = metric_override or info["metric"]
        for variant, langs in sorted(info["variants"].items()):
            for lang, branch in sorted(langs.items()):
                runs = sorted(branch["runs"], key=lambda r: r["x"])
                x = [r["x"] for r in runs]
                y = [r[metric] for r in runs]
                ax.plot(x, y,
                        color=VARIANT_COLOR.get(variant, "tab:gray"),
                        linestyle=LANG_STYLE.get(lang, "-"),
                        marker=LANG_MARKER.get(lang, "."),
                        label=f"{variant}-{lang}")

        ax.set_xscale("log", base=2)
        ax.set_xlabel(info["x_label"])
        ax.set_ylabel(metric)
        ax.set_title(kernel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote plot to {output}", file=sys.stderr)

    if show:
        plt.show()


def render_markdown(results, metric_override, plot_path, output):
    lines = ["# Sweep Results", "", f"![Sweep plot]({plot_path.name})", ""]

    for kernel in sorted(results):
        info = results[kernel]
        metric = metric_override or info["metric"]
        variants = info["variants"]
        lines.append(f"## {kernel}")
        lines.append("")

        columns = sorted(f"{variant}-{lang}"
                          for variant, langs in variants.items()
                          for lang in langs)
        hardware = {branch["hardware"]
                    for langs in variants.values()
                    for branch in langs.values()}
        lines.append(f"Hardware: {', '.join(sorted(hardware))}")
        lines.append("")
        lines.append(f"Metric: {metric}")
        lines.append("")

        by_column = {f"{variant}-{lang}": {r["x"]: r[metric] for r in branch["runs"]}
                     for variant, langs in variants.items()
                     for lang, branch in langs.items()}
        sizes = sorted({size for column in by_column.values() for size in column})

        lines.append(f"| {info['x_label']} | {' | '.join(columns)} |")
        lines.append(f"|---|{'---|' * len(columns)}")
        for size in sizes:
            row = [f"{by_column[col].get(size, float('nan')):.2f}" for col in columns]
            lines.append(f"| {size:,} | {' | '.join(row)} |")
        lines.append("")

    output.write_text("\n".join(lines))
    print(f"Wrote report to {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("sweep_results.json"))
    parser.add_argument("--plot-output", type=Path, default=Path("plot.png"))
    parser.add_argument("--md-output", type=Path, default=Path("report.md"))
    parser.add_argument("--metric", type=str, default=None,
                         help="Which per-run field to report for every kernel "
                              "(e.g. median_gbps, median_gflops); default: per kernel")
    parser.add_argument("--no-show", action="store_true",
                         help="Don't open an interactive plot window, just save the files")
    args = parser.parse_args()

    results = json.loads(args.input.read_text())
    if not results:
        print(f"No kernels found in {args.input}", file=sys.stderr)
        return 1
    old = [k for k, v in results.items() if "variants" not in v]
    if old:
        print(f"{args.input} uses the old sweep format (kernels: {', '.join(old)}); "
              f"re-run scripts/sweep.py", file=sys.stderr)
        return 1

    render_markdown(results, args.metric, args.plot_output, args.md_output)
    plot(results, args.metric, args.plot_output, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
