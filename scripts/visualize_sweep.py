#!/usr/bin/env python3
"""Visualize a sweep_results.json (see sweep.py): a bandwidth-vs-count plot
(one subplot per kernel, one line per variant/lang combination) and a
Markdown report with a results table per kernel.

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


def plot(results, metric, output, show):
    kernels = sorted(results)
    fig, axes = plt.subplots(len(kernels), 1, figsize=(8, 5 * len(kernels)), squeeze=False)

    for ax, kernel in zip(axes[:, 0], kernels):
        for variant, langs in sorted(results[kernel].items()):
            for lang, branch in sorted(langs.items()):
                runs = sorted(branch["runs"], key=lambda r: r["count"])
                x = [r["count"] for r in runs]
                y = [r[metric] for r in runs]
                ax.plot(x, y,
                        color=VARIANT_COLOR.get(variant, "tab:gray"),
                        linestyle=LANG_STYLE.get(lang, "-"),
                        marker=LANG_MARKER.get(lang, "."),
                        label=f"{variant}-{lang}")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Element count")
        ax.set_ylabel(metric)
        ax.set_title(kernel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote plot to {output}", file=sys.stderr)

    if show:
        plt.show()


def render_markdown(results, metric, plot_path, output):
    lines = ["# Sweep Results", "", f"![Sweep plot]({plot_path.name})", ""]

    for kernel in sorted(results):
        lines.append(f"## {kernel}")
        lines.append("")

        columns = sorted(f"{variant}-{lang}"
                          for variant, langs in results[kernel].items()
                          for lang in langs)
        hardware = {branch["hardware"]
                    for langs in results[kernel].values()
                    for branch in langs.values()}
        lines.append(f"Hardware: {', '.join(sorted(hardware))}")
        lines.append("")

        by_column = {f"{variant}-{lang}": {r["count"]: r[metric] for r in branch["runs"]}
                     for variant, langs in results[kernel].items()
                     for lang, branch in langs.items()}
        counts = sorted({count for column in by_column.values() for count in column})

        lines.append(f"| Count | {' | '.join(columns)} |")
        lines.append(f"|---|{'---|' * len(columns)}")
        for count in counts:
            row = [f"{by_column[col].get(count, float('nan')):.2f}" for col in columns]
            lines.append(f"| {count:,} | {' | '.join(row)} |")
        lines.append("")

    output.write_text("\n".join(lines))
    print(f"Wrote report to {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("sweep_results.json"))
    parser.add_argument("--plot-output", type=Path, default=Path("plot.png"))
    parser.add_argument("--md-output", type=Path, default=Path("report.md"))
    parser.add_argument("--metric", type=str, default="median_gbps",
                         help="Which per-run field to report "
                              "(e.g. median_gbps, mean_gbps, max_gbps)")
    parser.add_argument("--no-show", action="store_true",
                         help="Don't open an interactive plot window, just save the files")
    args = parser.parse_args()

    results = json.loads(args.input.read_text())
    if not results:
        print(f"No kernels found in {args.input}", file=sys.stderr)
        return 1

    render_markdown(results, args.metric, args.plot_output, args.md_output)
    plot(results, args.metric, args.plot_output, show=not args.no_show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
