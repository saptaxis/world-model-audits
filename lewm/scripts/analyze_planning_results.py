#!/usr/bin/env python3
"""Summarize JEPA planning results.

Consumes a `lunarlander_*_results.txt` written by eval.py and produces a
markdown summary: success rate, mean terminal distance, per-episode breakdown.

Usage:
    python lewm/scripts/analyze_planning_results.py \\
        --results-file /path/to/lunarlander_replay_results.txt \\
        --mode replay
"""

import argparse
import re
from pathlib import Path


def parse_results_file(path: Path) -> dict:
    """Parse the metrics block out of the results txt."""
    text = path.read_text()
    # Find the most recent ==== RESULTS ==== block
    parts = text.split("==== RESULTS ====")
    last = parts[-1] if parts else text
    metrics = {}
    for line in last.splitlines():
        m = re.match(r"\s*metrics:\s*(.*)", line)
        if m:
            # Try to eval as dict; fall back to raw string
            try:
                metrics.update(eval(m.group(1)))
            except Exception:
                metrics["raw_metrics_line"] = m.group(1)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-file", required=True, type=Path)
    p.add_argument("--mode", choices=["replay", "synthetic"], required=True)
    p.add_argument("--output-md", type=Path, default=None,
                   help="Where to write the markdown summary")
    args = p.parse_args()

    metrics = parse_results_file(args.results_file)

    md_lines = [
        f"# Planning Results — {args.mode} mode",
        "",
        f"**Source:** `{args.results_file}`",
        "",
        "## Metrics",
        "",
    ]
    for key, val in sorted(metrics.items()):
        md_lines.append(f"- **{key}:** {val}")

    md = "\n".join(md_lines)
    print(md)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md)
        print(f"\nWrote {args.output_md}")


if __name__ == "__main__":
    main()
