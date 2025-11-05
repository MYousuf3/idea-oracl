#!/usr/bin/env python3
"""Compute simple metrics comparing `position` (ranker output) to original `rank`.

Usage:
  python tools/compute_rank_metrics.py /path/to/ranked_papers.json

Outputs summary statistics and top movers.
"""
import json
import math
import statistics
import sys
from pathlib import Path


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(records):
    rows = []
    for r in records:
        # Accept either `rank` (original) and `position` (new) or missing values
        if "rank" not in r or "position" not in r:
            continue
        try:
            orig = int(r["rank"]) if r["rank"] is not None else None
            pos = int(r["position"]) if r["position"] is not None else None
        except Exception:
            continue
        if orig is None or pos is None:
            continue
        diff = pos - orig  # positive => moved down (higher numeric position)
        rows.append({"id": r.get("id", r.get("arxiv", "")), "title": r.get("title", ""), "rank": orig, "position": pos, "diff": diff, "absdiff": abs(diff)})

    if not rows:
        return None

    diffs = [r["diff"] for r in rows]
    absdiffs = [r["absdiff"] for r in rows]

    metrics = {
        "n": len(rows),
        "mean_signed_diff": statistics.mean(diffs),
        "mean_abs_diff": statistics.mean(absdiffs),
        "median_abs_diff": statistics.median(absdiffs),
        # Mean squared difference (MSE) of signed differences
        "mean_squared_diff": statistics.mean([d * d for d in diffs]),
        "max_abs_diff": max(absdiffs),
        "num_unchanged": sum(1 for d in diffs if d == 0),
    }

    # Top movers (by absolute change)
    top_movers = sorted(rows, key=lambda x: (-x["absdiff"], x["diff"]))[:10]

    return metrics, top_movers, rows


def print_report(path: Path, metrics_res):
    if metrics_res is None:
        print("No valid records with both 'rank' and 'position' found.")
        return

    metrics, top_movers, _ = metrics_res
    print(f"Input: {path}")
    print(f"Total papers analyzed: {metrics['n']}")
    print(f"Mean signed difference (position - rank): {metrics['mean_signed_diff']:.3f}")
    print(f"Mean absolute difference: {metrics['mean_abs_diff']:.3f}")
    print(f"Mean squared difference: {metrics['mean_squared_diff']:.3f}")
    print(f"Median absolute difference: {metrics['median_abs_diff']}")
    print(f"Max absolute difference: {metrics['max_abs_diff']}")
    print(f"Number unchanged: {metrics['num_unchanged']}")
    print("\nTop movers (by absolute difference):")
    for m in top_movers:
        print(f"  id={m['id']} | title={m['title'][:70]:70} | rank={m['rank']:3d} -> position={m['position']:3d} | diff={m['diff']:3d}")


def main(argv):
    if len(argv) < 2:
        print("Usage: compute_rank_metrics.py /path/to/ranked_papers.json")
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    records = load(path)
    metrics_res = compute_metrics(records)
    print_report(path, metrics_res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
