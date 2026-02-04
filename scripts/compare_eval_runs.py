"""
Compare Stage A metrics across multiple eval_results*.jsonl files.

Usage:
  python scripts/compare_eval_runs.py artifacts/eval_runA.jsonl artifacts/eval_runB.jsonl ...
  python scripts/compare_eval_runs.py --names A B C artifacts/eval_runA.jsonl artifacts/eval_runB.jsonl artifacts/eval_runC.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compare_eval_runs.py")
    ap.add_argument("files", nargs="+", help="Paths to eval_results*.jsonl (first line = _summary).")
    ap.add_argument("--names", nargs="*", help="Short names for each file (default: filename stem).")
    ap.add_argument("--stage", default="a", choices=["a", "b", "e2e"], help="Which stage summary to compare.")
    args = ap.parse_args(argv)

    paths = [Path(f) for f in args.files]
    names = list(args.names) if args.names else [p.stem for p in paths]
    if len(names) != len(paths):
        names = [p.stem for p in paths]

    rows: list[dict] = []
    for path, name in zip(paths, names, strict=True):
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        if not first:
            print(f"Skip (empty): {path}")
            continue
        data = json.loads(first)
        summary = data.get("_summary") or data
        rows.append({"name": name, "path": str(path), "summary": summary})

    if not rows:
        print("No valid eval files.")
        return 1

    # Stage A comparison table
    if args.stage == "a":
        stage_key = "stage_a_span_exact"
        headers = ["Run", "Precision", "Recall", "F1", "TP", "FP", "FN", "N"]
        col_widths = [
            max(len(r["name"]) for r in rows) + 1,
            10, 10, 8, 6, 6, 6, 4,
        ]

        def fmt_row(cells: list[str]) -> str:
            return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths, strict=True))

        print(fmt_row(headers))
        print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))

        for r in rows:
            s = (r["summary"] or {}).get(stage_key) or {}
            cells = [
                r["name"],
                f"{s.get('precision', 0):.4f}",
                f"{s.get('recall', 0):.4f}",
                f"{s.get('f1', 0):.4f}",
                str(s.get("tp", "")),
                str(s.get("fp", "")),
                str(s.get("fn", "")),
                str(r["summary"].get("n", "")),
            ]
            print(fmt_row(cells))

        # Best F1
        best = max(rows, key=lambda x: ((x["summary"] or {}).get(stage_key) or {}).get("f1", 0))
        best_name = best["name"]
        best_f1 = ((best["summary"] or {}).get(stage_key) or {}).get("f1")
        if best_f1 is not None:
            print()
            print(f"Best F1: {best_name} ({best_f1:.4f})")
    else:
        # Generic: print summary keys
        for r in rows:
            print(r["name"], ":", json.dumps(r["summary"], ensure_ascii=False)[:200], "...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
