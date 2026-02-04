from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="split_data.py")
    ap.add_argument("--input", default="data/spans_labeled.jsonl")
    ap.add_argument("--train-out", default="data/extractor_train.jsonl")
    ap.add_argument("--dev-out", default="data/extractor_dev.jsonl")
    ap.add_argument("--dev-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    inp = Path(args.input)
    rows = _read_jsonl(inp)
    if not rows:
        raise SystemExit(f"No rows found in {inp}")

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    dev_n = max(1, int(round(len(rows) * args.dev_ratio)))
    dev = rows[:dev_n]
    train = rows[dev_n:]
    if not train:
        raise SystemExit("Dev split consumed all data; decrease --dev-ratio")

    _write_jsonl(Path(args.train_out), train)
    _write_jsonl(Path(args.dev_out), dev)
    print(
        json.dumps(
            {
                "input": str(inp),
                "train_out": str(args.train_out),
                "dev_out": str(args.dev_out),
                "n_total": len(rows),
                "n_train": len(train),
                "n_dev": len(dev),
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

