"""Command line: `streamwatch detect` scores a CSV or stdin stream, `streamwatch bench` runs the benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Callable, Iterable, TextIO

from streamwatch.benchmark import make_streams, run
from streamwatch.detectors import EWMADetector, RollingZScoreDetector
from streamwatch.iforest import StreamingIsolationForest


def build(name: str, threshold: float | None, window: int) -> Callable[[], object]:
    if name == "zscore":
        return lambda: RollingZScoreDetector(window=window, threshold=threshold or 4.0)
    if name == "ewma":
        return lambda: EWMADetector(alpha=2 / (window + 1), threshold=threshold or 4.0)
    if name == "iforest":
        return lambda: StreamingIsolationForest(threshold=threshold or 0.67)
    raise ValueError(f"unknown detector {name!r}")


def read_values(f: TextIO, column: str | None) -> Iterable[float]:
    if column is None:
        for line in f:
            line = line.strip()
            if line:
                yield float(line.split(",")[-1])
        return
    for row in csv.DictReader(f):
        yield float(row[column])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="streamwatch")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="score a stream of numbers and print anomalies")
    d.add_argument("file", nargs="?", help="CSV or one number per line (default: stdin)")
    d.add_argument("--column", help="CSV column with the values (file must have a header)")
    d.add_argument("--detector", choices=["zscore", "ewma", "iforest"], default="zscore")
    d.add_argument("--threshold", type=float)
    d.add_argument("--window", type=int, default=50)
    d.add_argument("--all", action="store_true", help="print every point, not only anomalies")

    b = sub.add_parser("bench", help="run the synthetic benchmark")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    if args.cmd == "bench":
        rows = run({n: build(n, None, 50) for n in ("zscore", "ewma", "iforest")},
                   streams=make_streams(args.n, args.seed))
        print(f"{'detector':<10}{'stream':<18}{'precision':>10}{'recall':>8}{'f1':>7}")
        for r in rows:
            print(f"{r['detector']:<10}{r['stream']:<18}{r['precision']:>10.2f}{r['recall']:>8.2f}{r['f1']:>7.2f}")
        return 0

    det = build(args.detector, args.threshold, args.window)()
    f = open(args.file, newline="") if args.file else sys.stdin
    try:
        for x in read_values(f, args.column):
            r = det.update(x)
            if r.is_anomaly or args.all:
                print(json.dumps({"index": r.index, "value": r.value,
                                  "score": round(r.score, 4), "anomaly": r.is_anomaly}))
                sys.stdout.flush()
    finally:
        if args.file:
            f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
