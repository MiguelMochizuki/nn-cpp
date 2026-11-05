#!/usr/bin/env python3
"""Compare two bench CSVs. ratio > 1 means 'after' is faster. Usage: compare_csv.py before.csv after.csv"""
import csv
import sys

KEY = ("config", "label", "threads", "benchmark", "size")


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load(path):
    try:
        with open(path, newline="") as f:
            return {tuple(r.get(k, "") for k in KEY): r for r in csv.DictReader(f)}
    except OSError as e:
        print(f"compare_csv: {e}", file=sys.stderr)
        sys.exit(2)


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    for k in sorted(set(a) | set(b)):
        name = "/".join(x for x in k if x)
        if k not in b:
            print(f"{name}: removed")
        elif k not in a:
            print(f"{name}: added")
        else:
            x, y = a[k], b[k]
            if k[3] == "matmul":
                bv, av = num(x.get("gflops")), num(y.get("gflops"))
                ratio = av / bv if av is not None and bv else None  # after/before gflops
            else:
                bv, av = num(x.get("total_ms")), num(y.get("total_ms"))
                ratio = bv / av if bv is not None and av else None  # before/after ms
            if ratio is None:
                print(f"{name}: n/a (missing values)")
                continue
            noise = max(num(x.get("spread")) or 0.0, num(y.get("spread")) or 0.0)
            # spread is relative (IQR/median), same scale as ratio-1
            flag = "  within-noise" if abs(ratio - 1) < noise or ratio == 1 else ""
            print(f"{name}: {ratio:.2f}x{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
