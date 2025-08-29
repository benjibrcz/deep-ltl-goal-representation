#!/usr/bin/env python3
import argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = successes / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n)))) / denom
    return (center, max(0.0, center - margin), min(1.0, center + margin))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--by_pair", action="store_true", help="Also print per (s,t) pair.")
    ap.add_argument("--out_png", type=str, default="", help="If set, save a bar chart to this path.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # overall per-condition
    print("Overall (per condition):")
    rows = []
    for cond, sub in df.groupby("condition"):
        n = len(sub)
        s = int(sub["success"].sum())
        center, lo, hi = wilson(s, n)
        print(f"  {cond:>16s} | n={n:4d}  succ={s:4d}  rate={center:.3f}  95% CI=({lo:.3f},{hi:.3f})")
        rows.append((cond, center, lo, hi, n))
    rows.sort(key=lambda x: x[0])

    if args.by_pair:
        print("\nBy (s,t) pair:")
        for (scol, tcol), sub in df.groupby(["s","t"]):
            n = len(sub)
            s = int(sub["success"].sum())
            center, lo, hi = wilson(s, n)
            print(f"  {scol:>7s} → {tcol:<7s} | n={n:3d} succ={s:3d} rate={center:.3f} 95% CI=({lo:.3f},{hi:.3f})")

    if args.out_png:
        conds = [r[0] for r in rows]
        centers = [r[1] for r in rows]
        lows = [r[1]-r[2] for r in rows]
        highs = [r[3]-r[1] for r in rows]

        x = np.arange(len(conds))
        fig = plt.figure(figsize=(10, 5))
        plt.bar(x, centers, yerr=[lows, highs], capsize=4)
        plt.xticks(x, conds, rotation=20, ha="right")
        plt.ylabel("Success rate")
        plt.title("Goal-switch intervention: success by condition (Wilson 95% CI)")
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=200)
        print(f"\nSaved plot → {args.out_png}")

if __name__ == "__main__":
    main()
