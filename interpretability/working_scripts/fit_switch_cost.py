#!/usr/bin/env python3
import argparse, csv, math
from pathlib import Path

def load_rows(path, env_id, exp, colors, stateful_flag):
    rows = []
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("env_id")==env_id and r.get("exp")==exp and r.get("colors")==colors and r.get("stateful")==str(int(stateful_flag)):
                rows.append({
                    "depth": int(r["depth"]),
                    "pred": float(r["predicted_mean"]),
                    "emp": float(r["empirical_mean"]),
                    "N": int(r["N"]),
                })
    rows.sort(key=lambda x: x["depth"])
    return rows

def fit_lambda(rows):
    # minimize sum (pred * lam**d - emp)^2 over lam in [0,1]
    best_lam, best_err = 1.0, float("inf")
    for steps in (2000,):
        for i in range(steps+1):
            lam = i/steps
            err = 0.0
            for r in rows:
                depth = r["depth"]
                p = r["pred"] * (lam ** depth)
                e = r["emp"]
                diff = p - e
                err += diff*diff
            if err < best_err:
                best_err, best_lam = err, lam
    return best_lam, best_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interpretability/working_scripts/world_model_vs_depth_stateful.csv")
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--colors", default="green,blue,yellow,magenta")
    ap.add_argument("--stateful", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv, args.env_id, args.exp, args.colors, args.stateful)
    if not rows:
        print("No matching rows found. Did you pass the right --csv / flags?")
        return

    lam, _ = fit_lambda(rows)
    print(f"Best switch-cost λ (per link) = {lam:.3f}\n")
    print("depth | predicted | adjusted | empirical | MAE(adjusted)")
    print("----- | --------- | -------- | --------- | -------------")
    mae = 0.0
    for r in rows:
        adj = r["pred"] * (lam ** r["depth"])
        mae += abs(adj - r["emp"])
        print(f"{r['depth']:>5d} | {r['pred']:.3f}     | {adj:.3f}    | {r['emp']:.3f}     | {abs(adj-r['emp']):.3f}")
    print(f"\nMean absolute error (adjusted): {mae/len(rows):.3f}")

if __name__ == "__main__":
    main()
