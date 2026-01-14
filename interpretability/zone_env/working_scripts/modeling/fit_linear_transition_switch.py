#!/usr/bin/env python3
"""
Fit a one-step linear transition *at goal switches* only.

Builds true (t -> t+1) pairs where row_t has switch_flag==1 and row_{t+1}
is the immediate next step in the SAME chain.

Model:  h_{t+1} ≈ A h_t + B a_t + C [goal_old, goal_new, obs_features] + c

Flags:
  --use_actions          include a_t
  --use_goal             include one-hot of old & new goal (colors)
  --use_obs              include obs_features (if present in parquet)
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def one_hot(vals, classes):
    ix = {c:i for i,c in enumerate(classes)}
    out = np.zeros((len(vals), len(classes)), dtype=float)
    for r, v in enumerate(vals):
        j = ix.get(v, None)
        if j is not None: out[r, j] = 1.0
    return out

def build_switch_pairs(df, colors, use_actions, use_goal, use_obs):
    # sort & index for quick neighbor lookup within chain
    df = df.sort_values(["chain_id", "step_idx"]).reset_index(drop=True)
    df["next_key"] = list(zip(df["chain_id"], df["step_idx"] + 1))
    key_to_row = {(r.chain_id, r.step_idx): i for i, r in df.iterrows()}

    X_parts, Y = [], []
    h_list, a_list, old_goal, new_goal, obs_list = [], [], [], [], []

    # We want rows where THIS row is the first step after a goal change
    mask = df.get("switch_flag", 0) == 1
    switch_rows = df[mask]

    for i, r in switch_rows.iterrows():
        # t row = r; t+1 row must be same chain & next step
        j = key_to_row.get((r.chain_id, r.step_idx + 1))
        if j is None: 
            continue
        r_next = df.iloc[j]

        # Features at time t
        h_t = np.asarray(r["h"], dtype=float)
        a_t = np.asarray(r["a"], dtype=float) if use_actions else None

        # Goals: old goal is the color from previous step in same chain,
        # new goal is r["color"] (the target after switch)
        # To get old goal, look at t-1 if same chain exists:
        j_prev = key_to_row.get((r.chain_id, r.step_idx - 1))
        old_col = None
        if j_prev is not None:
            old_col = str(df.iloc[j_prev]["color"])
        new_col = str(r["color"])

        # Optional obs features (must exist in parquet as 'obs_features')
        obs_t = None
        if use_obs and "obs_features" in r and isinstance(r["obs_features"], (list, tuple)):
            obs_t = np.asarray(r["obs_features"], dtype=float)

        # Target h_{t+1}
        h_tp1 = np.asarray(r_next["h"], dtype=float)

        # Skip if shapes are off
        if h_t.ndim != 1 or h_tp1.ndim != 1:
            continue

        h_list.append(h_t)
        Y.append(h_tp1)

        if use_actions: a_list.append(a_t)
        if use_goal:
            old_goal.append(f"FG {old_col}" if old_col is not None else None)
            new_goal.append(f"FG {new_col}")
        if use_obs and obs_t is not None:
            obs_list.append(obs_t)

    if len(Y) == 0:
        raise RuntimeError("No valid (switch, next-step) pairs found.")

    H = np.stack(h_list, 0)
    Y = np.stack(Y, 0)
    parts = [H]
    scalers = {"H": StandardScaler().fit(H)}
    Hn = scalers["H"].transform(H)
    X = Hn

    if use_actions:
        A = np.stack(a_list, 0)
        scalers["A"] = StandardScaler().fit(A)
        An = scalers["A"].transform(A)
        X = np.concatenate([X, An], axis=1)

    if use_goal:
        # one-hot for old & new (same class set)
        classes = [f"FG {c.strip()}" for c in colors]
        O = one_hot(old_goal, classes)
        N = one_hot(new_goal, classes)
        X = np.concatenate([X, O, N], axis=1)
        scalers["GOAL_CLASSES"] = classes  # just to store

    if use_obs and len(obs_list) == len(H):
        F = np.stack(obs_list, 0)
        scalers["F"] = StandardScaler().fit(F)
        Fn = scalers["F"].transform(F)
        X = np.concatenate([X, Fn], axis=1)

    return X, Y, scalers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--use_actions", action="store_true")
    ap.add_argument("--use_goal", action="store_true")
    ap.add_argument("--use_obs", action="store_true")
    ap.add_argument("--alphas", type=str, default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    df = pd.read_parquet(args.parquet)
    X, Y, scalers = build_switch_pairs(df, colors, args.use_actions, args.use_goal, args.use_obs)

    # split
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = max(1, int(args.test_frac * n))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    # ridge over candidate alphas
    alphas = [float(x) for x in args.alphas.split(",")]
    best = None
    for a in alphas:
        mdl = Ridge(alpha=a, fit_intercept=True, random_state=args.seed)
        mdl.fit(Xtr, Ytr)
        r2 = r2_score(Yte, mdl.predict(Xte))
        if best is None or r2 > best[0]:
            best = (r2, a, mdl)
    r2, alpha, mdl = best

    print(f"Pairs (switch→next-step): {n} | test R^2={r2:.3f}  (alpha={alpha})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "transition_linear_switch.npz",
             coef=mdl.coef_, intercept=mdl.intercept_)
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"r2_test": float(r2), "alpha": float(alpha),
                   "n_pairs": int(n), "use_actions": args.use_actions,
                   "use_goal": args.use_goal, "use_obs": args.use_obs}, f, indent=2)

if __name__ == "__main__":
    main()
