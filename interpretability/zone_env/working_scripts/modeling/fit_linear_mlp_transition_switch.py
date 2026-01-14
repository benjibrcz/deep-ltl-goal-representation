#!/usr/bin/env python3
"""
Fit a one-step transition *at goal switches* only.

We build true (t -> t+1) pairs where row_t has switch_flag==1 and row_{t+1}
is the immediate next step in the SAME chain.

Models
------
Linear (default):
    h_{t+1} ≈ A h_t + B a_t + C [goal_old, goal_new, obs_features] + c
Non-linear (MLP):
    h_{t+1} ≈ f_MLP([h_t, a_t?, goal_onehots?, obs_features?])

Inputs
------
A rollouts parquet that contains at least the following columns for each row:
    chain_id : int
    step_idx : int (or sortable step index)
    h        : 1-D vector-like GRU hidden
    a        : 1-D vector-like action (only if --use_actions)
    color    : goal color string for this row
    switch_flag : 1 if this row is the *first* step after a goal change
Optionally:
    obs_features : 1-D vector-like features (only if --use_obs)

Outputs
-------
Linear:
  <out_dir>/transition_linear_switch.npz  (coef, intercept)
  <out_dir>/scalers.pkl                   (per-block scalers)
  <out_dir>/metrics.json                  (R^2, alpha, n_pairs, flags)

MLP:
  <out_dir>/transition_mlp_switch.pkl     (pickled MLPRegressor)
  <out_dir>/scalers.pkl                   (same scalers as linear path)
  <out_dir>/metrics.json                  (R^2, mlp params, n_pairs, flags)

Examples
--------
# Linear (as before)
python fit_linear_transition_switch.py \
  --parquet rollouts.parquet --out_dir out/lin \
  --use_actions --use_goal --use_obs \
  --alphas "1e-3,1e-2,1e-1,1,10"

# MLP (non-linear)
python fit_linear_transition_switch.py \
  --parquet rollouts.parquet --out_dir out/mlp \
  --use_actions --use_goal --use_obs \
  --probe_type mlp --mlp_hidden "128,64" --mlp_alpha 1e-3 --mlp_max_iter 400
"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor


def one_hot(vals, classes):
    ix = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(vals), len(classes)), dtype=float)
    for r, v in enumerate(vals):
        j = ix.get(v, None)
        if j is not None:
            out[r, j] = 1.0
    return out


def build_switch_pairs(df, colors, use_actions, use_goal, use_obs):
    """
    Build design matrix X and targets Y for rows where THIS row has switch_flag==1
    and the target is the immediate next step in the same chain.

    Returns
    -------
    X : np.ndarray (N, D)    standardized by per-block scalers
    Y : np.ndarray (N, H)
    scalers : dict           fitted scalers per block + GOAL_CLASSES
    """
    # sort & index for quick neighbor lookup within chain
    df = df.sort_values(["chain_id", "step_idx"]).reset_index(drop=True)
    key_to_row = {(int(r.chain_id), int(r.step_idx)): i for i, r in df.iterrows()}

    h_list, a_list, old_goal, new_goal, obs_list, Y = [], [], [], [], [], []

    # We want rows where THIS row is the first step after a goal change
    mask = df.get("switch_flag", 0) == 1
    switch_rows = df[mask]

    for _, r in switch_rows.iterrows():
        # t row = r; t+1 row must be same chain & next step
        j = key_to_row.get((int(r.chain_id), int(r.step_idx) + 1))
        if j is None:
            continue
        r_next = df.iloc[j]

        # Features at time t
        h_t = np.asarray(r["h"], dtype=float)
        if h_t.ndim != 1:
            continue

        a_t = None
        if use_actions:
            if "a" in r and r["a"] is not None:
                a_t = np.asarray(r["a"], dtype=float).ravel()
            else:
                # If actions requested but missing, default to zeros later after we learn the dim.
                a_t = None

        # Goals (old = color at t-1, new = r["color"])
        # Old goal: look at t-1 if same chain exists
        j_prev = key_to_row.get((int(r.chain_id), int(r.step_idx) - 1))
        old_col = str(df.iloc[j_prev]["color"]) if j_prev is not None else None
        new_col = str(r["color"])

        # Optional obs features (must exist and be vector-like)
        obs_t = None
        if use_obs and "obs_features" in r:
            try:
                obs_t = np.asarray(r["obs_features"], dtype=float).ravel()
            except Exception:
                obs_t = None

        # Target h_{t+1}
        h_tp1 = np.asarray(r_next["h"], dtype=float)
        if h_tp1.ndim != 1:
            continue

        h_list.append(h_t)
        Y.append(h_tp1)
        if use_actions:
            a_list.append(a_t)
        if use_goal:
            old_goal.append(f"FG {old_col}" if old_col is not None else None)
            new_goal.append(f"FG {new_col}")
        if use_obs and obs_t is not None:
            obs_list.append(obs_t)

    if len(Y) == 0:
        raise RuntimeError("No valid (switch, next-step) pairs found.")

    # --- assemble blocks + scalers ---
    H = np.stack(h_list, 0)
    Y = np.stack(Y, 0)
    parts = []
    scalers = {}

    # h block (always)
    scalers["H"] = StandardScaler().fit(H)
    Hn = scalers["H"].transform(H)
    parts.append(Hn)
    meta_blocks = [{"name": "h", "dim": Hn.shape[1]}]

    # action block (optional)
    if use_actions:
        # infer action dim from first non-None
        dim_a = None
        for a in a_list:
            if a is not None:
                dim_a = a.size
                break
        if dim_a is not None:
            A = np.stack([np.zeros(dim_a, dtype=float) if a is None else a for a in a_list], 0)
            scalers["A"] = StandardScaler().fit(A)
            An = scalers["A"].transform(A)
            parts.append(An)
            meta_blocks.append({"name": "a", "dim": dim_a})
        else:
            # no actionable actions found
            pass

    # goal one-hots (old & new)
    if use_goal:
        classes = [f"FG {c.strip()}" for c in colors]
        O = one_hot(old_goal, classes)
        N = one_hot(new_goal, classes)
        parts.append(O)
        parts.append(N)
        scalers["GOAL_CLASSES"] = classes
        meta_blocks.append({"name": "goal_old", "dim": O.shape[1]})
        meta_blocks.append({"name": "goal_new", "dim": N.shape[1]})

    # obs features block (optional; only if we had one per row)
    if use_obs and len(obs_list) == len(h_list):
        F = np.stack(obs_list, 0)
        scalers["F"] = StandardScaler().fit(F)
        Fn = scalers["F"].transform(F)
        parts.append(Fn)
        meta_blocks.append({"name": "obs", "dim": Fn.shape[1]})

    X = np.concatenate(parts, axis=1) if len(parts) else Hn
    return X.astype(np.float32), Y.astype(np.float32), scalers, meta_blocks


def parse_hidden(s: str):
    """Parse '128,64' -> (128, 64); '', 'None' -> None."""
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() == "none":
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


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

    # NEW: non-linear probe options
    ap.add_argument("--probe_type", type=str, default="linear",
                    choices=["linear", "mlp"],
                    help="linear (Ridge sweep) or mlp (MLPRegressor).")
    ap.add_argument("--mlp_hidden", type=str, default="128,64",
                    help="Comma list of hidden sizes for MLP, e.g. '128,64'.")
    ap.add_argument("--mlp_alpha", type=float, default=3e-4,
                    help="L2 regularization (alpha) for MLPRegressor.")
    ap.add_argument("--mlp_max_iter", type=int, default=300,
                    help="Max iterations for MLPRegressor.")
    ap.add_argument("--mlp_early_stopping", action="store_true",
                    help="Enable early stopping in MLPRegressor.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    df = pd.read_parquet(args.parquet)
    X, Y, scalers, meta_blocks = build_switch_pairs(
        df, colors,
        use_actions=args.use_actions,
        use_goal=args.use_goal,
        use_obs=args.use_obs
    )

    # split (simple random split across pairs; switch rows are already filtered)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_te = max(1, int(args.test_frac * n))
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "n_pairs": int(n),
        "test_frac": float(args.test_frac),
        "seed": int(args.seed),
        "use_actions": bool(args.use_actions),
        "use_goal": bool(args.use_goal),
        "use_obs": bool(args.use_obs),
        "feature_blocks": meta_blocks,
    }

    if args.probe_type == "linear":
        # Ridge over candidate alphas (multioutput)
        alphas = [float(x) for x in args.alphas.split(",")]
        best = None
        for a in alphas:
            mdl = Ridge(alpha=a, fit_intercept=True, random_state=args.seed)
            mdl.fit(Xtr, Ytr)
            r2 = r2_score(Yte, mdl.predict(Xte))
            if best is None or r2 > best[0]:
                best = (r2, a, mdl)
        r2, alpha, mdl = best

        print(f"[Linear] Pairs (switch→next-step): {n} | test R^2={r2:.3f}  (alpha={alpha})")

        # Save linear artifacts (backward compatible names)
        np.savez(out_dir / "transition_linear_switch.npz",
                 coef=mdl.coef_, intercept=mdl.intercept_)
        with open(out_dir / "scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        metrics.update({
            "probe_type": "linear",
            "r2_test": float(r2),
            "alpha": float(alpha),
            "alphas_sweep": alphas,
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    else:
        # MLP Regressor (non-linear)
        hidden = parse_hidden(args.mlp_hidden)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden or (),
            activation="relu",
            solver="adam",
            alpha=float(args.mlp_alpha),
            max_iter=int(args.mlp_max_iter),
            early_stopping=bool(args.mlp_early_stopping),
            n_iter_no_change=10,
            validation_fraction=0.1,
            batch_size="auto",
            random_state=args.seed,
            verbose=False,
        )
        mlp.fit(Xtr, Ytr)
        r2 = r2_score(Yte, mlp.predict(Xte))
        print(f"[MLP]    Pairs (switch→next-step): {n} | test R^2={r2:.3f}  "
              f"(hidden={hidden}, alpha={args.mlp_alpha}, max_iter={args.mlp_max_iter}, "
              f"early_stopping={args.mlp_early_stopping})")

        # Save MLP + scalers
        with open(out_dir / "transition_mlp_switch.pkl", "wb") as f:
            pickle.dump(mlp, f)
        with open(out_dir / "scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        metrics.update({
            "probe_type": "mlp",
            "r2_test": float(r2),
            "mlp_hidden": hidden if hidden is not None else [],
            "mlp_alpha": float(args.mlp_alpha),
            "mlp_max_iter": int(args.mlp_max_iter),
            "mlp_early_stopping": bool(args.mlp_early_stopping),
        })
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
