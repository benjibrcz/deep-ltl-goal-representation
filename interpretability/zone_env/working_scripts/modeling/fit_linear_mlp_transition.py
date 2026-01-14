#!/usr/bin/env python3
"""
Fit a linear controlled transition model for the GRU hidden state, and (optionally) a nonlinear MLP:

    Linear:  h_{t+1} ≈ A h_t + B u_t + c
    MLP   :  h_{t+1} ≈ f_MLP([h_t, u_t])

where u_t concatenates (optionally) the action a_t and/or a one-hot goal indicator.
Evaluates 1-step test R^2 and free k-step rollouts (MSE and cosine) for both models.

Outputs
-------
Linear (always):
- <out_dir>/transition_linear.npz  (A, B, c, feature meta)
- <out_dir>/scalers.pkl            (sklearn scalers for X and y, pickle)
- <out_dir>/metrics.json           (R^2, rollout MSE/cosine per k, and settings)

MLP (only if --mlp):
- <out_dir>/transition_mlp.pkl     (sklearn MLPRegressor, pickle)
- <out_dir>/metrics_mlp.json       (R^2, rollout MSE/cosine per k, and settings)

Usage example
-------------
python interpretability/working_scripts/fit_linear_transition.py \
  --parquet interpretability/working_scripts/rollouts_stateful.parquet \
  --out_dir interpretability/working_scripts/transition_linear \
  --use_actions --use_goal --colors "green,blue,yellow,magenta" \
  --k_eval "1,2,5,10" --alphas "1e-4,1e-3,1e-2,1e-1,1,10" --seed 0 \
  --mlp --mlp_hidden "256,128" --mlp_alpha 1e-3 --mlp_max_iter 400
"""
import argparse, json, os, sys, pickle, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

# ---------------- small utils ----------------
def is_vector_like(x) -> bool:
    try:
        arr = np.asarray(x)
        return arr.ndim == 1 and arr.size > 0
    except Exception:
        return False

def parse_goal_color(s: str) -> Optional[str]:
    # pull the *last* alphabetic token; handles "FG green", "F blue" etc.
    if not isinstance(s, str):
        return None
    m = re.findall(r"[A-Za-z]+", s)
    return m[-1].lower() if m else None

def group_split_indices(groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into train/test by group labels (e.g., chain_id)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        n = len(groups)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = max(1, int(round(n * test_frac)))
        return idx[n_test:], idx[:n_test]
    rng.shuffle(uniq)
    n_test_g = max(1, int(round(len(uniq) * test_frac)))
    test_g = set(uniq[:n_test_g])
    te_mask = np.array([g in test_g for g in groups])
    tr_idx = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]
    return tr_idx, te_idx

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ------------- dataset assembly --------------
def build_pairs(df: pd.DataFrame,
                group_col: str = "chain_id",
                step_col_candidates: List[str] = ["t", "step"]) -> pd.DataFrame:
    """
    Create aligned (h_t, a_t, goal_t) -> h_{t+1} rows.
    Keeps rows where both h_t and h_{t+1} exist inside the same chain.
    Adds columns: h_t, a_t, goal_t, h_tp1
    """
    if group_col not in df.columns:
        # assume single chain; synthesize chain_id
        df = df.copy()
        df[group_col] = 0

    step_col = None
    for c in step_col_candidates:
        if c in df.columns:
            step_col = c
            break

    # If no explicit step col, create one by order within chain.
    if step_col is None:
        df = df.copy()
        df["_tmp_idx"] = df.groupby(group_col).cumcount()
        step_col = "_tmp_idx"

    # Keep only rows with vector-like h
    df = df[df["h"].apply(is_vector_like)].copy()
    if df.empty:
        raise RuntimeError("No rows with vector-like 'h' were found.")

    # Normalize goal text column name
    goal_col = None
    for gc in ["goal_text", "goal", "goal_str", "goal_now"]:
        if gc in df.columns:
            goal_col = gc
            break

    # Align within each chain
    rows = []
    for g, gdf in df.groupby(group_col, sort=False):
        gdf = gdf.sort_values(step_col)
        hs = gdf["h"].apply(np.asarray).tolist()
        # optional fields
        actions = gdf["a"].apply(np.asarray).tolist() if "a" in gdf.columns else [None]*len(gdf)
        goals   = gdf[goal_col].tolist() if goal_col is not None else [None]*len(gdf)

        for i in range(len(gdf)-1):
            h_t   = hs[i]
            h_tp1 = hs[i+1]
            a_t   = actions[i]
            goal_t= goals[i]
            rows.append({
                "chain_id": g,
                "h_t": h_t,
                "a_t": a_t,
                "goal_t": goal_t,
                "h_tp1": h_tp1,
            })
    return pd.DataFrame(rows)

def make_goal_onehot(goal_series: pd.Series, palette: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    colors = []
    for s in goal_series:
        colors.append(parse_goal_color(s) if isinstance(s, str) else None)
    uniq = sorted({c for c in colors if c is not None})
    if palette:
        # keep only those in palette, in that order; include any extras at end
        ordered = [c for c in palette if c in uniq] + [c for c in uniq if c not in palette]
    else:
        ordered = uniq
    idx_map = {c: i for i, c in enumerate(ordered)}
    onehots = []
    for c in colors:
        x = np.zeros(len(ordered), dtype=np.float32)
        if c in idx_map:
            x[idx_map[c]] = 1.0
        onehots.append(x)
    return np.stack(onehots, axis=0) if ordered else np.zeros((len(colors), 0), dtype=np.float32), ordered

# ------------- model fitting / eval ----------
def fit_ridge_multioutput(Xtr, Ytr, Xte, Yte, alphas: List[float]) -> Tuple[Ridge, float]:
    best_r2 = -np.inf
    best = None
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=True, solver="auto", random_state=0)
        reg.fit(Xtr, Ytr)
        r2 = reg.score(Xte, Yte)
        if r2 > best_r2:
            best_r2, best = r2, reg
    return best, best_r2

def rollout_free_linear(A: np.ndarray, B: np.ndarray, c: np.ndarray,
                        h0: np.ndarray, U_seq: np.ndarray, k: int) -> np.ndarray:
    """
    Linear free rollout for k steps: returns predicted h_{t+1..t+k}
    A: (H,H), B: (H,U) or (H,0), c: (H,)
    h0: (H,), U_seq: (k, U)
    """
    H = h0.shape[0]
    preds = np.zeros((k, H), dtype=np.float32)
    h = h0.copy()
    for i in range(k):
        u = U_seq[i] if U_seq.shape[1] > 0 else np.zeros(0, dtype=np.float32)
        h = (A @ h) + (B @ u if B.size else 0) + c
        preds[i] = h
    return preds

def rollout_free_mlp(mlp, Xsc: StandardScaler, Ysc: StandardScaler,
                     h0: np.ndarray, U_seq: np.ndarray, k: int) -> np.ndarray:
    """
    MLP free rollout: iteratively apply f([h_t, u_t]) in scaled space, inverse-transform outputs.
    """
    H = h0.shape[0]
    preds = np.zeros((k, H), dtype=np.float32)
    h = h0.copy()
    for i in range(k):
        u = U_seq[i] if U_seq.shape[1] > 0 else np.zeros(0, dtype=np.float32)
        x = np.concatenate([h, u], axis=0, dtype=np.float32)[None, :]
        x_s = Xsc.transform(x)
        y_s = mlp.predict(x_s)
        y   = Ysc.inverse_transform(y_s)
        h   = y[0].astype(np.float32)
        preds[i] = h
    return preds

def cosine(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

# ------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_actions", action="store_true", help="Include action a_t in control u_t.")
    ap.add_argument("--use_goal", action="store_true", help="Include goal one-hot in control u_t.")
    ap.add_argument("--colors", type=str, default="", help="Comma list of goal colors to order one-hot.")
    ap.add_argument("--alphas", type=str, default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--k_eval", type=str, default="1,2,5,10")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    # Nonlinear probe options
    ap.add_argument("--mlp", action="store_true", help="Also fit an MLPRegressor as a nonlinear transition.")
    ap.add_argument("--mlp_hidden", type=str, default="128,64")
    ap.add_argument("--mlp_alpha", type=float, default=1e-3)
    ap.add_argument("--mlp_max_iter", type=int, default=300)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alphas = [float(s) for s in args.alphas.split(",")]
    k_eval = [int(s) for s in args.k_eval.split(",")]
    out_dir = Path(args.out_dir); ensure_dir(out_dir)

    # ---------- load parquet ----------
    df = pd.read_parquet(args.parquet)
    n0 = len(df)
    # Build pairs (requires vector-like h and next-step within chain)
    pairs = build_pairs(df)
    if pairs.empty:
        raise RuntimeError("No (h_t, h_{t+1}) pairs could be built from parquet.")
    print(f"Loaded {n0} rows; built {len(pairs)} (h_t, a_t, goal_t -> h_t+1) pairs.")

    # ---------- assemble features ----------
    H = len(pairs.iloc[0]["h_t"])
    X_parts = [np.stack(pairs["h_t"].apply(np.asarray).values, axis=0).astype(np.float32)]
    meta = {"feature_blocks": [{"name": "h_t", "dim": H}]}
    U_dim = 0

    if args.use_actions:
        if pairs["a_t"].isna().all():
            print("⚠️  No action column found; --use_actions ignored.")
        else:
            A_mat = []
            dim_a = None
            for a in pairs["a_t"].values:
                if a is None:
                    A_mat.append(None)
                else:
                    arr = np.asarray(a, dtype=np.float32).ravel()
                    dim_a = dim_a or arr.size
                    A_mat.append(arr)
            # replace Nones with zeros of correct size
            if dim_a is None:
                print("⚠️  Actions present but empty; ignoring actions.")
            else:
                A_arr = np.stack([np.zeros(dim_a, dtype=np.float32) if v is None else v for v in A_mat], axis=0)
                X_parts.append(A_arr)
                U_dim += dim_a
                meta["feature_blocks"].append({"name": "a_t", "dim": dim_a})

    goal_onehot = None
    goal_order = []
    if args.use_goal:
        goal_series = pairs["goal_t"] if "goal_t" in pairs.columns else pd.Series([None]*len(pairs))
        palette = [s.strip().lower() for s in args.colors.split(",") if s.strip()] or None
        goal_onehot, goal_order = make_goal_onehot(goal_series, palette)
        if goal_onehot.shape[1] == 0:
            print("⚠️  No goal text detected; --use_goal ignored.")
        else:
            X_parts.append(goal_onehot.astype(np.float32))
            U_dim += goal_onehot.shape[1]
            meta["feature_blocks"].append({"name": "goal_onehot", "dim": goal_onehot.shape[1], "order": goal_order})

    X = np.concatenate(X_parts, axis=1).astype(np.float32)
    Y = np.stack(pairs["h_tp1"].apply(np.asarray).values, axis=0).astype(np.float32)

    # group split by chain
    groups = pairs["chain_id"].to_numpy()
    tr_idx, te_idx = group_split_indices(groups, args.test_frac, args.seed)
    Xtr, Xte = X[tr_idx], X[te_idx]
    Ytr, Yte = Y[tr_idx], Y[te_idx]

    # standardize X, Y (Y centered for stability; intercept still learned)
    Xsc = StandardScaler().fit(Xtr)
    Ysc = StandardScaler(with_mean=True, with_std=True).fit(Ytr)
    Xtr_s, Xte_s = Xsc.transform(Xtr), Xsc.transform(Xte)
    Ytr_s, Yte_s = Ysc.transform(Ytr), Ysc.transform(Yte)

    # ===== Linear fit (always) =====
    reg, r2_1step = fit_ridge_multioutput(Xtr_s, Ytr_s, Xte_s, Yte_s, alphas)
    print(f"[Linear] One-step held-out R^2: {r2_1step:.3f}")

    # unpack coefficients into A,B,c on *original* (unscaled) space
    # Model in scaled space: Ys = W Xs + b
    W = reg.coef_.astype(np.float64)             # (H, D)
    b = reg.intercept_.astype(np.float64)        # (H,)
    # Undo scaling: Y = Ymean + Ystd * (W * (X - Xmean)/Xstd + b)
    Xstd = Xsc.scale_.astype(np.float64)
    Xmean= Xsc.mean_.astype(np.float64)
    Ystd = Ysc.scale_.astype(np.float64)
    Ymean= Ysc.mean_.astype(np.float64)

    WD = (Ystd[:, None] * W) / Xstd[None, :]
    c_vec = Ymean + (Ystd * b) - (WD @ Xmean)

    # partition WD into [A | B] according to feature_blocks
    Hdim = H
    A = WD[:, :Hdim]
    B = WD[:, Hdim:] if WD.shape[1] > Hdim else np.zeros((Hdim, 0), dtype=np.float64)
    c = c_vec

    # Build per-row control vector U_t (for rollouts)
    if B.shape[1] > 0:
        U_all = X[:, Hdim:].astype(np.float32)
    else:
        U_all = np.zeros((len(X), 0), dtype=np.float32)

    # attach indices back to pairs for grouping (for rollouts)
    pairs = pairs.reset_index(drop=True)
    pairs["_row"] = np.arange(len(pairs))
    te_rows = set(te_idx.tolist())

    # Build sequences (test chains only)
    seqs = []
    for g, gdf in pairs.groupby("chain_id"):
        rows = gdf.sort_index()["_row"].tolist()
        seqs.append(rows)

    def eval_free_rollouts(make_rollout_fn):
        roll_metrics = defaultdict(dict)
        for k in k_eval:
            mses, coss = [], []
            for rows in seqs:
                rows_k = [r for r in rows if r in te_rows]
                if len(rows_k) < (k+1):
                    continue
                for i in range(len(rows_k) - k):
                    r0 = rows_k[i]
                    h0 = X[r0, :Hdim]
                    Y_true = Y[r0+1 : r0+1+k]
                    if Y_true.shape[0] < k:
                        break
                    U_seq = U_all[r0+1 : r0+1+k] if U_all.shape[1] > 0 else np.zeros((k,0), dtype=np.float32)
                    Y_hat = make_rollout_fn(h0, U_seq, k)
                    mses.append(float(np.mean((Y_hat - Y_true)**2)))
                    coss.append(float(np.mean([cosine(Y_hat[j], Y_true[j]) for j in range(k)])))
            roll_metrics[k]["mse"] = float(np.mean(mses)) if mses else None
            roll_metrics[k]["cosine"] = float(np.mean(coss)) if coss else None
        return roll_metrics

    # Evaluate linear rollouts
    roll_metrics_linear = eval_free_rollouts(lambda h0, U_seq, k: rollout_free_linear(A, B, c, h0, U_seq, k))

    # Save linear artifacts
    np.savez_compressed(out_dir / "transition_linear.npz",
                        A=A, B=B, c=c,
                        H=H, U_dim=B.shape[1],
                        meta=json.dumps(meta))
    with open(out_dir / "scalers.pkl", "wb") as f:
        pickle.dump({"Xsc": Xsc, "Ysc": Ysc}, f)
    metrics = {
        "model": "linear_ridge",
        "r2_one_step": float(r2_1step),
        "k_eval": k_eval,
        "rollout": roll_metrics_linear,
        "use_actions": bool(args.use_actions),
        "use_goal": bool(args.use_goal),
        "goal_order": goal_order if args.use_goal else [],
        "alphas": alphas,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "n_pairs": int(len(pairs)),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n[Linear] Saved:")
    print(f"  A,B,c   → {out_dir/'transition_linear.npz'}")
    print(f"  scalers → {out_dir/'scalers.pkl'}")
    print(f"  metrics → {out_dir/'metrics.json'}")

    # ===== Optional MLP fit =====
    if args.mlp:
        hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
        mlp = MLPRegressor(hidden_layer_sizes=hidden,
                           alpha=args.mlp_alpha,
                           max_iter=args.mlp_max_iter,
                           random_state=args.seed)
        mlp.fit(Xtr_s, Ytr_s)
        r2_mlp = float(mlp.score(Xte_s, Yte_s))
        print(f"[MLP]    One-step held-out R^2: {r2_mlp:.3f}")

        # Free rollouts with iterative application of MLP
        roll_metrics_mlp = eval_free_rollouts(lambda h0, U_seq, k: rollout_free_mlp(mlp, Xsc, Ysc, h0, U_seq, k))

        # Save MLP artifacts
        with open(out_dir / "transition_mlp.pkl", "wb") as f:
            pickle.dump({"mlp": mlp, "Xsc": Xsc, "Ysc": Ysc,
                         "feature_meta": meta}, f)

        metrics_mlp = {
            "model": "mlp",
            "r2_one_step": r2_mlp,
            "k_eval": k_eval,
            "rollout": roll_metrics_mlp,
            "use_actions": bool(args.use_actions),
            "use_goal": bool(args.use_goal),
            "goal_order": goal_order if args.use_goal else [],
            "mlp_hidden": hidden,
            "mlp_alpha": float(args.mlp_alpha),
            "mlp_max_iter": int(args.mlp_max_iter),
            "test_frac": args.test_frac,
            "seed": args.seed,
            "n_pairs": int(len(pairs)),
        }
        with open(out_dir / "metrics_mlp.json", "w") as f:
            json.dump(metrics_mlp, f, indent=2)
        print("\n[MLP] Saved:")
        print(f"  model   → {out_dir/'transition_mlp.pkl'}")
        print(f"  metrics → {out_dir/'metrics_mlp.json'}")

    # Console summary
    print("\nSummary:")
    print(f"  [Linear] One-step R^2 (test): {r2_1step:.3f}")
    for k in k_eval:
        m = metrics["rollout"][k]
        print(f"  [Linear] Free rollout k={k:<2d}:  MSE={m['mse']!s:<12}  mean cos={m['cosine']!s}")
    if args.mlp:
        print(f"  [MLP]    One-step R^2 (test): {metrics_mlp['r2_one_step']:.3f}")
        for k in k_eval:
            m = metrics_mlp["rollout"][k]
            print(f"  [MLP]    Free rollout k={k:<2d}:  MSE={m['mse']!s:<12}  mean cos={m['cosine']!s}")

if __name__ == "__main__":
    main()
