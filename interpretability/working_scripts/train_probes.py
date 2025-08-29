#!/usr/bin/env python3
"""
Train simple probes on hidden states from log_rollouts.py.

Tasks:
  - target color (multiclass): predict active color from h
  - accepting (binary): predict info['accepting'] from h (skip if only one class)
  - hit_target (binary): predict success_step (target prop hit on this step)
  - per-color proposition presence (binary one-vs-rest): c ∈ info['propositions']

Group split by chain_id to avoid leakage.

Example:
  python interpretability/working_scripts/train_probes.py \
    --parquet interpretability/working_scripts/rollouts_stateful.parquet \
    --colors "green,blue,yellow,magenta" \
    --out_dir interpretability/working_scripts/probes --seed 0
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import array

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ───────────────────────── helpers ─────────────────────────

def is_vector_like(x):
    if x is None:
        return False
    if isinstance(x, (list, tuple, np.ndarray, array.array)):
        try:
            a = np.asarray(x)
            if a.ndim == 0:
                return False
            if a.ndim > 1:
                a = a.reshape(-1)
            return a.size > 0 and np.issubdtype(a.dtype, np.number)
        except Exception:
            return False
    return False

def normalize_vectors(series):
    kept_idx = []
    vecs = []
    for i, x in series.items():
        if is_vector_like(x):
            a = np.asarray(x, dtype=np.float32).reshape(-1)
            vecs.append(a)
            kept_idx.append(i)
    return vecs, np.array(kept_idx, dtype=int)

def to_X(df):
    vecs, kept = normalize_vectors(df["h"])
    if len(vecs) == 0:
        raise RuntimeError("No vector-like 'h' rows after normalization.")
    # enforce uniform length
    lengths = [v.shape[0] for v in vecs]
    u = np.unique(lengths)
    if len(u) != 1:
        common = int(pd.Series(lengths).mode().iloc[0])
        keep = [i for i, v in enumerate(vecs) if v.shape[0] == common]
        vecs = [vecs[i] for i in keep]
        kept = kept[keep]
    X = np.stack(vecs, axis=0)
    return X, kept

def train_test_groups(groups, test_frac=0.25, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx = np.arange(len(groups))
    tr_idx, te_idx = next(gss.split(idx, groups=groups))
    return tr_idx, te_idx

def fit_linear_probe(X_tr, y_tr, X_te, y_te, multiclass=False, balanced=True):
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr)
    Xte = scaler.transform(X_te)
    if multiclass:
        clf = LogisticRegression(
            max_iter=2000, n_jobs=1, multi_class="auto",
            class_weight=None  # class balance not needed for multiclass color here
        )
    else:
        clf = LogisticRegression(
            max_iter=2000, n_jobs=1, solver="lbfgs",
            class_weight=("balanced" if balanced else None)
        )
    clf.fit(Xtr, y_tr)
    y_pred = clf.predict(Xte)
    y_proba = None
    try:
        y_proba = clf.predict_proba(Xte)
    except Exception:
        pass
    return clf, scaler, y_pred, y_proba

def safe_binary_metrics(y_te, yhat, yproba):
    acc = accuracy_score(y_te, yhat)
    f1  = f1_score(y_te, yhat, average="binary")
    try:
        if yproba is not None and len(np.unique(y_te)) >= 2:
            auroc = roc_auc_score(y_te, yproba[:,1])
        else:
            auroc = float("nan")
    except Exception:
        auroc = float("nan")
    return acc, f1, auroc

# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--out_dir", type=str, default="interpretability/working_scripts/probes")
    args = ap.parse_args()

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    df0 = pd.read_parquet(args.parquet)
    vecs, kept_idx = normalize_vectors(df0["h"])
    df = df0.iloc[kept_idx].reset_index(drop=True)
    print(f"Loaded {len(df0)} rows; kept {len(df)} with vector-like h; chains={df['chain_id'].nunique()}")

    # quick label balance diagnostics
    acc_rate = float(df["accepting"].astype(int).mean()) if "accepting" in df else 0.0
    suc_rate = float(df["success_step"].astype(int).mean())
    print(f"Positive rates: accepting={acc_rate:.4f}  success_step={suc_rate:.4f}")
    for c in colors:
        pr = df["propositions"].apply(lambda s: (isinstance(s, str) and (c in s.split(";")))).astype(int).mean()
        print(f"  prop[{c}] rate={float(pr):.4f}")

    # features
    X, kept_again = to_X(df)
    hidden_size = X.shape[1]
    print(f"Hidden size inferred: {hidden_size}")

    # split by chain
    groups = df["chain_id"].values
    tr_idx, te_idx = train_test_groups(groups, test_frac=args.test_frac, seed=args.seed)
    X_tr, X_te = X[tr_idx], X[te_idx]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Probe 1: target color (multiclass) ----------
    color_to_id = {c:i for i,c in enumerate(colors)}
    y_color = df["color"].map(color_to_id).values
    y_tr, y_te = y_color[tr_idx], y_color[te_idx]

    clf_c, sc_c, yhat_c, yproba_c = fit_linear_probe(X_tr, y_tr, X_te, y_te, multiclass=True)
    acc_c = accuracy_score(y_te, yhat_c)
    f1_c  = f1_score(y_te, yhat_c, average="macro")
    print(f"\n[Probe] target color (multiclass): acc={acc_c:.3f}  macroF1={f1_c:.3f}")
    with open(out_dir / "probe_color.json", "w") as f:
        json.dump({"acc": acc_c, "macroF1": f1_c, "hidden_size": hidden_size}, f, indent=2)

    # ---------- Probe 2: accepting (binary) ----------
    if "accepting" in df:
        y_acc = df["accepting"].astype(int).values
        y_tr_acc, y_te_acc = y_acc[tr_idx], y_acc[te_idx]
        if len(np.unique(y_tr_acc)) >= 2:
            clf_a, sc_a, yhat_a, yproba_a = fit_linear_probe(X_tr, y_tr_acc, X_te, y_te_acc, multiclass=False, balanced=True)
            acc_a, f1_a, auroc_a = safe_binary_metrics(y_te_acc, yhat_a, yproba_a)
            print(f"[Probe] accepting (binary): acc={acc_a:.3f}  F1={f1_a:.3f}  AUROC={auroc_a:.3f}")
            with open(out_dir / "probe_accepting.json", "w") as f:
                json.dump({"acc": acc_a, "F1": f1_a, "AUROC": auroc_a, "hidden_size": hidden_size}, f, indent=2)
        else:
            print("[Probe] accepting (binary): skipped (train split has only one class).")
    else:
        print("[Probe] accepting: skipped (column missing)")

    # ---------- Probe 3: hit_target (binary, success_step) ----------
    y_hit = df["success_step"].astype(int).values
    y_tr_hit, y_te_hit = y_hit[tr_idx], y_hit[te_idx]
    if len(np.unique(y_tr_hit)) >= 2:
        clf_h, sc_h, yhat_h, yproba_h = fit_linear_probe(X_tr, y_tr_hit, X_te, y_te_hit, multiclass=False, balanced=True)
        acc_h, f1_h, auroc_h = safe_binary_metrics(y_te_hit, yhat_h, yproba_h)
        print(f"[Probe] hit_target (success_step): acc={acc_h:.3f}  F1={f1_h:.3f}  AUROC={auroc_h:.3f}")
        with open(out_dir / "probe_hit_target.json", "w") as f:
            json.dump({"acc": acc_h, "F1": f1_h, "AUROC": auroc_h, "hidden_size": hidden_size}, f, indent=2)
    else:
        print("[Probe] hit_target: skipped (train split has only one class).")

    # ---------- Probe 4: proposition presence per color (one-vs-rest) ----------
    prop_scores = {}
    for c in colors:
        y = df["propositions"].apply(lambda s: (isinstance(s, str) and (c in s.split(";")))).astype(int).values
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2:
            print(f"[Probe] prop:{c:>7s} skipped (train split has only one class).")
            continue
        clf_p, sc_p, yhat_p, yproba_p = fit_linear_probe(X_tr, y_tr, X_te, y_te, multiclass=False, balanced=True)
        acc, f1  = accuracy_score(y_te, yhat_p), f1_score(y_te, yhat_p, average="binary")
        try:
            auroc = roc_auc_score(y_te, yproba_p[:,1]) if yproba_p is not None and len(np.unique(y_te))>=2 else float("nan")
        except Exception:
            auroc = float("nan")
        prop_scores[c] = {"acc": acc, "F1": f1, "AUROC": auroc}
        print(f"[Probe] prop:{c:>7s}  acc={acc:.3f}  F1={f1:.3f}  AUROC={auroc:.3f}")

    with open(out_dir / "probe_props.json", "w") as f:
        json.dump(prop_scores, f, indent=2)

    # Save split info
    np.savez(out_dir / "splits_and_maps.npz",
             tr_idx=tr_idx, te_idx=te_idx,
             hidden_size=np.array([hidden_size]),
             color_to_id=np.array(list(color_to_id.items()), dtype=object))
    print(f"\nSaved probe metrics → {out_dir}")

if __name__ == "__main__":
    main()
