#!/usr/bin/env python3
from __future__ import annotations
import argparse, math
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def stable_sort_centers(centers_xy: np.ndarray) -> np.ndarray:
    # deterministic ID assignment: sort by x then y
    idx = np.lexsort((centers_xy[:, 1], centers_xy[:, 0]))
    return centers_xy[idx].astype(float)


def compute_centers_from_success(df: pd.DataFrame, random_state: int = 0
                                 ) -> Dict[Tuple[int, str], np.ndarray]:
    """
    For each (seed,color), cluster SUCCESS hits into 2 centers with KMeans.
    Returns {(seed,color): array shape (2,2)}. Skips pairs with <2 success rows
    or degenerate clusters.
    """
    out: Dict[Tuple[int, str], np.ndarray] = {}
    ok = df[(df["success_step"] == True) & df["pos_x"].notna() & df["pos_y"].notna()]
    if ok.empty:
        return out
    for (seed, color), g in ok.groupby(["world_seed", "color"]):
        pts = g[["pos_x", "pos_y"]].to_numpy(dtype=float)
        if len(pts) < 2:
            continue
        km = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        labels = km.fit_predict(pts)
        # require both clusters present
        if len(set(labels.tolist())) < 2:
            continue
        out[(int(seed), str(color))] = stable_sort_centers(km.cluster_centers_)
    return out


def label_link(seg: pd.DataFrame, centers: np.ndarray,
               tail_frac: float = 0.2, tail_min: int = 10) -> int:
    """
    Label a (chain,seed,link) segment by which center the agent ends up near.
    Use the mean of the last max(tail_min, tail_frac*len) steps.
    """
    seg = seg.sort_values("step_idx")
    n = len(seg)
    tail_n = max(tail_min, int(math.ceil(n * tail_frac)))
    tail_n = min(tail_n, n)
    p = seg.tail(tail_n)[["pos_x", "pos_y"]].astype(float).mean().to_numpy()
    d = np.linalg.norm(centers - p[None, :], axis=1)
    return int(np.argmin(d))  # 0 or 1


def mean_first_k(vecs: List[List[float]], k: int, k_start: int = 0) -> Optional[np.ndarray]:
    if not vecs:
        return None
    s = vecs[k_start:k_start + k]
    if len(s) == 0:
        return None
    chosen = []
    d = None
    for v in s:
        if v is None:
            return None
        a = np.asarray(v, dtype=float).ravel()
        if d is None:
            d = a.shape[0]
        elif a.shape[0] != d:
            return None
        chosen.append(a)
    return np.mean(np.stack(chosen, axis=0), axis=0)


def seed_split_train_test(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          test_frac: float, seed: int):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    # if either split collapses to one class, fall back to another seed a few times
    tries = 0
    while (len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2) and tries < 10:
        tries += 1
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed + tries)
        tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    return tr_idx, te_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--use_h", action="store_true")
    ap.add_argument("--use_actor_in", action="store_true")
    ap.add_argument("--k_list", type=str, default="1,3,5,10,20,50")
    ap.add_argument("--k_start", type=int, default=0, help="start index for the first-k window")
    ap.add_argument("--min_steps_per_link", type=int, default=20)
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--tail_min", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_by_seed", action="store_true", help="seed-level split (recommended)")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--use_actor_penult", action="store_true")

    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    # required cols
    need = {"world_seed", "chain_id", "link_idx", "step_idx", "color", "pos_x", "pos_y", "success_step"}
    miss = sorted(list(need - set(df.columns)))
    if miss:
        raise SystemExit(f"Parquet missing columns: {miss}")

    # normalize types
    df = df.copy()
    for c in ["pos_x", "pos_y", "step_idx", "world_seed", "chain_id", "link_idx"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["color"] = df["color"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["pos_x", "pos_y", "world_seed"]).copy()
    df["world_seed"] = df["world_seed"].astype(int)

    # compute centers from SUCCESS HITS ONLY (simple & precise)
    centers = compute_centers_from_success(df, random_state=args.seed)
    print(f"[centers] learned from success hits: {len(centers)} (seed,color) pairs")

    # build per-link labels using end-of-link position vs 2 centers
    records = []
    groups = df.groupby(["world_seed", "chain_id", "link_idx"])
    labeled, skipped = 0, 0
    for (seed, chain, link), seg in groups:
        if len(seg) < args.min_steps_per_link:
            continue
        color = str(seg["color"].mode().iloc[0])
        key = (int(seed), color)
        if key not in centers:
            skipped += 1
            continue
        y = label_link(seg, centers[key], tail_frac=args.tail_frac, tail_min=args.tail_min)

        hs = seg["h"].tolist() if args.use_h and "h" in seg.columns else None
        actor_ins = seg["obs_features"].tolist() if args.use_actor_in and "obs_features" in seg.columns else None
        actor_penult = seg["actor_penult"].tolist() if args.use_actor_penult and "actor_penult" in seg.columns else None

        records.append({
            "seed": int(seed),
            "chain": int(chain),
            "link": int(link),
            "color": color,
            "y": int(y),
            "hs": hs,
            "actor_ins": actor_ins,
            "actor_penult": actor_penult,
            "length": int(len(seg)),
        })
        labeled += 1

    print(f"[links] labeled={labeled}  skipped_no_centers_or_short={skipped}")

    if len(records) == 0:
        print("No labeled links; exiting.")
        return

    ks = [int(x) for x in args.k_list.split(",")]
    rng = np.random.default_rng(args.seed)

    for k in ks:
        # H features
        if args.use_h:
            Xh, Yh, Gh = [], [], []
            for r in records:
                xh = mean_first_k(r["hs"], k, args.k_start)
                if xh is None:
                    continue
                Xh.append(xh); Yh.append(r["y"]); Gh.append(r["seed"])
            if len(Xh) > 0 and len(set(Yh)) >= 2:
                Xh = np.stack(Xh); Yh = np.asarray(Yh); Gh = np.asarray(Gh)
                if args.split_by_seed:
                    tr_idx, te_idx = seed_split_train_test(Xh, Yh, Gh, args.test_frac, args.seed)
                else:
                    tr_idx, te_idx = train_test_split(np.arange(len(Yh)), test_size=args.test_frac,
                                                      random_state=args.seed, stratify=Yh)
                clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xh[tr_idx], Yh[tr_idx])
                yhat = clf.predict(Xh[te_idx])
                acc = accuracy_score(Yh[te_idx], yhat)
                try:
                    auc = roc_auc_score(Yh[te_idx], clf.predict_proba(Xh[te_idx])[:, 1])
                except Exception:
                    auc = None
                print(f"  [h@k={k}] n={len(te_idx)} acc={acc:.3f} auc={None if auc is None else round(auc,3)}")
            else:
                print(f"  [h@k={k}] n=0")

        # actor_in features
        if args.use_actor_in:
            Xa, Ya, Ga = [], [], []
            for r in records:
                xa = mean_first_k(r["actor_ins"], k, args.k_start)
                if xa is None:
                    continue
                Xa.append(xa); Ya.append(r["y"]); Ga.append(r["seed"])
            if len(Xa) > 0 and len(set(Ya)) >= 2:
                Xa = np.stack(Xa); Ya = np.asarray(Ya); Ga = np.asarray(Ga)
                if args.split_by_seed:
                    tr_idx, te_idx = seed_split_train_test(Xa, Ya, Ga, args.test_frac, args.seed)
                else:
                    tr_idx, te_idx = train_test_split(np.arange(len(Ya)), test_size=args.test_frac,
                                                      random_state=args.seed, stratify=Ya)
                clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xa[tr_idx], Ya[tr_idx])
                yhat = clf.predict(Xa[te_idx])
                acc = accuracy_score(Ya[te_idx], yhat)
                try:
                    auc = roc_auc_score(Ya[te_idx], clf.predict_proba(Xa[te_idx])[:, 1])
                except Exception:
                    auc = None
                print(f"  [actor_in@k={k}] n={len(te_idx)} acc={acc:.3f} auc={None if auc is None else round(auc,3)}")
            else:
                print(f"  [actor_in@k={k}] n=0")

        # actor penultimate features
        if args.use_actor_penult:
            Xp, Yp, Gp = [], [], []
            for r in records:
                xp = mean_first_k(r["actor_penult"], k, args.k_start)
                if xp is None:
                    continue
                Xp.append(xp); Yp.append(r["y"]); Gp.append(r["seed"])
            if len(Xp) > 0 and len(set(Yp)) >= 2:
                Xp = np.stack(Xp); Yp = np.asarray(Yp); Gp = np.asarray(Gp)
                if args.split_by_seed:
                    tr_idx, te_idx = seed_split_train_test(Xp, Yp, Gp, args.test_frac, args.seed)
                else:
                    tr_idx, te_idx = train_test_split(np.arange(len(Yp)), test_size=args.test_frac,
                                                    random_state=args.seed, stratify=Yp)
                clf = LogisticRegression(max_iter=200, solver="lbfgs").fit(Xp[tr_idx], Yp[tr_idx])
                yhat = clf.predict(Xp[te_idx])
                acc = accuracy_score(Yp[te_idx], yhat)
                try:
                    auc = roc_auc_score(Yp[te_idx], clf.predict_proba(Xp[te_idx])[:, 1])
                except Exception:
                    auc = None
                print(f"  [actor_penult@k={k}] n={len(te_idx)} acc={acc:.3f} auc={None if auc is None else round(auc,3)}")
            else:
                print(f"  [actor_penult@k={k}] n=0")



if __name__ == "__main__":
    main()
