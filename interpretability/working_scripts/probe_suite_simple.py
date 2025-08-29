#!/usr/bin/env python3
"""
probe_suite_simple.py

Linear probes with group-aware splits (no leakage) + robust fallbacks.
Use --debug to print dataset diagnostics.

Examples:
  python probe_suite_simple.py --data rollouts.npz \
    --hooks hook_env_mlp1 hook_env_mlp3 hook_ltl_rnn_h hook_actor_h5 \
    --task transition --task planning --task actor_alignment --debug
"""

import argparse
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score

# -------------------------
# Default key mapping
# -------------------------
KEYMAP = {
    "obs": "obs",                        # (N, D_obs)
    "next_obs": "next_obs",              # (N, D_obs)
    "action": "action",                  # (N, D_act)
    "ap": "ap",                          # (N, P)
    "next_ap": "next_ap",                # (N, P)
    "next_positives": "next_positives",  # (N, P)
    "vec_to_next_pos": "vec_to_next_pos" # (N, 2)
}

# -------------------------
# I/O
# -------------------------
def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def safe_get(data: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    return data.get(key, None)

# -------------------------
# Debug / summary
# -------------------------
def pct(x): return f"{100*x:.1f}%"

def summarize(data: Dict[str, np.ndarray], hooks: List[str]) -> None:
    print("\n--- Dataset Summary (first 14 keys) ---")
    for i, k in enumerate(sorted(data.keys())):
        if i >= 14: 
            print("... (more keys omitted)")
            break
        arr = data[k]
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        finite = None
        try:
            if arr.ndim == 1:
                finite = np.isfinite(arr).mean()
            else:
                finite = np.all(np.isfinite(arr), axis=1).mean()
        except Exception:
            finite = None
        extras = []
        if k in ("traj_id", "link_idx"):
            extras.append(f"uniq={len(np.unique(arr))}")
        if k == "vec_to_next_pos":
            try:
                v = np.asarray(arr, float)
                ok = np.all(np.isfinite(v), axis=1)
                zn = (np.linalg.norm(v, axis=1) < 1e-8).mean()
                extras.append(f"zero-norm={pct(zn)}")
                extras.append(f"finite-rows={pct(ok.mean())}")
            except Exception:
                pass
        print(f"{k:18s} shape={shape} dtype={dtype} finite-rows={pct(finite) if finite is not None else 'n/a'} {' '.join(extras)}")
    # Hook presence
    missing = [h for h in hooks if h not in data]
    if missing:
        print(f"\n[warn] missing hooks: {missing}")
    # Traj info
    traj = data.get("traj_id", None)
    if traj is not None:
        print(f"traj_id: N={len(traj)}, uniq={len(np.unique(traj))}")
    print("--- End Summary ---\n")

# -------------------------
# Common helpers
# -------------------------
def finite_rows(*arrays: np.ndarray) -> Optional[np.ndarray]:
    """Rows finite across all provided arrays."""
    masks = []
    for arr in arrays:
        if arr is None:
            continue
        if arr.ndim == 1:
            masks.append(np.isfinite(arr))
        else:
            masks.append(np.all(np.isfinite(arr), axis=1))
    if not masks:
        return None
    mask = masks[0]
    for m in masks[1:]:
        mask &= m
    return mask

def ensure_2d(y: np.ndarray) -> np.ndarray:
    return y if y.ndim == 2 else y.reshape(-1, 1)

def need_rows(n: int) -> int:
    """Adaptive minimum rows required to try a probe."""
    return max(10, int(0.02 * n))

def group_split_from(groups: Optional[np.ndarray],
                     n_rows: int,
                     test_size: float = 0.2,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group-aware split; if <2 groups or too small train/test, fallback to row split.
    """
    rng = np.random.default_rng(seed)
    # Fallback: row-wise
    def row_split():
        idx = np.arange(n_rows)
        rng.shuffle(idx)
        n_test = max(1, int(test_size * n_rows))
        te = idx[:n_test]
        tr = idx[n_test:] if n_rows - n_test > 0 else idx[:0]
        return tr, te

    if groups is None:
        return row_split()

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return row_split()

    rng.shuffle(uniq)
    n_test_g = max(1, int(test_size * len(uniq)))
    te_groups = set(uniq[:n_test_g])
    tr_mask = ~np.isin(groups, list(te_groups))
    te_mask =  np.isin(groups, list(te_groups))
    tr = np.where(tr_mask)[0]
    te = np.where(te_mask)[0]

    # If split is too small, fallback
    if len(tr) < 2 or len(te) < 2:
        return row_split()
    return tr, te

def make_next_pairs(arr: np.ndarray, traj_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(arr_t, arr_tp1, mask, pair_groups) where mask keeps within-trajectory pairs."""
    same = (traj_id[:-1] == traj_id[1:])
    return arr[:-1], arr[1:], same, traj_id[:-1]

def ridge_regress(X_tr, y_tr, X_te, y_te) -> float:
    """Ridge with standardization; returns R^2 or NaN if ill-posed."""
    y_te = ensure_2d(y_te); y_tr = ensure_2d(y_tr)
    if X_te.shape[0] < 2 or X_tr.shape[0] < 2:
        return float("nan")
    # avoid undefined R^2
    y_te_flat = y_te.mean(axis=1)
    if np.std(y_te_flat) < 1e-12:
        return float("nan")
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg", Ridge(alpha=1.0)),
    ])
    try:
        model.fit(X_tr, y_tr)
    except Exception:
        return float("nan")
    y_pred = model.predict(X_te)
    try:
        return float(r2_score(y_te, y_pred, multioutput="uniform_average"))
    except Exception:
        return float("nan")

def one_vs_rest_logreg(X_tr, Y_tr, X_te, Y_te, class_weight="balanced") -> Tuple[float, float]:
    """Multi-label (columns) logistic regression, one-vs-rest."""
    if X_tr.shape[0] < 2 or X_te.shape[0] < 2:
        return float("nan"), float("nan")
    n_labels = Y_tr.shape[1]
    accs, f1s = [], []
    for j in range(n_labels):
        ytr = Y_tr[:, j].astype(int)
        yte = Y_te[:, j].astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight=class_weight)),
        ])
        try:
            clf.fit(X_tr, ytr)
        except Exception:
            continue
        yhat = clf.predict(X_te)
        accs.append(accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat))
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.mean(f1s))

def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine(u, v) with numerical safety. Returns (cosine, valid_mask)."""
    un = np.linalg.norm(u, axis=1, keepdims=True)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    valid = (un.squeeze() > eps) & (vn.squeeze() > eps)
    un = np.clip(un, eps, None)
    vn = np.clip(vn, eps, None)
    cos = np.sum(u * v, axis=1) / (un.squeeze() * vn.squeeze())
    return cos, valid

# -------------------------
# Probes
# -------------------------
def run_transition_probe(hook, action, next_obs, traj_id=None, seed=0, debug=False) -> float:
    m = finite_rows(hook, action, next_obs)
    if m is None: return float("nan")
    H, A, Y = hook[m], action[m], next_obs[m]
    if H.shape[0] < need_rows(H.shape[0]): return float("nan")
    X = np.concatenate([H, A], axis=1)
    groups = traj_id[m] if traj_id is not None else None
    tr, te = group_split_from(groups, X.shape[0], 0.2, seed)
    if debug: print(f"[transition] n={X.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(X[tr], Y[tr], X[te], Y[te])

def run_planning_probe(rnn_h, next_positives, traj_id=None, seed=0, debug=False) -> Tuple[float, float]:
    m = finite_rows(rnn_h, next_positives)
    if m is None: return float("nan"), float("nan")
    H, Y = rnn_h[m], next_positives[m].astype(int)
    if H.shape[0] < need_rows(H.shape[0]): return float("nan"), float("nan")
    groups = traj_id[m] if traj_id is not None else None
    tr, te = group_split_from(groups, H.shape[0], 0.2, seed)
    if debug: print(f"[planning] n={H.shape[0]} tr={len(tr)} te={len(te)}")
    return one_vs_rest_logreg(H[tr], Y[tr], H[te], Y[te])

def run_actor_alignment_probe(hook_actor, action, vec_to_next_pos, traj_id=None, seed=0, debug=False) -> float:
    m = finite_rows(hook_actor, action, vec_to_next_pos)
    if m is None: return float("nan")
    H = hook_actor[m]; A = action[m]; V = vec_to_next_pos[m]
    align, valid = cosine(A, V)
    H = H[valid]; align = align[valid]
    if H.shape[0] < need_rows(H.shape[0]) or np.std(align) < 1e-12:
        return float("nan")
    groups = (traj_id[m][valid] if traj_id is not None else None)
    tr, te = group_split_from(groups, H.shape[0], 0.2, seed)
    if debug: print(f"[actor_alignment] n={H.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(H[tr], align[tr].reshape(-1,1), H[te], align[te].reshape(-1,1))

def run_value_next(hook, action, value, traj_id, seed=0, debug=False):
    H_t, H_tp1, m_h, g = make_next_pairs(hook, traj_id)
    A_t, A_tp1, m_a, _ = make_next_pairs(action, traj_id)
    V_t, V_tp1, m_v, _ = make_next_pairs(value, traj_id)
    m = m_h & m_a & m_v
    if m.sum() < need_rows(m.sum()):
        return float("nan")
    X = np.concatenate([H_t[m], A_t[m]], axis=1)
    y = ensure_2d(V_tp1[m])
    tr, te = group_split_from(g[m], X.shape[0], 0.2, seed)
    if debug: print(f"[value_next] n={X.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(X[tr], y[tr], X[te], y[te])

def run_bellman(hook, reward, value, traj_id, gamma=0.99, seed=0, debug=False):
    V_t, V_tp1, m_v, g = make_next_pairs(value, traj_id)
    R_t, R_tp1, m_r, _ = make_next_pairs(ensure_2d(reward), traj_id)
    H_t, H_tp1, m_h, _ = make_next_pairs(hook, traj_id)
    m = m_v & m_r & m_h
    if m.sum() < need_rows(m.sum()):
        return float("nan"), float("nan")
    delta = R_t[m] + gamma * V_tp1[m] - V_t[m]
    X = H_t[m]
    tr, te = group_split_from(g[m], X.shape[0], 0.2, seed)
    if debug: print(f"[bellman] n={X.shape[0]} tr={len(tr)} te={len(te)}")
    r2 = ridge_regress(X[tr], delta[tr], X[te], delta[te])
    rmse = float(np.sqrt(np.mean((delta[te] - np.mean(delta[tr]))**2)))
    return r2, rmse

def run_value_as_distance(hook, vec_to_next_pos, traj_id=None, seed=0, debug=False):
    v = np.asarray(vec_to_next_pos, float)
    m = np.all(np.isfinite(v), axis=1) & np.isfinite(hook).all(axis=1)
    if m.sum() < need_rows(m.sum()):
        return float("nan")
    dist = np.linalg.norm(v[m], axis=1).reshape(-1,1)
    H = hook[m]
    groups = traj_id[m] if traj_id is not None else None
    tr, te = group_split_from(groups, H.shape[0], 0.2, seed)
    if debug: print(f"[value_distance] n={H.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(H[tr], (-dist)[tr], H[te], (-dist)[te])

def run_q1_probe(hook, action, reward, value, traj_id, gamma=0.99, seed=0, debug=False):
    H_t, H_tp1, m_h, g = make_next_pairs(hook, traj_id)
    A_t, A_tp1, m_a, _ = make_next_pairs(action, traj_id)
    V_t, V_tp1, m_v, _ = make_next_pairs(value, traj_id)
    R_t, R_tp1, m_r, _ = make_next_pairs(ensure_2d(reward), traj_id)
    m = m_h & m_a & m_v & m_r
    if m.sum() < need_rows(m.sum()):
        return float("nan")
    X = np.concatenate([H_t[m], A_t[m]], axis=1)
    y = R_t[m] + gamma * V_tp1[m]
    tr, te = group_split_from(g[m], X.shape[0], 0.2, seed)
    if debug: print(f"[q1] n={X.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(X[tr], y[tr], X[te], y[te])

def run_advantage_sign(hook, reward, value, traj_id, gamma=0.99, seed=0, debug=False):
    V_t, V_tp1, m_v, g = make_next_pairs(value, traj_id)
    R_t, R_tp1, m_r, _ = make_next_pairs(reward, traj_id)
    H_t, H_tp1, m_h, _ = make_next_pairs(hook, traj_id)
    m = m_v & m_r & m_h
    if m.sum() < need_rows(m.sum()):
        return float("nan"), float("nan")
    adv = (R_t[m] + gamma * V_tp1[m] - V_t[m]).ravel()
    y = (adv > 0).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    X = H_t[m]
    tr, te = group_split_from(g[m], X.shape[0], 0.2, seed)
    if debug: print(f"[advantage_sign] n={X.shape[0]} tr={len(tr)} te={len(te)}")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    try:
        clf.fit(X[tr], y[tr])
    except Exception:
        return float("nan"), float("nan")
    yhat = clf.predict(X[te])
    return float(accuracy_score(y[te], yhat)), float(f1_score(y[te], yhat))

def build_successor_props(ap, traj_id, gamma=0.99, K=50):
    N, P = ap.shape
    g = np.zeros_like(ap, dtype=float)
    start = 0
    while start < N:
        end = start + 1
        while end < N and traj_id[end] == traj_id[start]:
            end += 1
        seg = ap[start:end].astype(float)
        L = len(seg)
        w = gamma ** np.arange(min(K, L))
        for t in range(L):
            kmax = min(K, L - t)
            g[start + t] = (seg[t:t + kmax] * w[:kmax, None]).sum(axis=0)
        start = end
    return g

def run_successor_probe(hook, ap, traj_id, gamma=0.99, K=50, seed=0, debug=False):
    G = build_successor_props(ap, traj_id, gamma, K)
    if np.var(G) < 1e-12:
        return float("nan")
    m = finite_rows(hook, G)
    if m is None:
        return float("nan")
    H, Y = hook[m], G[m]
    if H.shape[0] < need_rows(H.shape[0]):
        return float("nan")
    tr, te = group_split_from(traj_id[m], H.shape[0], 0.2, seed)
    if debug: print(f"[successor_props] n={H.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(H[tr], Y[tr], H[te], Y[te])

def build_tth(traj_id, link_idx, success_step, cap=100):
    N = len(traj_id)
    tth = np.full(N, cap, dtype=float)
    i = 0
    while i < N:
        j = i + 1
        while j < N and traj_id[j] == traj_id[i] and link_idx[j] == link_idx[i]:
            j += 1
        seg = success_step[i:j].astype(bool)
        nxt = np.where(seg)[0]
        if len(nxt):
            for t in range(j - i):
                k = nxt[nxt >= t]
                if len(k):
                    tth[i + t] = min(cap, k[0] - t)
        i = j
    return tth.reshape(-1, 1)

def run_tth_probe(hook, traj_id, link_idx, success_step, seed=0, debug=False):
    y = build_tth(traj_id, link_idx, success_step, cap=100)
    if np.var(y) < 1e-12:
        return float("nan")
    m = finite_rows(hook, y)
    if m is None:
        return float("nan")
    H, Y = hook[m], y[m]
    if H.shape[0] < need_rows(H.shape[0]):
        return float("nan")
    tr, te = group_split_from(traj_id[m], H.shape[0], 0.2, seed)
    if debug: print(f"[tth] n={H.shape[0]} tr={len(tr)} te={len(te)}")
    return ridge_regress(H[tr], (-Y)[tr], H[te], (-Y)[te])

def delta_r2(base_X, extra_X, y, groups=None, seed=0, debug=False):
    m = finite_rows(base_X, extra_X, y)
    if m is None:
        return float("nan"), float("nan")
    Xb, Xe, Y = base_X[m], extra_X[m], ensure_2d(y[m])
    if Xb.shape[0] < need_rows(Xb.shape[0]):
        return float("nan"), float("nan")
    g = (groups[m] if groups is not None else None)
    tr, te = group_split_from(g, Xb.shape[0], 0.2, seed)
    if debug: print(f"[delta_r2] n={Xb.shape[0]} tr={len(tr)} te={len(te)}")
    r2_b  = ridge_regress(Xb[tr], Y[tr], Xb[te], Y[te])
    Xbe   = np.concatenate([Xb, Xe], axis=1)
    r2_be = ridge_regress(Xbe[tr], Y[tr], Xbe[te], Y[te])
    return r2_be, (r2_be - r2_b)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Simple probe suite (robust & group-aware)")
    ap.add_argument("--data", type=str, required=True, help="Path to rollouts .npz")
    ap.add_argument("--hooks", type=str, nargs="+", default=[],
                    help="Hook keys to consider (e.g., hook_env_mlp1 hook_ltl_rnn_h hook_actor_h5)")
    ap.add_argument("--task", type=str, action="append", default=[],
                    choices=[
                        "transition", "planning", "actor_alignment",
                        "value_prediction", "distance_prediction",
                        "value_next", "bellman", "value_distance",
                        "q1", "advantage_sign", "successor_props",
                        "tth", "delta_r2"
                    ],
                    help="Tasks to run (can pass multiple)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Print dataset diagnostics")
    args = ap.parse_args()

    data = load_npz(args.data)
    print(f"Loaded keys: {list(data.keys())[:10]}{' ...' if len(data.keys())>10 else ''}")

    if args.debug:
        summarize(data, args.hooks)

    traj_id  = safe_get(data, "traj_id")
    link_idx = safe_get(data, "link_idx")

    results: List[Tuple[str, str, Dict[str, float]]] = []

    # --- Transition ---
    if "transition" in args.task:
        action = safe_get(data, KEYMAP["action"])
        next_obs = safe_get(data, KEYMAP["next_obs"])
        if action is None or next_obs is None:
            print("[skip] transition: missing action/next_obs")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None: 
                    print(f"[skip] transition: missing hook '{hk}'")
                    continue
                r2 = run_transition_probe(H, action, next_obs, traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("transition", hk, {"R2_next_obs": r2}))

    # --- Planning ---
    if "planning" in args.task:
        next_pos = safe_get(data, KEYMAP["next_positives"])
        if next_pos is None:
            print("[skip] planning: missing next_positives")
        else:
            rnn_hooks = [k for k in args.hooks if ("ltl" in k or "rnn" in k)] or args.hooks
            picked = False
            for hk in rnn_hooks:
                H = safe_get(data, hk)
                if H is None: 
                    continue
                acc, f1 = run_planning_probe(H, next_pos, traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("planning", hk, {"acc_macro": acc, "f1_macro": f1}))
                picked = True
                break
            if not picked:
                print("[skip] planning: no suitable rnn hook found")

    # --- Actor alignment ---
    if "actor_alignment" in args.task:
        action = safe_get(data, KEYMAP["action"])
        vec = safe_get(data, KEYMAP["vec_to_next_pos"])
        if action is None or vec is None:
            print("[skip] actor_alignment: missing action/vec_to_next_pos")
        else:
            actor_hooks = [k for k in args.hooks if "actor" in k] or args.hooks
            for hk in actor_hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] actor_alignment: missing hook '{hk}'")
                    continue
                r2 = run_actor_alignment_probe(H, action, vec, traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("actor_alignment", hk, {"R2_alignment": r2}))

    # --- Value prediction (single step) ---
    if "value_prediction" in args.task:
        critic = safe_get(data, "critic")
        if critic is None:
            print("[skip] value_prediction: missing 'critic'")
        else:
            zeros = np.zeros((critic.shape[0], 1), dtype=float)
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None: 
                    print(f"[skip] value_prediction: missing hook '{hk}'")
                    continue
                r2 = run_transition_probe(H, zeros, ensure_2d(critic), traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("value_prediction", hk, {"R2_value": r2}))

    # --- Distance prediction (||vec||) ---
    if "distance_prediction" in args.task:
        vec = safe_get(data, KEYMAP["vec_to_next_pos"])
        if vec is None:
            print("[skip] distance_prediction: missing vec_to_next_pos")
        else:
            dist = np.linalg.norm(vec, axis=1).reshape(-1,1)
            zeros = np.zeros((len(dist), 1), dtype=float)
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] distance_prediction: missing hook '{hk}'")
                    continue
                r2 = run_transition_probe(H, zeros, dist, traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("distance_prediction", hk, {"R2_distance": r2}))

    # --- Value next ---
    if "value_next" in args.task:
        critic = safe_get(data, "critic")
        action = safe_get(data, KEYMAP["action"])
        if traj_id is None or critic is None or action is None:
            print("[skip] value_next: need traj_id, critic, action")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] value_next: missing hook '{hk}'")
                    continue
                r2 = run_value_next(H, action, critic, traj_id, seed=args.seed, debug=args.debug)
                results.append(("value_next", hk, {"R2_Vtp1": r2}))

    # --- Bellman residual ---
    if "bellman" in args.task:
        critic = safe_get(data, "critic")
        reward = safe_get(data, "reward")
        if traj_id is None or critic is None or reward is None:
            print("[skip] bellman: need traj_id, critic, reward")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] bellman: missing hook '{hk}'")
                    continue
                r2, rmse = run_bellman(H, ensure_2d(reward), ensure_2d(critic), traj_id, gamma=0.99, seed=args.seed, debug=args.debug)
                results.append(("bellman", hk, {"R2_delta": r2, "RMSE_delta": rmse}))

    # --- Value vs distance ---
    if "value_distance" in args.task:
        vec = safe_get(data, "vec_to_next_pos")
        if vec is None:
            print("[skip] value_distance: need vec_to_next_pos")
        else:
            for hk in args.hooks + ["critic"]:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] value_distance: missing '{hk}'")
                    continue
                H2 = H if H.ndim > 1 else H.reshape(-1,1)
                r2 = run_value_as_distance(H2, vec, traj_id=traj_id, seed=args.seed, debug=args.debug)
                results.append(("value_distance", hk, {"R2_-dist": r2}))

    # --- Q1 ---
    if "q1" in args.task:
        critic = safe_get(data, "critic")
        reward = safe_get(data, "reward")
        action = safe_get(data, KEYMAP["action"])
        if traj_id is None or critic is None or reward is None or action is None:
            print("[skip] q1: need traj_id, critic, reward, action")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] q1: missing hook '{hk}'")
                    continue
                r2 = run_q1_probe(H, action, reward, critic, traj_id, gamma=0.99, seed=args.seed, debug=args.debug)
                results.append(("q1", hk, {"R2_Q1": r2}))

    # --- Advantage sign ---
    if "advantage_sign" in args.task:
        critic = safe_get(data, "critic")
        reward = safe_get(data, "reward")
        if traj_id is None or critic is None or reward is None:
            print("[skip] advantage_sign: need traj_id, critic, reward")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] advantage_sign: missing hook '{hk}'")
                    continue
                acc, f1 = run_advantage_sign(H, reward, critic, traj_id, gamma=0.99, seed=args.seed, debug=args.debug)
                results.append(("advantage_sign", hk, {"acc": acc, "F1": f1}))

    # --- Delta R^2 (baseline: [obs, action]) ---
    if "delta_r2" in args.task:
        obs = safe_get(data, "obs")
        action = safe_get(data, KEYMAP["action"])
        next_obs = safe_get(data, "next_obs")
        critic = safe_get(data, "critic")
        if obs is None or action is None or next_obs is None or critic is None:
            print("[skip] delta_r2: need obs, action, next_obs, critic")
        else:
            base_X = np.concatenate([obs, action], axis=1)
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] delta_r2 transition: missing hook '{hk}'")
                    continue
                r2_be, delta = delta_r2(base_X, H, next_obs, groups=traj_id, seed=args.seed, debug=args.debug)
                results.append(("delta_r2_transition", hk, {"R2_total": r2_be, "R2_delta": delta}))
            # value_next variant (pairwise)
            if traj_id is not None:
                def pairs(x): return x[:-1], x[1:], (traj_id[:-1] == traj_id[1:]), traj_id[:-1]
                obs_t, obs_tp1, m_o, g = pairs(obs)
                act_t, act_tp1, m_a, _ = pairs(action)
                critic_t, critic_tp1, m_v, _ = pairs(critic)
                m = m_o & m_a & m_v
                if m.sum() >= need_rows(m.sum()):
                    for hk in args.hooks:
                        H = safe_get(data, hk)
                        if H is None:
                            print(f"[skip] delta_r2_value_next: missing hook '{hk}'")
                            continue
                        H_t, H_tp1, m_h, _ = pairs(H)
                        mm = m & m_h
                        if mm.sum() >= need_rows(mm.sum()):
                            base_pairs = np.concatenate([obs_t[mm], act_t[mm]], axis=1)
                            r2_be, delta = delta_r2(base_pairs, H_t[mm], ensure_2d(critic_tp1[mm]), groups=g[mm], seed=args.seed, debug=args.debug)
                            results.append(("delta_r2_value_next", hk, {"R2_total": r2_be, "R2_delta": delta}))

    if not results:
        print("No results produced. Run with --debug and check keys/masks/variance.")
        sys.exit(0)

    col1_w = max(len(r[0]) for r in results) + 2
    col2_w = max(len(r[1]) for r in results) + 2
    print("\n=== Probe Results ===")
    print(f"{'task'.ljust(col1_w)}{'hook'.ljust(col2_w)}metrics")
    for task, hook, metrics in results:
        items = []
        for k, v in metrics.items():
            items.append(f"{k}={v:.4f}" if isinstance(v, float) and np.isfinite(v) else f"{k}={v}")
        print(f"{task.ljust(col1_w)}{hook.ljust(col2_w)}{', '.join(items)}")

if __name__ == "__main__":
    main()
