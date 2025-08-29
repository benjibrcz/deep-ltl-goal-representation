"""
probe_suite_simple.py

A simple, self-contained script to run linear probes for a few core hypotheses:
- Transition substrate in env_net (predict next observation given current hook + action)
- Planning content in ltl_net (decode next positives from GRU state)
- Plan-conditioned control in actor (decode plan-alignment from actor trunk)

The script expects a single .npz file containing arrays. You can adapt the KEYMAP
below to match your dump's key names.

Example usage:
    python probe_suite_simple.py --data rollouts.npz \
        --hooks hook_env_mlp1 hook_env_mlp3 hook_ltl_rnn_h hook_actor_h5 \
        --task transition --task planning --task actor_alignment

Notes:
- Keep it simple & readable: no deep frameworks, just sklearn Ridge/Logistic.
- Skips gracefully if a requested key is missing.
- Reports R^2/Accuracy in a compact table.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# sklearn is widely available and keeps the script simple.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# -------------------------
# Default key mapping
# -------------------------
# Adjust these if your rollout dump uses different names.
KEYMAP = {
    # Observations & actions
    "obs": "obs",                 # (N, D_obs) current flattened observation vector (optional if not used)
    "next_obs": "next_obs",       # (N, D_obs) next flattened observation vector (for transition probe)
    "action": "action",           # (N, D_act) continuous action taken (e.g., 2-dim)
    "ap": "ap",                   # (N, P) boolean/binary AP truth vector at s_t
    "next_ap": "next_ap",         # (N, P) boolean/binary AP truth vector at s_{t+1}

    # Planning-related labels
    "next_positives": "next_positives",   # (N, P) binary: which APs are required next by chosen sequence
    "vec_to_next_pos": "vec_to_next_pos", # (N, 2) vector from agent -> next-positive target zone (for alignment)
    "nearest_zone_vec": "nearest_zone_vec", # (N, 2) optional: vector to nearest zone (for baseline)

    # Hooks (activations) — examples; pass actual names via --hooks
    # e.g. "hook_env_mlp1": (N, d1), "hook_ltl_rnn_h": (N, d2), "hook_actor_h5": (N, d3)
}

# -------------------------
# Utilities
# -------------------------

def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    return out

def safe_get(data: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    return data.get(key, None)

def finite_rows(*arrays: np.ndarray) -> np.ndarray:
    """Return a boolean mask of rows where all arrays are finite (no NaNs/inf)."""
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
        mask = mask & m
    return mask

def train_test_indices(n: int, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return train_idx, test_idx


def trajectory_aware_split(traj_id: np.ndarray, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split trajectories into train/test sets to avoid temporal leakage."""
    np.random.seed(seed)
    unique_trajs = np.unique(traj_id)
    n_test = max(1, int(len(unique_trajs) * test_size))
    
    # Randomly select trajectories for test set
    test_trajs = np.random.choice(unique_trajs, size=n_test, replace=False)
    train_trajs = np.setdiff1d(unique_trajs, test_trajs)
    
    # Get indices for each set
    train_mask = np.isin(traj_id, train_trajs)
    test_mask = np.isin(traj_id, test_trajs)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return train_indices, test_indices


def world_aware_split(world_id: np.ndarray, test_size: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Split by worlds to ensure complete separation and no data leakage."""
    np.random.seed(seed)
    unique_worlds = np.unique(world_id)
    n_test = max(1, int(len(unique_worlds) * test_size))
    
    # Randomly select worlds for test set
    test_worlds = np.random.choice(unique_worlds, size=n_test, replace=False)
    train_worlds = np.setdiff1d(unique_worlds, test_worlds)
    
    # Get indices for each set
    train_mask = np.isin(world_id, train_worlds)
    test_mask = np.isin(world_id, test_worlds)
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    return train_indices, test_indices

def one_vs_rest_logreg(X_tr, Y_tr, X_te, Y_te) -> Tuple[float, float]:
    """
    Multi-label (columns) logistic regression, one-vs-rest.
    Returns (macro_accuracy, macro_f1).
    """
    n_labels = Y_tr.shape[1]
    accs, f1s = [], []
    for j in range(n_labels):
        ytr = Y_tr[:, j].astype(int)
        yte = Y_te[:, j].astype(int)
        # Skip degenerate labels
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        clf = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))])
        clf.fit(X_tr, ytr)
        yhat = clf.predict(X_te)
        accs.append(accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat))
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.mean(f1s))

def ridge_regress(X_tr, y_tr, X_te, y_te) -> float:
    """
    Fit Ridge regression with standardization. Return R^2.
    Safely handles vector or multioutput targets and bails out when y_te is (near) constant.
    """
    model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                      ("reg", Ridge(alpha=1.0))])

    # if y_te has ~zero variance, R^2 is undefined / non-finite per sklearn docs
    # (constant y_true -> R^2 is NaN for perfect, -Inf for imperfect). We exit early.
    y_te_flat = y_te if y_te.ndim == 1 else y_te.mean(axis=1)
    if X_te.shape[0] < 2 or np.std(y_te_flat) < 1e-12:
        return float("nan")

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return float(r2_score(y_te, y_pred, multioutput="uniform_average"))


def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cosine(u, v) with numerical safety.
    Returns:
        cos  : shape (N,) cosine similarities in [-1, 1]
        valid: shape (N,) boolean mask where both norms are > eps
    """
    un = np.linalg.norm(u, axis=1, keepdims=True)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    valid = (un.squeeze() > eps) & (vn.squeeze() > eps)

    un = np.clip(un, eps, None)
    vn = np.clip(vn, eps, None)
    cos = np.sum(u * v, axis=1) / (un.squeeze() * vn.squeeze())
    return cos, valid


# -------------------------
# Probe runners
# -------------------------

def run_transition_probe(hook: np.ndarray, action: np.ndarray, next_obs: np.ndarray, seed: int = 0) -> float:
    """
    Transition substrate: predict next_obs from [hook, action].
    Reports R^2.
    """
    mask = finite_rows(hook, action, next_obs)
    hook, action, next_obs = hook[mask], action[mask], next_obs[mask]
    X = np.concatenate([hook, action], axis=1)
    n = X.shape[0]
    tr, te = train_test_indices(n, test_size=0.2, seed=seed)
    r2 = ridge_regress(X[tr], next_obs[tr], X[te], next_obs[te])
    return r2

def run_planning_probe(rnn_h: np.ndarray, next_positives: np.ndarray, seed: int = 0) -> Tuple[float, float]:
    """
    Planning content: predict next-positive APs from LTL GRU final state.
    Reports (macro-accuracy, macro-F1).
    """
    mask = finite_rows(rnn_h, next_positives)
    rnn_h, Y = rnn_h[mask], next_positives[mask].astype(int)
    n = rnn_h.shape[0]
    tr, te = train_test_indices(n, test_size=0.2, seed=seed)
    acc, f1 = one_vs_rest_logreg(rnn_h[tr], Y[tr], rnn_h[te], Y[te])
    return acc, f1

def run_actor_alignment_probe(hook_actor, action, vec_to_next_pos, seed=0) -> float:
    """
    Decode plan-alignment: predict cos(action, vec_to_next_pos) from actor hook.
    Masks invalid rows (zero-norm vectors) and skips degenerate test targets.
    """
    mask = finite_rows(hook_actor, action, vec_to_next_pos)
    if mask is None or mask.sum() < 5:
        return float("nan")

    H = hook_actor[mask]
    A = action[mask]
    V = vec_to_next_pos[mask]

    align, valid = cosine(A, V)
    H = H[valid]
    align = align[valid]

    if H.shape[0] < 10 or np.std(align) < 1e-12:
        return float("nan")

    tr, te = train_test_indices(H.shape[0], test_size=0.2, seed=seed)
    return ridge_regress(H[tr], align[tr], H[te], align[te])


def make_next_pairs(arr, traj_id):
    """Return (arr_t, arr_tp1, mask) where mask keeps only within-trajectory pairs."""
    n = len(arr)
    same = (traj_id[:-1] == traj_id[1:])
    arr_t   = arr[:-1]
    arr_tp1 = arr[1:]
    mask = same
    return arr_t, arr_tp1, mask


def run_value_next(hook, action, value, traj_id, seed=0):
    """Critic dynamics probe: predict next value from current hook + action."""
    h_t, h_tp1, m1 = make_next_pairs(hook, traj_id)
    a_t, a_tp1, m2 = make_next_pairs(action, traj_id)
    v_t, v_tp1, m3 = make_next_pairs(value, traj_id)
    mask = m1 & m2 & m3
    if mask.sum() < 10:
        return float("nan")
    X = np.concatenate([h_t[mask], a_t[mask]], axis=1)
    y = v_tp1[mask]
    tr, te = train_test_indices(X.shape[0], 0.2, seed)
    return ridge_regress(X[tr], y[tr], X[te], y[te])


def run_bellman(hook, reward, value, traj_id, gamma=0.99, seed=0):
    """Bellman residual probe: check critic consistency and predictability of residuals."""
    v_t, v_tp1, m_v = make_next_pairs(value, traj_id)
    r_t, r_tp1, m_r = make_next_pairs(reward, traj_id)
    H_t, H_tp1, m_h = make_next_pairs(hook, traj_id)
    mask = m_v & m_r & m_h
    if mask.sum() < 50:
        return float("nan"), float("nan")
    delta = r_t[mask].reshape(-1,1) + gamma * v_tp1[mask] - v_t[mask]
    X = H_t[mask]
    tr, te = train_test_indices(X.shape[0], 0.2, seed)
    r2 = ridge_regress(X[tr], delta[tr], X[te], delta[te])
    rmse = float(np.sqrt(np.mean((delta[te] - np.mean(delta[tr]))**2)))
    return r2, rmse


def run_value_as_distance(hook, vec_to_next_pos, seed=0):
    """Value-as-distance probe: does value encode distance to goal?"""
    v = np.asarray(vec_to_next_pos, float)
    mask = np.all(np.isfinite(v), axis=1)
    if mask.sum() < 10:
        return float("nan")
    dist = np.linalg.norm(v[mask], axis=1).reshape(-1,1)
    H = hook[mask]
    tr, te = train_test_indices(H.shape[0], 0.2, seed)
    # regress negative distance so "higher is better"
    return ridge_regress(H[tr], (-dist)[tr], H[te], (-dist)[te])


def run_q1_probe(hook, action, reward, value, traj_id, gamma=0.99, seed=0):
    """One-step Q probe: predict Q1(s_t, a_t) = r_t + γV_{t+1} from hook + action."""
    # align consecutive pairs
    def pairs(x):
        return x[:-1], x[1:], traj_id[:-1] == traj_id[1:]
    H_t, H_tp1, m_h = pairs(hook)
    A_t, A_tp1, m_a = pairs(action)
    V_t, V_tp1, m_v = pairs(value)
    R_t, R_tp1, m_r = pairs(reward.reshape(-1,1))
    m = m_h & m_a & m_v & m_r
    if m.sum() < 50:
        return float("nan")
    X = np.concatenate([H_t[m], A_t[m]], axis=1)
    y = R_t[m] + gamma * V_tp1[m]
    tr, te = train_test_indices(X.shape[0], 0.2, seed)
    return ridge_regress(X[tr], y[tr], X[te], y[te])


def run_advantage_sign(hook, reward, value, traj_id, gamma=0.99, seed=0):
    """Advantage sign probe: predict if action increased value (advantage > 0)."""
    V_t, V_tp1, m_v = value[:-1], value[1:], traj_id[:-1]==traj_id[1:]
    R_t = reward[:-1]
    H_t = hook[:-1]
    m = m_v
    if m.sum() < 50:  # need some positives/negatives
        return float("nan"), float("nan")
    adv = (R_t[m] + gamma*V_tp1[m].flatten() - V_t[m].flatten())
    y = (adv > 0).astype(int)
    X = H_t[m]
    clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    tr, te = train_test_indices(X.shape[0], 0.2, seed)
    clf.fit(X[tr], y[tr])
    yhat = clf.predict(X[te])
    return float(accuracy_score(y[te], yhat)), float(f1_score(y[te], yhat))


def build_successor_props(ap, traj_id, gamma=0.99, K=50):
    """Build successor proposition features: discounted future AP counts."""
    N, P = ap.shape
    g = np.zeros_like(ap, dtype=float)
    # process per-trajectory forward
    start = 0
    while start < N:
        end = start + 1
        while end < N and traj_id[end]==traj_id[start]:
            end += 1
        seg = ap[start:end].astype(float)
        # vectorized forward pass
        w = gamma ** np.arange(min(K, len(seg)))
        for t in range(len(seg)):
            kmax = min(K, len(seg)-t)
            g[start+t] = (seg[t:t+kmax] * w[:kmax, None]).sum(axis=0)
        start = end
    return g


def run_successor_probe(hook, ap, traj_id, gamma=0.99, K=50, seed=0):
    """Successor propositions probe: predict discounted future AP counts."""
    G = build_successor_props(ap, traj_id, gamma, K)
    # Check if G has any variance (not all zeros)
    if np.var(G) < 1e-12:
        return float("nan")  # No variance in successor features
    m = finite_rows(hook, G)
    if m is None or m.sum() < 10:
        return float("nan")
    H, Y = hook[m], G[m]
    tr, te = train_test_indices(H.shape[0], 0.2, seed)
    return ridge_regress(H[tr], Y[tr], H[te], Y[te])


def run_successor_probe_modified(hook, vec_to_next_pos, traj_id, gamma=0.99, K=20, seed=0):
    """Modified successor probe: predict discounted future distances to goal."""
    # Instead of APs, use distance to goal as a temporal target
    distances = np.linalg.norm(vec_to_next_pos, axis=1)
    
    # Build discounted future distances
    N = len(distances)
    g = np.zeros(N, dtype=float)
    
    # Process per-trajectory forward
    start = 0
    while start < N:
        end = start + 1
        while end < N and traj_id[end]==traj_id[start]:
            end += 1
        seg = distances[start:end]
        
        # Vectorized forward pass: for each t, predict discounted future distances
        w = gamma ** np.arange(min(K, len(seg)))
        for t in range(len(seg)):
            kmax = min(K, len(seg)-t)
            g[start+t] = (seg[t:t+kmax] * w[:kmax]).sum()
        start = end
    
    # Check if we have variance
    if np.var(g) < 1e-12:
        return float("nan")
    
    m = finite_rows(hook, g.reshape(-1,1))
    if m is None or m.sum() < 10:
        return float("nan")
    H, Y = hook[m], g[m].reshape(-1,1)
    traj_id_masked = traj_id[m]
    
    # Use world-aware splitting to ensure complete separation
    world_ids = traj_id_masked // 10000
    tr, te = world_aware_split(world_ids, 0.2, seed)
    return ridge_regress(H[tr], Y[tr], H[te], Y[te])


def build_tth(traj_id, link_idx, success_step, cap=100):
    """Build time-to-hit: steps until success within current link."""
    N = len(traj_id)
    tth = np.full(N, cap, dtype=float)
    i = 0
    while i < N:
        j = i+1
        while j < N and traj_id[j]==traj_id[i] and link_idx[j]==link_idx[i]:
            j += 1
        seg = success_step[i:j].astype(bool)
        # from each t, distance to next True
        nxt = np.where(seg)[0]
        if len(nxt):
            for t in range(j-i):
                k = nxt[nxt >= t]
                if len(k): tth[i+t] = min(cap, k[0]-t)
        i = j
    return tth.reshape(-1,1)


def run_tth_probe(hook, traj_id, link_idx, success_step, seed=0):
    """Time-to-hit probe: predict steps until success within current link."""
    y = build_tth(traj_id, link_idx, success_step, cap=100)
    # Check if y has any variance (not all cap values)
    if np.var(y) < 1e-12:
        return float("nan")  # No variance in TTH (all at cap)
    m = finite_rows(hook, y)
    if m is None or m.sum() < 10:
        return float("nan")
    H, Y = hook[m], y[m]
    tr, te = train_test_indices(H.shape[0], 0.2, seed)
    # regress negative TTH so higher is "closer"
    return ridge_regress(H[tr], (-Y)[tr], H[te], (-Y)[te])


def run_tth_probe_modified(hook, vec_to_next_pos, traj_id, link_idx, seed=0):
    """Modified TTH probe: predict distance progression within current link."""
    # Instead of time-to-success, predict how distance changes within a link
    distances = np.linalg.norm(vec_to_next_pos, axis=1)
    
    # Build distance progression: for each step, predict final distance in this link
    N = len(distances)
    y = np.zeros(N, dtype=float)
    
    # Process per-trajectory and per-link
    start = 0
    while start < N:
        end = start + 1
        while end < N and traj_id[end]==traj_id[start] and link_idx[end]==link_idx[start]:
            end += 1
        
        # For each step in this link, predict the final distance
        seg = distances[start:end]
        if len(seg) > 1:
            final_dist = seg[-1]  # Distance at the end of this link
            for t in range(len(seg)):
                y[start+t] = final_dist
        else:
            y[start+t] = seg[0]  # Single step link
        
        start = end
    
    # Check if we have variance
    if np.var(y) < 1e-12:
        return float("nan")
    
    m = finite_rows(hook, y.reshape(-1,1))
    if m is None or m.sum() < 10:
        return float("nan")
    H, Y = hook[m], y[m].reshape(-1,1)
    traj_id_masked = traj_id[m]
    
    # Use world-aware splitting to ensure complete separation
    world_ids = traj_id_masked // 10000
    tr, te = world_aware_split(world_ids, 0.2, seed)
    return ridge_regress(H[tr], Y[tr], H[te], Y[te])


def build_success_H(traj_id, link_idx, success_step, H=20):
    """Build success within H steps: binary classification."""
    y = np.zeros(len(traj_id), dtype=int)
    i = 0
    while i < len(traj_id):
        j = i+1
        while j < len(traj_id) and traj_id[j]==traj_id[i] and link_idx[j]==link_idx[i]:
            j += 1
        seg = success_step[i:j].astype(bool)
        for t in range(j-i):
            y[i+t] = int(seg[t:min(j-i, t+H)].any())
        i = j
    return y


def run_success_H(hook, traj_id, link_idx, success_step, H=20, seed=0):
    """Success within H steps probe: binary classification."""
    y = build_success_H(traj_id, link_idx, success_step, H=H)
    m = finite_rows(hook, y.reshape(-1,1))
    if m is None or m.sum() < 10:
        return float("nan"), float("nan")
    X, Y = hook[m], y[m]
    clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    tr, te = train_test_indices(X.shape[0], 0.2, seed)
    clf.fit(X[tr], Y[tr])
    yhat = clf.predict(X[te])
    return float(accuracy_score(Y[te], yhat)), float(f1_score(Y[te], yhat))


def run_success_H_modified(hook, vec_to_next_pos, traj_id, link_idx, H=20, seed=0):
    """Modified success_H probe: predict if we'll get close to goal within H steps."""
    # Instead of binary success, predict if we'll get within a threshold distance
    distances = np.linalg.norm(vec_to_next_pos, axis=1)
    threshold = 2.0  # Consider "close" if within 2 units
    
    # Build target: will we get close within H steps in this link?
    N = len(distances)
    y = np.zeros(N, dtype=int)
    
    # Process per-trajectory and per-link
    start = 0
    while start < N:
        end = start + 1
        while end < N and traj_id[end]==traj_id[start] and link_idx[end]==link_idx[start]:
            end += 1
        
        # For each step in this link, check if we get close within H steps
        seg = distances[start:end]
        for t in range(len(seg)):
            # Look ahead up to H steps
            look_ahead = min(H, len(seg) - t)
            if look_ahead > 0:
                # Check if any of the next H steps get us close
                future_distances = seg[t:t+look_ahead]
                y[start+t] = int(np.any(future_distances < threshold))
            else:
                y[start+t] = 0  # End of link
        
        start = end
    
    # Check if we have variance
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    
    m = finite_rows(hook, y.reshape(-1,1))
    if m is None or m.sum() < 10:
        return float("nan"), float("nan")
    X, Y = hook[m], y[m]
    clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    traj_id_masked = traj_id[m]
    
    # Use world-aware splitting to ensure complete separation
    world_ids = traj_id_masked // 10000
    tr, te = world_aware_split(world_ids, 0.2, seed)
    clf.fit(X[tr], Y[tr])
    yhat = clf.predict(X[te])
    return float(accuracy_score(Y[te], yhat)), float(f1_score(Y[te], yhat))


def delta_r2(base_X, extra_X, y, seed=0):
    """Compute incremental R²: how much does extra_X add beyond base_X?"""
    m = finite_rows(base_X, extra_X, y)
    if m is None or m.sum() < 10:
        return float("nan"), float("nan")
    Xb, Xe, Y = base_X[m], extra_X[m], y[m]
    tr, te = train_test_indices(Xb.shape[0], 0.2, seed)
    # baseline
    r2_b = ridge_regress(Xb[tr], Y[tr], Xb[te], Y[te])
    # baseline + hook
    Xbe = np.concatenate([Xb, Xe], axis=1)
    r2_be = ridge_regress(Xbe[tr], Y[tr], Xbe[te], Y[te])
    return r2_be, (r2_be - r2_b)



# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple probe suite")
    ap.add_argument("--data", type=str, required=True, help="Path to rollouts .npz")
    ap.add_argument("--hooks", type=str, nargs="+", default=[],
                    help="Hook keys to consider (e.g., hook_env_mlp1 hook_ltl_rnn_h hook_actor_h5)")
    ap.add_argument("--task", type=str, action="append", default=[],
                    choices=["transition", "planning", "actor_alignment", "value_prediction", "success_prediction", "distance_prediction", "action_quality", "value_next", "bellman", "value_distance", "q1", "advantage_sign", "successor_props", "successor_props_modified", "tth", "tth_modified", "success_H", "success_H_modified", "delta_r2"],
                    help="Which tasks to run (can pass multiple)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_npz(args.data)
    print(f"Loaded keys: {list(data.keys())[:10]}{' ...' if len(data.keys())>10 else ''}")

    results = []

    # Transition probe: requires any hook + action + next_obs
    if "transition" in args.task:
        for hk in args.hooks:
            hook = safe_get(data, hk)
            if hook is None:
                print(f"[skip] transition: missing hook '{hk}'")
                continue
            action = safe_get(data, KEYMAP["action"])
            next_obs = safe_get(data, KEYMAP["next_obs"])
            if action is None or next_obs is None:
                print(f"[skip] transition: missing '{KEYMAP['action']}' or '{KEYMAP['next_obs']}'")
                break
            try:
                r2 = run_transition_probe(hook, action, next_obs, seed=args.seed)
                results.append(("transition", hk, {"R2_next_obs": r2}))
            except Exception as e:
                print(f"[error] transition on '{hk}': {e}")

    # Planning probe: LTL GRU final state -> next positives
    if "planning" in args.task:
        # try typical keys in order
        candidate_rnn_keys = [k for k in args.hooks if "ltl" in k or "rnn" in k] or ["hook_ltl_rnn_h"]
        rnn_key = None
        for k in candidate_rnn_keys:
            if k in data:
                rnn_key = k
                break
        if rnn_key is None:
            print("[skip] planning: no rnn hook found in provided hooks")
        else:
            rnn_h = data[rnn_key]
            next_pos = safe_get(data, KEYMAP["next_positives"])
            if next_pos is None:
                print(f"[skip] planning: missing '{KEYMAP['next_positives']}'")
            else:
                try:
                    acc, f1 = run_planning_probe(rnn_h, next_pos, seed=args.seed)
                    results.append(("planning", rnn_key, {"acc_macro": acc, "f1_macro": f1}))
                except Exception as e:
                    print(f"[error] planning on '{rnn_key}': {e}")

    # Actor alignment probe: actor trunk -> cos(action, vec_to_next_pos)
    if "actor_alignment" in args.task:
        # choose an actor hook from provided hooks
        actor_key = None
        for k in args.hooks:
            if "actor" in k and ("h" in k or "enc" in k):
                if k in data:
                    actor_key = k
                    break
        if actor_key is None:
            print("[skip] actor_alignment: no actor hook found in provided hooks")
        else:
            H = data[actor_key]
            action = safe_get(data, KEYMAP["action"])
            vec = safe_get(data, KEYMAP["vec_to_next_pos"])
            if action is None or vec is None:
                print(f"[skip] actor_alignment: missing '{KEYMAP['action']}' or '{KEYMAP['vec_to_next_pos']}'")
            else:
                try:
                    r2 = run_actor_alignment_probe(H, action, vec, seed=args.seed)
                    results.append(("actor_alignment", actor_key, {"R2_alignment": r2}))
                except Exception as e:
                    print(f"[error] actor_alignment on '{actor_key}': {e}")

    # Value prediction probe: any hook -> critic value
    if "value_prediction" in args.task:
        for hk in args.hooks:
            hook = safe_get(data, hk)
            if hook is None:
                print(f"[skip] value_prediction: missing hook '{hk}'")
                continue
            critic_value = safe_get(data, "critic")
            if critic_value is None:
                print(f"[skip] value_prediction: missing 'critic'")
                break
            try:
                r2 = run_transition_probe(hook, np.zeros((hook.shape[0], 1)), critic_value, seed=args.seed)
                results.append(("value_prediction", hk, {"R2_value": r2}))
            except Exception as e:
                print(f"[error] value_prediction on '{hk}': {e}")

    # Multi-step success prediction probe: any hook -> success in future steps
    if "success_prediction" in args.task:
        for hk in args.hooks:
            hook = safe_get(data, hk)
            if hook is None:
                print(f"[skip] success_prediction: missing hook '{hk}'")
                continue
            # Use next_positives as a proxy for immediate success
            success_target = safe_get(data, KEYMAP["next_positives"])
            if success_target is None:
                print(f"[skip] success_prediction: missing '{KEYMAP['next_positives']}'")
                break
            try:
                # Create a more interesting target: predict if we'll have multiple successes in a window
                # This creates variance and tests the network's ability to predict future success patterns
                window_size = 5
                if len(success_target) >= window_size:
                    # For each step, predict if we'll have at least 2 successes in the next window_size steps
                    future_success = []
                    for i in range(len(success_target) - window_size + 1):
                        window_successes = success_target[i:i+window_size].sum(axis=1)
                        future_success.append((window_successes >= 2).astype(float).mean())
                    
                    # Pad the beginning to match the original length
                    padded_success = np.array([0.0] * (window_size - 1) + future_success)
                    
                    r2 = run_transition_probe(hook, np.zeros((hook.shape[0], 1)), padded_success.reshape(-1, 1), seed=args.seed)
                    results.append(("success_prediction", hk, {"R2_future_success": r2}))
                else:
                    print(f"[skip] success_prediction: insufficient data for window size {window_size}")
            except Exception as e:
                print(f"[error] success_prediction on '{hk}': {e}")

    # Distance to goal probe: any hook -> distance to current goal
    if "distance_prediction" in args.task:
        for hk in args.hooks:
            hook = safe_get(data, hk)
            if hook is None:
                print(f"[skip] distance_prediction: missing hook '{hk}'")
                continue
            # Use vec_to_next_pos magnitude as distance to goal
            vec_to_goal = safe_get(data, KEYMAP["vec_to_next_pos"])
            if vec_to_goal is None:
                print(f"[skip] distance_prediction: missing '{KEYMAP['vec_to_next_pos']}'")
                break
            try:
                # Compute distance magnitude
                distance = np.linalg.norm(vec_to_goal, axis=1, keepdims=True)
                r2 = run_transition_probe(hook, np.zeros((hook.shape[0], 1)), distance, seed=args.seed)
                results.append(("distance_prediction", hk, {"R2_distance": r2}))
            except Exception as e:
                print(f"[error] distance_prediction on '{hk}': {e}")

    # Action quality prediction probe: any hook -> action quality (how good the action is)
    if "action_quality" in args.task:
        for hk in args.hooks:
            hook = safe_get(data, hk)
            if hook is None:
                print(f"[skip] action_quality: missing hook '{hk}'")
                continue
            # Use the change in distance to goal as a proxy for action quality
            vec_to_goal = safe_get(data, KEYMAP["vec_to_next_pos"])
            if vec_to_goal is None:
                print(f"[skip] action_quality: missing '{KEYMAP['vec_to_next_pos']}'")
                break
            try:
                # Compute action quality: negative change in distance (closer = better action)
                # This requires comparing current and next distances
                if len(vec_to_goal) > 1:
                    current_distances = np.linalg.norm(vec_to_goal[:-1], axis=1)
                    next_distances = np.linalg.norm(vec_to_goal[1:], axis=1)
                    action_quality = -(next_distances - current_distances)  # Negative because closer is better
                    
                    # Pad to match original length
                    padded_quality = np.array([0.0] + action_quality.tolist())
                    
                    r2 = run_transition_probe(hook, np.zeros((hook.shape[0], 1)), padded_quality.reshape(-1, 1), seed=args.seed)
                    results.append(("action_quality", hk, {"R2_action_quality": r2}))
                else:
                    print(f"[skip] action_quality: insufficient data for comparison")
            except Exception as e:
                print(f"[error] action_quality on '{hk}': {e}")

    # Critic dynamics probe: predict next value from current hook + action
    if "value_next" in args.task:
        traj_id = safe_get(data, "traj_id")
        critic  = safe_get(data, "critic")
        if traj_id is None or critic is None:
            print("[skip] value_next: need 'traj_id' and 'critic' in NPZ")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] value_next: missing hook '{hk}'")
                    continue
                r2 = run_value_next(H, safe_get(data, KEYMAP["action"]), critic, traj_id, seed=args.seed)
                results.append(("value_next", hk, {"R2_Vtp1": r2}))

    # Bellman residual probe: check critic consistency and predictability
    if "bellman" in args.task:
        traj_id = safe_get(data, "traj_id")
        critic  = safe_get(data, "critic")
        reward  = safe_get(data, "reward")
        if traj_id is None or critic is None or reward is None:
            print("[skip] bellman: need 'traj_id', 'critic', 'reward'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] bellman: missing hook '{hk}'")
                    continue
                r2, rmse = run_bellman(H, reward.reshape(-1,1), critic.reshape(-1,1), traj_id, gamma=0.99, seed=args.seed)
                results.append(("bellman", hk, {"R2_delta": r2, "RMSE_delta": rmse}))

    # Value-as-distance probe: does value encode distance to goal?
    if "value_distance" in args.task:
        vec = safe_get(data, "vec_to_next_pos")
        if vec is None:
            print("[skip] value_distance: need 'vec_to_next_pos'")
        else:
            for hk in args.hooks + ["critic"]:  # include scalar value as a "hook"
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] value_distance: missing '{hk}'")
                    continue
                r2 = run_value_as_distance(H if H.ndim>1 else H.reshape(-1,1), vec, seed=args.seed)
                results.append(("value_distance", hk, {"R2_-dist": r2}))

    # One-step Q probe: predict Q1(s_t, a_t) = r_t + γV_{t+1}
    if "q1" in args.task:
        traj_id = safe_get(data, "traj_id")
        critic  = safe_get(data, "critic")
        reward  = safe_get(data, "reward")
        if traj_id is None or critic is None or reward is None:
            print("[skip] q1: need 'traj_id', 'critic', 'reward'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] q1: missing hook '{hk}'")
                    continue
                r2 = run_q1_probe(H, safe_get(data, KEYMAP["action"]), reward, critic, traj_id, gamma=0.99, seed=args.seed)
                results.append(("q1", hk, {"R2_Q1": r2}))

    # Advantage sign probe: predict if action increased value
    if "advantage_sign" in args.task:
        traj_id = safe_get(data, "traj_id")
        critic  = safe_get(data, "critic")
        reward  = safe_get(data, "reward")
        if traj_id is None or critic is None or reward is None:
            print("[skip] advantage_sign: need 'traj_id', 'critic', 'reward'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] advantage_sign: missing hook '{hk}'")
                    continue
                acc, f1 = run_advantage_sign(H, reward, critic, traj_id, gamma=0.99, seed=args.seed)
                results.append(("advantage_sign", hk, {"acc": acc, "F1": f1}))

    # Successor propositions probe: predict discounted future AP counts
    if "successor_props" in args.task:
        traj_id = safe_get(data, "traj_id")
        ap      = safe_get(data, "ap")
        if traj_id is None or ap is None:
            print("[skip] successor_props: need 'traj_id', 'ap'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] successor_props: missing hook '{hk}'")
                    continue
                r2 = run_successor_probe(H, ap, traj_id, gamma=0.99, K=50, seed=args.seed)
                results.append(("successor_props", hk, {"R2_successor": r2}))

    # Modified successor probe: predict discounted future distances to goal
    if "successor_props_modified" in args.task:
        traj_id = safe_get(data, "traj_id")
        vec_to_next_pos = safe_get(data, "vec_to_next_pos")
        if traj_id is None or vec_to_next_pos is None:
            print("[skip] successor_props_modified: need 'traj_id', 'vec_to_next_pos'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] successor_props_modified: missing hook '{hk}'")
                    continue
                r2 = run_successor_probe_modified(H, vec_to_next_pos, traj_id, gamma=0.99, K=20, seed=args.seed)
                results.append(("successor_props_modified", hk, {"R2_successor_dist": r2}))

    # Time-to-hit probe: predict steps until success within current link
    if "tth" in args.task:
        traj_id = safe_get(data, "traj_id")
        link_idx = safe_get(data, "link_idx")
        success_step = safe_get(data, "success_step")
        if traj_id is None or link_idx is None or success_step is None:
            print("[skip] tth: need 'traj_id', 'link_idx', 'success_step'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] tth: missing hook '{hk}'")
                    continue
                r2 = run_tth_probe(H, traj_id, link_idx, success_step, seed=args.seed)
                results.append(("tth", hk, {"R2_TTH": r2}))

    # Modified TTH probe: predict distance progression within current link
    if "tth_modified" in args.task:
        traj_id = safe_get(data, "traj_id")
        link_idx = safe_get(data, "link_idx")
        vec_to_next_pos = safe_get(data, "vec_to_next_pos")
        if traj_id is None or link_idx is None or vec_to_next_pos is None:
            print("[skip] tth_modified: need 'traj_id', 'link_idx', 'vec_to_next_pos'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] tth_modified: missing hook '{hk}'")
                    continue
                r2 = run_tth_probe_modified(H, vec_to_next_pos, traj_id, link_idx, seed=args.seed)
                results.append(("tth_modified", hk, {"R2_TTH_dist": r2}))

    # Success within H steps probe: binary classification
    if "success_H" in args.task:
        traj_id = safe_get(data, "traj_id")
        link_idx = safe_get(data, "link_idx")
        success_step = safe_get(data, "success_step")
        if traj_id is None or link_idx is None or success_step is None:
            print("[skip] success_H: need 'traj_id', 'link_idx', 'success_step'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] success_H: missing hook '{hk}'")
                    continue
                # Test multiple horizons
                for H_horizon in [5, 10, 20]:
                    acc, f1 = run_success_H(H, traj_id, link_idx, success_step, H=H_horizon, seed=args.seed)
                    results.append(("success_H", hk, {"H": H_horizon, "acc": acc, "F1": f1}))

    # Modified success_H probe: predict if we'll get close to goal within H steps
    if "success_H_modified" in args.task:
        traj_id = safe_get(data, "traj_id")
        link_idx = safe_get(data, "link_idx")
        vec_to_next_pos = safe_get(data, "vec_to_next_pos")
        if traj_id is None or link_idx is None or vec_to_next_pos is None:
            print("[skip] success_H_modified: need 'traj_id', 'link_idx', 'vec_to_next_pos'")
        else:
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] success_H_modified: missing hook '{hk}'")
                    continue
                # Test multiple horizons
                for H_horizon in [5, 10, 20]:
                    acc, f1 = run_success_H_modified(H, vec_to_next_pos, traj_id, link_idx, H=H_horizon, seed=args.seed)
                    results.append(("success_H_modified", hk, {"H": H_horizon, "acc": acc, "F1": f1}))

    # Incremental R² probe: how much does hook add beyond baseline?
    if "delta_r2" in args.task:
        obs = safe_get(data, "obs")
        action = safe_get(data, KEYMAP["action"])
        next_obs = safe_get(data, "next_obs")
        critic = safe_get(data, "critic")
        if obs is None or action is None or next_obs is None or critic is None:
            print("[skip] delta_r2: need 'obs', 'action', 'next_obs', 'critic'")
        else:
            # Baseline: obs + action
            base_X = np.concatenate([obs, action], axis=1)
            
            # Test transition prediction
            for hk in args.hooks:
                H = safe_get(data, hk)
                if H is None:
                    print(f"[skip] delta_r2 transition: missing hook '{hk}'")
                    continue
                r2_be, delta = delta_r2(base_X, H, next_obs, seed=args.seed)
                results.append(("delta_r2_transition", hk, {"R2_total": r2_be, "R2_delta": delta}))
            
            # Test value_next prediction
            traj_id = safe_get(data, "traj_id")
            if traj_id is not None:
                # Get next values for consecutive pairs
                def pairs(x):
                    return x[:-1], x[1:], traj_id[:-1] == traj_id[1:]
                _, V_tp1, m_v = pairs(critic)
                if m_v.sum() >= 50:
                    for hk in args.hooks:
                        H = safe_get(data, hk)
                        if H is None:
                            print(f"[skip] delta_r2_value_next: missing hook '{hk}'")
                            continue
                        H_t, _, m_h = pairs(H)
                        A_t, _, m_a = pairs(action)
                        m = m_v & m_h & m_a
                        if m.sum() >= 50:
                            base_X_pairs = np.concatenate([obs[:-1][m], A_t[m]], axis=1)
                            r2_be, delta = delta_r2(base_X_pairs, H_t[m], V_tp1[m], seed=args.seed)
                            results.append(("delta_r2_value_next", hk, {"R2_total": r2_be, "R2_delta": delta}))

    # Print compact table
    if not results:
        print("No results produced. Check your keys and tasks.")
        sys.exit(0)

    # Simple text table
    col1_w = max(len(r[0]) for r in results) + 2
    col2_w = max(len(r[1]) for r in results) + 2
    print("\n=== Probe Results ===")
    print(f"{'task'.ljust(col1_w)}{'hook'.ljust(col2_w)}metrics")
    for task, hook, metrics in results:
        met_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) and np.isfinite(v) else f"{k}={v}" for k, v in metrics.items()])
        print(f"{task.ljust(col1_w)}{hook.ljust(col2_w)}{met_str}")

if __name__ == "__main__":
    main()
