#!/usr/bin/env python3
"""
Build and visualize letter manifolds from sweep NPZs.

Outputs:
- grid_letter_pca.png (+ centroids)
- goal_letter_pca.png (+ centroids and Reach↔Avoid connectors)
- prototypes.npz (mu_grid[letter], mu_goal[(role,letter)])
- alignment_eval.json (Procrustes, per-letter cosine)
- role_offset_eval.json (cosine improvements after subtracting Δ_role)

Assumes NPZ contains per-step arrays from 03e_log_rollouts_letter_sweep.py:
- 'letter_id' (int, -1 for non-letter), 'agent_pos' (T,2)
- 'goal_mode' (object), 'goal_reach', 'goal_reach2', 'goal_avoid' (int or -1)
- hook arrays such as 'hook_env_mlp3', 'actor_mid', 'hook_ltl_rnn_h'
"""

import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def pca_project(X: np.ndarray, n=2, whiten=False) -> np.ndarray:
    if X.shape[0] < 2:
        return np.c_[np.arange(X.shape[0]), np.zeros(X.shape[0])]
    Xs = StandardScaler(with_mean=True, with_std=whiten).fit_transform(X)
    pcs = PCA(n_components=n, random_state=0).fit_transform(Xs)
    return pcs


def to_letters(indices: np.ndarray, vocab: List[str]) -> np.ndarray:
    out = []
    for idx in indices:
        if idx is None or idx < 0:
            out.append(None)
        else:
            out.append(str(vocab[int(idx)]).upper())
    return np.array(out, dtype=object)


def find_first_two_letters(letter_ids: np.ndarray) -> Tuple[int, int]:
    """Return indices tA, tB over a sequence of letter_ids (-1 for none)."""
    tA = -1
    first = None
    for i, v in enumerate(letter_ids):
        if int(v) != -1:
            tA = i
            first = int(v)
            break
    if tA < 0:
        return -1, -1
    tB = -1
    for j in range(tA + 1, len(letter_ids)):
        v = int(letter_ids[j])
        if v != -1 and v != first:
            tB = j
            break
    return tA, tB


def active_goal_labels_per_step(goal_mode: np.ndarray, reach: np.ndarray, reach2: np.ndarray, avoid: np.ndarray,
                                letters_vocab: List[str], epi: np.ndarray, letter_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build stepwise labels for goal roles (Reach/Avoid) + optional phase for reach2.

    Returns:
      - labels: array of strings like 'Reach:A' or 'Avoid:C'
      - phases: array of strings in {'pre','between','post',None}
    """
    T = len(goal_mode)
    labels = []
    phases = [None] * T
    # Precompute per-episode tA/tB for phases
    # episode ids can be arbitrary; compute mapping indices->mask
    ep_unique = np.array(sorted(set(map(int, epi.tolist())))) if epi.dtype != object else np.unique(epi)
    ep_to_phase = {}
    for e in ep_unique:
        mask = (epi == e)
        tA, tB = find_first_two_letters(letter_id[mask])
        ep_to_phase[e] = (tA, tB, np.where(mask)[0][0])  # store start index to adjust

    for t in range(T):
        mode = str(goal_mode[t])
        lab_t = []
        # roles
        r = int(reach[t]) if reach is not None else -1
        r2 = int(reach2[t]) if reach2 is not None else -1
        av = int(avoid[t]) if avoid is not None else -1
        if mode in ("reach", "reach_avoid") and r >= 0:
            lab_t.append(f"Reach:{letters_vocab[r].upper()}")
        if mode in ("reach2", "reach2_avoid"):
            if r >= 0:
                lab_t.append(f"Reach:{letters_vocab[r].upper()}")
            if r2 >= 0:
                lab_t.append(f"Reach:{letters_vocab[r2].upper()}")
        if mode in ("avoid", "reach_avoid", "reach2_avoid") and av >= 0:
            lab_t.append(f"Avoid:{letters_vocab[av].upper()}")
        if not lab_t:
            lab_t = [None]
        # phase for reach2 flavors
        if "reach2" in mode:
            e = int(epi[t])
            tA, tB, start_idx = ep_to_phase.get(e, (-1, -1, 0))
            # local time within episode
            t_local = t - start_idx
            ph = None
            if tA >= 0 and t_local < tA:
                ph = 'pre'
            elif tB >= 0 and t_local < tB:
                ph = 'between'
            elif tB >= 0 and t_local >= tB:
                ph = 'post'
            phases[t] = ph
        labels.append(lab_t)
    # For goal dataset, we will expand multi-label steps into multiple examples (one per role)
    return np.array(labels, dtype=object), np.array(phases, dtype=object)


def balance_by_class(idxs: np.ndarray, classes: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kept = []
    for cls in sorted(set([c for c in classes if c is not None])):
        m = (classes == cls)
        ids = idxs[m]
        if len(ids) > cap:
            ids = rng.choice(ids, size=cap, replace=False)
        kept.append(ids)
    return np.concatenate(kept) if kept else np.array([], dtype=int)


def residualize_position(X: np.ndarray, pos: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Remove linear effect of position (x,y). Returns residual X - X_hat."""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(pos, X)
    return X - model.predict(pos)


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    """Find orthogonal W minimizing ||W A - B||_F. Returns (W, rms_error). A,B: [D,K]."""
    # Center columns
    A0 = A - A.mean(axis=1, keepdims=True)
    B0 = B - B.mean(axis=1, keepdims=True)
    M = B0 @ A0.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    diff = W @ A0 - B0
    rms = np.sqrt(np.mean(diff ** 2))
    return W, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=str, default='interpretability/letter_world/datasets/letter_sweep_all_layers_e1_s10.npz')
    ap.add_argument('--out_dir', type=str, default='interpretability/letter_world/results/letter_manifolds')
    ap.add_argument('--grid_hook', type=str, default=None, help="Override: e.g., hook_env_mlp3 or actor_mid")
    ap.add_argument('--goal_hook', type=str, default=None, help="Override: e.g., hook_ltl_rnn_h or hook_ltl_gru_h")
    ap.add_argument('--cap_grid', type=int, default=400)
    ap.add_argument('--cap_goal', type=int, default=400)
    ap.add_argument('--residualize_pos', action='store_true')
    ap.add_argument('--residualize_action', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--whiten', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)

    # Select hooks
    def pick_hook(candidates: List[str]) -> str:
        for k in candidates:
            if k in d.files:
                return k
        return None

    grid_hook = args.grid_hook or pick_hook(['hook_env_mlp3', 'actor_mid', 'feature_t'])
    goal_hook = args.goal_hook or pick_hook(['hook_ltl_rnn_h', 'hook_ltl_gru_h', 'hook_ltl.rnn'])
    if grid_hook is None or goal_hook is None:
        raise KeyError(f"Could not find hooks. grid='{grid_hook}', goal='{goal_hook}'. Available: {d.files}")

    # Align lengths conservatively
    T = min(*(len(d[k]) for k in [grid_hook, goal_hook, 'letter_id', 'agent_pos', 'goal_mode', 'goal_reach', 'goal_reach2', 'goal_avoid']))

    Xgrid_all = ensure_2d(d[grid_hook])[:T]
    Xgoal_all = ensure_2d(d[goal_hook])[:T]
    letters_vocab = list(d['letters_vocab']) if 'letters_vocab' in d.files else [chr(ord('a') + i) for i in range(12)]
    letter_now = np.asarray(d['letter_id'])[:T]
    ep = np.asarray(d['episode'])[:T]
    pos = np.asarray(d['agent_pos'])[:T]
    goal_mode = np.asarray(d['goal_mode'])[:T]
    reach = np.asarray(d['goal_reach'])[:T]
    reach2 = np.asarray(d['goal_reach2'])[:T]
    avoid = np.asarray(d['goal_avoid'])[:T]

    # --- Dataset A: Grid-letter ---
    mask_grid = letter_now != -1
    idxs = np.where(mask_grid)[0]
    letters_grid = to_letters(letter_now[idxs], letters_vocab)
    Xg = Xgrid_all[idxs]
    # Optional residualization for confounds
    if args.residualize_pos or args.residualize_action:
        D_list = []
        if args.residualize_pos:
            D_list.append(pos[idxs].astype(np.float32))
        if args.residualize_action and ('action' in d.files):
            act = np.asarray(d['action'])[:T][idxs]
            # one-hot
            num_a = int(act.max()) + 1 if act.size > 0 else 0
            A = np.zeros((len(act), num_a), dtype=np.float32)
            for i, a in enumerate(act):
                A[i, int(a)] = 1.0
            D_list.append(A)
        if D_list:
            D = np.concatenate(D_list, axis=1)
            Xg = Xg - Ridge(alpha=1.0, fit_intercept=True).fit(D, Xg).predict(D)
    # Balance per letter
    kept = balance_by_class(idxs=np.arange(len(idxs)), classes=letters_grid, cap=args.cap_grid, seed=args.seed)
    idxs_grid = idxs[kept]
    X_grid = Xgrid_all[idxs_grid]
    y_grid = letters_grid[kept]

    # Prototypes grid
    mu_grid: Dict[str, np.ndarray] = {}
    for L in sorted(set(y_grid)):
        mu_grid[L] = X_grid[y_grid == L].mean(axis=0)

    # PCA grid plot
    XYg = pca_project(X_grid, n=2, whiten=args.whiten)
    plt.figure(figsize=(7, 6), dpi=150)
    uniq = list(dict.fromkeys(y_grid))
    for L in uniq:
        m = (y_grid == L)
        plt.scatter(XYg[m, 0], XYg[m, 1], s=8, alpha=0.6, label=L)
    # centroids in PCA space
    for L in uniq:
        m = (y_grid == L)
        cx, cy = XYg[m, :].mean(axis=0)
        plt.scatter([cx], [cy], s=80, marker='X', edgecolors='k')
        plt.text(cx, cy, f" {L}", fontsize=8, weight='bold')
    plt.title(f"Grid letter manifold ({grid_hook})")
    plt.legend(fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'grid_letter_pca.png'))
    plt.close()

    # --- Dataset B: Goal-letter ---
    step_labels, phases = active_goal_labels_per_step(goal_mode, reach, reach2, avoid, letters_vocab, ep, letter_now)
    # Expand multi-role steps into (idx, role_label)
    pairs = []
    for t, labs in enumerate(step_labels):
        for lab in (labs if isinstance(labs, (list, tuple, np.ndarray)) else [labs]):
            if lab is None:
                continue
            pairs.append((t, lab))
    if not pairs:
        raise RuntimeError("No goal labels could be constructed; check modes/labels in NPZ.")
    idxs_goal = np.array([p[0] for p in pairs], dtype=int)
    y_goal = np.array([p[1] for p in pairs], dtype=object)
    X_goal = Xgoal_all[idxs_goal]
    # Balance classes
    kept_g = balance_by_class(np.arange(len(y_goal)), y_goal, cap=args.cap_goal, seed=args.seed)
    X_goal = X_goal[kept_g]
    y_goal = y_goal[kept_g]

    # Prototypes goal
    mu_goal: Dict[Tuple[str, str], np.ndarray] = {}
    roles = []
    letters_goal = []
    for lab in y_goal:
        role, letter = str(lab).split(':', 1)
        roles.append(role)
        letters_goal.append(letter)
    roles = np.array(roles, dtype=object)
    letters_goal = np.array(letters_goal, dtype=object)
    rl_set = sorted(set(y_goal))
    for rl in rl_set:
        role, letter = rl.split(':', 1)
        mu_goal[(role, letter)] = X_goal[y_goal == rl].mean(axis=0)

    # PCA goal plot with centroids and Reach<->Avoid connectors
    XYh = pca_project(X_goal, n=2, whiten=args.whiten)
    plt.figure(figsize=(7, 6), dpi=150)
    uniqgol = list(dict.fromkeys(y_goal))
    for c in uniqgol:
        m = (y_goal == c)
        plt.scatter(XYh[m, 0], XYh[m, 1], s=8, alpha=0.6, label=c)
    # centroids and connectors
    cent: Dict[Tuple[str, str], np.ndarray] = {}
    for rl in uniqgol:
        m = (y_goal == rl)
        cx, cy = XYh[m, :].mean(axis=0)
        role, letter = rl.split(':', 1)
        cent[(role, letter)] = np.array([cx, cy])
        plt.scatter([cx], [cy], s=80, marker='X', edgecolors='k')
        plt.text(cx, cy, f" {rl}", fontsize=7)
    # connectors per letter if both roles exist
    letters_u = sorted(set([L for _, L in cent.keys()]))
    for L in letters_u:
        a = cent.get(('Reach', L))
        b = cent.get(('Avoid', L))
        if a is not None and b is not None:
            plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.4)
    plt.title(f"Goal letter manifold ({goal_hook})")
    plt.legend(fontsize=7, frameon=False, ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'goal_letter_pca.png'))
    plt.close()

    # Role-neutral replot: subtract global role offset Δ from Reach samples, re-PCA
    # Compute Δ from per-class means
    mu_goal_reach = {L: mu_goal.get(('Reach', L)) for L in letters_vocab if ('Reach', L) in mu_goal}
    mu_goal_avoid = {L: mu_goal.get(('Avoid', L)) for L in letters_vocab if ('Avoid', L) in mu_goal}
    letters_r = sorted(set(mu_goal_reach.keys()) & set(mu_goal_avoid.keys()))
    if letters_r:
        deltas = [mu_goal_reach[L] - mu_goal_avoid[L] for L in letters_r]
        d_role = np.mean(np.stack(deltas, axis=0), axis=0)
        # transform X_goal for Reach samples
        X_goal_neu = X_goal.copy()
        reach_mask = np.array([str(l).startswith('Reach:') for l in y_goal])
        X_goal_neu[reach_mask] = X_goal_neu[reach_mask] - d_role
        XYh2 = pca_project(X_goal_neu, n=2, whiten=args.whiten)
        plt.figure(figsize=(7, 6), dpi=150)
        uniqgol = list(dict.fromkeys(y_goal))
        for c in uniqgol:
            m = (y_goal == c)
            plt.scatter(XYh2[m, 0], XYh2[m, 1], s=8, alpha=0.6, label=c)
        # centroids
        cent2 = {}
        for rl in uniqgol:
            m = (y_goal == rl)
            cx, cy = XYh2[m, :].mean(axis=0)
            role, letter = rl.split(':', 1)
            cent2[(role, letter)] = np.array([cx, cy])
            plt.scatter([cx], [cy], s=80, marker='X', edgecolors='k')
            plt.text(cx, cy, f" {rl}", fontsize=7)
        for L in sorted(set([L for (_, L) in cent2.keys()])):
            a = cent2.get(('Reach', L)); b = cent2.get(('Avoid', L))
            if a is not None and b is not None:
                plt.plot([a[0], b[0]], [a[1], b[1]], 'k--', alpha=0.4)
        plt.title(f"Goal letter manifold (role-neutral)")
        plt.legend(fontsize=7, frameon=False, ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'goal_letter_pca_neutral.png'))
        plt.close()

    # --- Alignment analyses ---
    # Prepare per-letter matrices where both prototypes exist
    letters_inter = sorted(set(mu_grid.keys()) & set([L for (_, L) in mu_goal.keys() if _ == 'Reach']))
    if letters_inter:
        A = np.stack([mu_goal[('Reach', L)] for L in letters_inter], axis=1)  # [D,K]
        B = np.stack([mu_grid[L] for L in letters_inter], axis=1)             # [D,K]
        W, rms = orthogonal_procrustes(A, B)
        # Cosines after alignment (map goal -> grid space) and symmetric (grid -> goal space)
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        cos_goal_to_grid = {L: cos((W @ mu_goal[('Reach', L)]), mu_grid[L]) for L in letters_inter}
        cos_grid_to_goal = {L: cos(mu_goal[('Reach', L)], (W.T @ mu_grid[L])) for L in letters_inter}
        with open(os.path.join(args.out_dir, 'alignment_eval.json'), 'w') as f:
            json.dump({
                'letters': letters_inter,
                'rms_error': float(rms),
                'cos_goal_to_grid': cos_goal_to_grid,
                'cos_grid_to_goal': cos_grid_to_goal
            }, f, indent=2)

        # Neutral alignment: goal Reach prototypes minus Δ vs grid prototypes
        if letters_r:
            A_neu = np.stack([mu_goal[('Reach', L)] - d_role for L in letters_inter], axis=1)
            Wn, rmsn = orthogonal_procrustes(A_neu, B)
            cos_goal_to_grid_n = {L: cos((Wn @ (mu_goal[('Reach', L)] - d_role)), mu_grid[L]) for L in letters_inter}
            with open(os.path.join(args.out_dir, 'alignment_neutral_eval.json'), 'w') as f:
                json.dump({
                    'letters': letters_inter,
                    'rms_error': float(rmsn),
                    'cos_goal_to_grid_neutral': cos_goal_to_grid_n
                }, f, indent=2)

    # Role offset
    letters_role = sorted(set([L for (_, L) in mu_goal.keys() if ('Reach', L) in mu_goal and ('Avoid', L) in mu_goal]))
    if letters_role:
        deltas = [mu_goal[('Reach', L)] - mu_goal[('Avoid', L)] for L in letters_role]
        d_role = np.mean(np.stack(deltas, axis=0), axis=0)
        improve = {}
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        for L in letters_role:
            a = mu_goal[('Reach', L)] - d_role
            b = mu_goal[('Avoid', L)]
            improve[L] = cos(a, b)
        with open(os.path.join(args.out_dir, 'role_offset_eval.json'), 'w') as f:
            json.dump({
                'letters': letters_role,
                'avg_delta_norm': float(np.linalg.norm(d_role)),
                'cos_reach_minus_delta_vs_avoid': improve
            }, f, indent=2)

    # Save prototypes
    # Convert dicts to arrays with stable ordering
    letters_order = sorted(mu_grid.keys())
    roles_letters_order = sorted([f"{r}:{L}" for (r, L) in mu_goal.keys()])
    mu_grid_arr = np.stack([mu_grid[L] for L in letters_order], axis=0) if letters_order else np.zeros((0, X_grid.shape[1]))
    mu_goal_arr = np.stack([mu_goal[(r, L)] for (r, L) in [tuple(x.split(':', 1)) for x in roles_letters_order]], axis=0) if roles_letters_order else np.zeros((0, X_goal.shape[1]))
    np.savez_compressed(os.path.join(args.out_dir, 'prototypes.npz'),
                        mu_grid=mu_grid_arr, letters=letters_order,
                        mu_goal=mu_goal_arr, role_letter=roles_letters_order,
                        grid_hook=grid_hook, goal_hook=goal_hook)

    print(f"[done] Wrote results to {args.out_dir}")


if __name__ == '__main__':
    main()
