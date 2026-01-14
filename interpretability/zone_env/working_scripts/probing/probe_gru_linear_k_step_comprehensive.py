#!/usr/bin/env python3
"""
Comprehensive GRU k-step transition probe (group split, CV, open-loop)

What it does
------------
• Collect per-world sequences of (h_t, a_t) from a trained model (GRU last-layer hidden via hook)
• Build aligned pairs for multiple horizons k (default: 1,5,10,20)
• Split data by world or randomly (train/val/test); choose ridge α on val only
• Predict either Δh_k = h_{t+k} − h_t (default) or h_{t+k} (with --predict_hprime)
• Report baselines, R²(h-only), R²([h,a]), ΔR², partial R² via Frisch–Waugh–Lovell (actions | hidden)
• Fit W_k in h_{t+k} ≈ W_k h_t; compare to (I + W_1)^k (Frobenius distance & spectral radii)
• Evaluate teacher-forcing and open-loop rollouts using learned 1-step [h,a]→Δh
• Log variance of Δh_k vs k, per-unit stats, avg |corr(h_t, h_{t+k})|, and a tiny h→a probe

Usage (examples)
----------------
python probe_gru_linear_k_step.py --deterministic --num_loops 2 --n_worlds 20 --max_step 400 \
  --k_list 1,5,10,20 --split world

python probe_gru_linear_k_step.py --deterministic --num_loops 2 --n_worlds 20 --max_step 400 \
  --k_list 1,10,20 --split random --predict_hprime
"""

import argparse, random, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ─────────────────── repo imports ────────────────────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
# ──────────────────────────────────────────────────────

# ----------------- small globals ---------------------
ENV, EXP, SEED = "PointLtl2-v0", "big_test", 0
GOALS          = [f"FG {c}" for c in ["blue", "green", "yellow", "magenta"]]
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
# -----------------------------------------------------


# ---------- small helpers ----------
def group_split_indices(groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split by distinct group labels (worlds). Returns (train_idx, test_idx)."""
    uniq = np.unique(groups)
    n_groups = len(uniq)
    if n_groups < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_test = max(1, min(int(round(n_groups * test_frac)), n_groups - 1))
    rng_local = np.random.default_rng(seed)
    test_groups = rng_local.choice(uniq, size=n_test, replace=False)
    te_mask = np.isin(groups, test_groups)
    tr_mask = ~te_mask
    return np.where(tr_mask)[0], np.where(te_mask)[0]


def train_val_split(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simple random split indices for (train_sub, val) within train."""
    n_val = max(1, int(round(n * val_frac)))
    perm = np.random.default_rng(seed).permutation(n)
    return perm[n_val:], perm[:n_val]


def ridge_cv_fit(X_tr: np.ndarray, y_tr: np.ndarray, alphas: List[float], val_frac: float, seed: int):
    """Choose alpha on a val split carved from train; refit on full train; return (model, chosen_alpha, val_r2)."""
    idx_sub, idx_val = train_val_split(len(X_tr), val_frac, seed)
    X_sub, y_sub = X_tr[idx_sub], y_tr[idx_sub]
    X_val, y_val = X_tr[idx_val], y_tr[idx_val]

    best = None; best_alpha = None; best_r2 = -np.inf
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=False, solver="svd")
        reg.fit(X_sub, y_sub)
        r2 = reg.score(X_val, y_val)
        if r2 > best_r2:
            best_r2, best_alpha, best = r2, a, reg

    # refit on all train with chosen alpha
    reg = Ridge(alpha=best_alpha, fit_intercept=False, solver="svd").fit(X_tr, y_tr)
    return reg, best_alpha, best_r2


def safe_unitwise_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    vals = []
    for i in range(y_true.shape[1]):
        try:
            vals.append(r2_score(y_true[:, i], y_pred[:, i]))
        except Exception:
            vals.append(np.nan)
    return np.asarray(vals)


def spectral_radius(M: np.ndarray) -> float:
    try:
        ev = np.linalg.eigvals(M)
        return float(np.max(np.abs(ev)))
    except Exception:
        return float("nan")


def fwl_partial_r2(X: np.ndarray, A: np.ndarray, y: np.ndarray, alphas: List[float], seed: int, val_frac: float):
    """
    Frisch–Waugh–Lovell: R²(y ~ [X,A]) - R²(y ~ X)
    Implemented as: regress y on X (ridge) => residual r_y; regress r_y on A (ridge), report R² on test residuals.
    Returns model on [X,A] chosen alpha for A stage and partial R² on test (computed later by caller).
    """
    # Split will be handled by caller; this function just returns a closure for residualization
    def fit_on(train_idx, test_idx):
        Xtr, Atr, ytr = X[train_idx], A[train_idx], y[train_idx]
        Xte, Ate, yte = X[test_idx], A[test_idx], y[test_idx]

        # standardize X and A separately on train
        scX = StandardScaler().fit(Xtr)
        scA = StandardScaler().fit(Atr)

        Xtr_s, Xte_s = scX.transform(Xtr), scX.transform(Xte)
        Atr_s, Ate_s = scA.transform(Atr), scA.transform(Ate)

        # Stage 1: y ~ X
        regX, alphaX, _ = ridge_cv_fit(Xtr_s, ytr, alphas, val_frac, seed+1)
        ry_tr = ytr - regX.predict(Xtr_s)
        ry_te = yte - regX.predict(Xte_s)

        # Stage 2: r_y ~ A
        regA, alphaA, _ = ridge_cv_fit(Atr_s, ry_tr, alphas, val_frac, seed+2)
        r2_partial = regA.score(Ate_s, ry_te)
        return r2_partial, (alphaX, alphaA)

    return fit_on


def build_pairs_k(H_seq: Dict[int, List[np.ndarray]],
                  A_seq: Dict[int, List[np.ndarray]],
                  k: int,
                  predict_hprime: bool):
    """
    Build arrays H_t, A_t, H_tk (or Δh_k) and group labels G, ensuring pairs stay within the same world.
    """
    H_list, A_list, Y_list, G_list = [], [], [], []
    for w in sorted(H_seq.keys()):
        H_w = np.asarray(H_seq[w], dtype=np.float32)  # shape (T_w, hidden)
        A_w = np.asarray(A_seq[w], dtype=np.float32)  # shape (T_w, act)
        T_w = min(len(A_w), len(H_w))  # both recorded once per decision

        if T_w - k <= 0:
            continue

        H_t   = H_w[:T_w - k]
        H_tk  = H_w[k:T_w]
        A_t   = A_w[:T_w - k]

        if predict_hprime:
            Y = H_tk
        else:
            Y = H_tk - H_t

        H_list.append(H_t)
        A_list.append(A_t)
        Y_list.append(Y)
        G_list.append(np.full((len(H_t),), w, dtype=np.int32))

    if not H_list:
        return None

    H = np.vstack(H_list)
    A = np.vstack(A_list)
    Y = np.vstack(Y_list)
    G = np.concatenate(G_list)
    return H, A, Y, G


def eval_rollouts_open_loop(H_seq, A_seq, model_1step, sc_ha, max_k: int, test_worlds: List[int]):
    """
    Evaluate teacher-forcing and open-loop rollouts using learned 1-step Δh model:
        Δh = [H, A] W  (Ridge with SVD, fitted on train)
    h_{t+1} = h_t + Δh_pred
    Returns dict with avg R² over horizons 1..max_k.
    """
    Mh = model_1step.coef_[:, :model_1step.coef_.shape[1] - A_seq[next(iter(A_seq))][0].shape[0]]  # weights on H
    Ma = model_1step.coef_[:, -A_seq[next(iter(A_seq))][0].shape[0]:]                              # weights on A
    # NOTE: using reg.coef_ for multioutput; shape (hidden, D_in). Ridge in sklearn stores coef_ as (n_targets, n_features)

    def one_step(h, a):
        x = np.concatenate([h[None, :], a[None, :]], axis=1)
        x = sc_ha.transform(x)
        dh = model_1step.predict(x)[0]
        return h + dh

    # collect starting points from test worlds
    all_r2_open = {k: [] for k in range(1, max_k + 1)}
    all_r2_tf   = {k: [] for k in range(1, max_k + 1)}

    for w in test_worlds:
        H_w = np.asarray(H_seq[w], dtype=np.float32)
        A_w = np.asarray(A_seq[w], dtype=np.float32)
        T_w = min(len(H_w), len(A_w))
        for t in range(0, T_w - max_k):
            h_true = H_w[t]
            # open loop
            h_pred = h_true.copy()
            for k in range(1, max_k + 1):
                h_pred = one_step(h_pred, A_w[t + k - 1])
                r2 = r2_score(H_w[t + k], h_pred)
                all_r2_open[k].append(r2)
            # teacher forcing
            h_tf = h_true.copy()
            for k in range(1, max_k + 1):
                h_tf = one_step(H_w[t + k - 1], A_w[t + k - 1])  # use true h at each step
                r2 = r2_score(H_w[t + k], h_tf)
                all_r2_tf[k].append(r2)

    def agg(d):
        return {k: (float(np.mean(v)) if len(v) > 0 else float("nan")) for k, v in d.items()}

    return {"open_loop": agg(all_r2_open), "teacher_forcing": agg(all_r2_tf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--n_worlds", type=int, default=20)
    ap.add_argument("--max_step", type=int, default=400)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--split", choices=["world", "random"], default="world")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--alphas", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--k_list", type=str, default="1,5,10,20")
    ap.add_argument("--predict_hprime", action="store_true", help="predict h_{t+k} instead of Δh_k")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    # ── build env/model ─────────────────────────────
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    hidden_sz = model.ltl_net.rnn.hidden_size
    print("Observation shape:", dummy.observation_space.shape)
    print("Num LTLNet params:", sum(p.numel() for p in model.parameters()))
    dummy.close()

    # ── hooks: capture hidden after each policy forward, per-world ──
    H_seq: Dict[int, List[np.ndarray]] = {w: [] for w in range(args.n_worlds)}
    current_world_id = -1

    def rnn_hook(_, __, out):
        h = out[1][-1].detach().cpu().numpy().squeeze().astype(np.float32)
        H_seq[current_world_id].append(h)

    handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    # ── gather sequences ────────────────────────────
    A_seq: Dict[int, List[np.ndarray]] = {w: [] for w in range(args.n_worlds)}
    action_dim = None

    for wid in range(args.n_worlds):
        current_world_id = wid
        goal     = GOALS[wid % len(GOALS)]
        env      = make_env(ENV, FixedSampler.partial(goal), sequence=False)
        props    = set(env.get_propositions())
        planner  = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent    = Agent(model, planner, propositions=props)
        obs      = env.reset(seed=SEED + 100 * wid)
        agent.reset()

        for _ in range(args.max_step):
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=args.deterministic)
            a = act.flatten().astype(np.float32)
            if action_dim is None:
                action_dim = a.shape[0]
            A_seq[wid].append(a)
            obs, *_ = env.step(a)

        print(f"world {wid}: actions={len(A_seq[wid])}, h_samples={len(H_seq[wid])}")
        env.close()

    handle.remove()

    print(f"GRU hidden        : {hidden_sz}")
    print(f"Action dim        : {action_dim}")

    # ── per-k analysis ──────────────────────────────
    max_k = max(k_list)

    # Pre-train a 1-step model on train for rollouts later (we will fit it after splits)
    # Build k=1 data (Δh) for fitting the rollout model
    data_k1 = build_pairs_k(H_seq, A_seq, k=1, predict_hprime=False)
    if data_k1 is None:
        raise RuntimeError("No pairs available; reduce k or increase rollouts.")
    H1, A1, dH1, G1 = data_k1

    # Split indices
    if args.split == "world":
        tr_idx, te_idx = group_split_indices(G1, args.test_frac, SEED)
        if tr_idx.size == 0 or te_idx.size == 0:
            # fallback
            n = len(H1); perm = rng.permutation(n); n_test = max(1, int(round(n * args.test_frac)))
            te_idx = perm[:n_test]; tr_idx = perm[n_test:]
    else:
        n = len(H1); perm = rng.permutation(n); n_test = max(1, int(round(n * args.test_frac)))
        te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    # Standardize [H,A] on train for k=1 rollout model
    X1_ha = np.hstack([H1, A1])
    sc_ha_roll = StandardScaler().fit(X1_ha[tr_idx])

    # Fit 1-step Δh model ([H,A]→Δh) with CV
    reg1, alpha1, _ = ridge_cv_fit(sc_ha_roll.transform(X1_ha[tr_idx]), dH1[tr_idx], alphas, args.val_frac, SEED+7)

    # Identify test worlds (for open-loop rollouts) if split by world; else use all worlds
    if args.split == "world":
        test_worlds = list(np.unique(G1[te_idx]))
    else:
        test_worlds = list(H_seq.keys())

    # Open-loop / teacher-forcing rollouts up to max_k
    rollout_scores = eval_rollouts_open_loop(H_seq, A_seq, reg1, sc_ha_roll, max_k=max_k, test_worlds=test_worlds)

        # ---- choose a fixed test-world set once (used for all k) ----
    SPLIT_SEED = SEED + 101
    FIXED_TEST_WORLDS = np.array([], dtype=int)  # default: empty -> falls back to random split

    if args.split == "world":
        init = build_pairs_k(H_seq, A_seq, k=1, predict_hprime=args.predict_hprime)
        if init is not None:
            _, _, _, G1 = init           # world IDs for k=1 pairs
            uniq_worlds = np.unique(G1)
            if uniq_worlds.size >= 2:
                rng_fixed = np.random.default_rng(SPLIT_SEED)
                n_test = max(1, int(round(len(uniq_worlds) * args.test_frac)))
                FIXED_TEST_WORLDS = rng_fixed.choice(uniq_worlds, size=n_test, replace=False)
                FIXED_TEST_WORLDS = np.array(sorted(FIXED_TEST_WORLDS), dtype=int)

    print("Test worlds fixed for all k:", FIXED_TEST_WORLDS.tolist())
    # Loop over k values for full report
    for k in k_list:
        print("\n" + "=" * 90)
        print(f"k = {k}")
        print("=" * 90)

        data = build_pairs_k(H_seq, A_seq, k=k, predict_hprime=args.predict_hprime)
        if data is None:
            print("No pairs for this k; skipping.")
            continue
        H, A, Y, G = data  # H = h_t,  Y = Δh_k  (or h_{t+k} if predict_hprime)

        # ---- B) Δh_k consistency + robust lag-k correlation -----------------
        # true h_{t+k} and target Δh_k derived from what's in Y
        Htk_all   = (Y if args.predict_hprime else H + Y)
        dH_target = (Y - H) if args.predict_hprime else Y  # should equal (Htk_all - H)

        num = np.linalg.norm((Htk_all - H) - dH_target, ord='fro')
        den = np.linalg.norm((Htk_all - H), ord='fro') + 1e-12
        rel_err = float(num / den)
        print(f"Δh_k consistency rel.err: {rel_err:.3g}")

        # avg |corr(h_t, h_{t+k})|
        with np.errstate(invalid='ignore', divide='ignore'):
            cors = []
            for i in range(H.shape[1]):
                c = np.corrcoef(H[:, i], Htk_all[:, i])[0, 1]
                if np.isfinite(c):
                    cors.append(abs(c))
            print(f"avg |corr(h_t,h_{{t+{k}}})| {np.nanmean(cors):.6f}")
        # ---------------------------------------------------------------------

        # Split indices for this k
        if args.split == "world" and FIXED_TEST_WORLDS.size > 0:
            is_test = np.isin(G, FIXED_TEST_WORLDS)
            te_idx = np.where(is_test)[0]
            tr_idx = np.where(~is_test)[0]
            if te_idx.size == 0 or tr_idx.size == 0:
                # graceful fallback to random pair-level split
                n = len(H); perm = rng.permutation(n); n_test = max(1, int(round(n * args.test_frac)))
                te_idx = perm[:n_test]; tr_idx = perm[n_test:]
        else:
            n = len(H); perm = rng.permutation(n); n_test = max(1, int(round(n * args.test_frac)))
            te_idx = perm[:n_test]; tr_idx = perm[n_test:]

        # standardize features on train only
        X_h  = H
        X_ha = np.hstack([H, A])

        sc_h  = StandardScaler().fit(X_h[tr_idx])
        sc_ha = StandardScaler().fit(X_ha[tr_idx])

        Xh_tr,  Xh_te  = sc_h.transform(X_h[tr_idx]),   sc_h.transform(X_h[te_idx])
        Xha_tr, Xha_te = sc_ha.transform(X_ha[tr_idx]), sc_ha.transform(X_ha[te_idx])
        y_tr,   y_te   = Y[tr_idx], Y[te_idx]

        # baselines
        if args.predict_hprime:
            # baselines for predicting h'
            y_true_te = Htk_all[te_idx]                    # = h_{t+k}
            yhat_copy = H[te_idx]                          # copy h_t
            yhat_mean = np.tile(np.mean(Htk_all[tr_idx], axis=0, keepdims=True),
                                (len(te_idx), 1))          # mean h' on train
            print(f"Baseline copy(h_t)    held-out R² : {r2_score(y_true_te, yhat_copy):.3f}")
            print(f"Baseline mean(train)  held-out R² : {r2_score(y_true_te, yhat_mean):.3f}")
        else:
            # baselines for Δh_k
            yhat_zero = np.zeros_like(y_te)
            yhat_mean = np.tile(np.mean(y_tr, axis=0, keepdims=True), (len(te_idx), 1))
            print(f"Baseline Δh≡0         held-out R² : {r2_score(y_te, yhat_zero):.3f}")
            print(f"Baseline mean(train)  held-out R² : {r2_score(y_te, yhat_mean):.3f}")

        # Ridge: h-only vs [h,a]
        reg_h,  alpha_h,  _ = ridge_cv_fit(Xh_tr,  y_tr, alphas, args.val_frac, SEED + k*11 + 1)
        reg_ha, alpha_ha, _ = ridge_cv_fit(Xha_tr, y_tr, alphas, args.val_frac, SEED + k*11 + 2)

        if args.predict_hprime:
            # predict h'
            y_true_te = Htk_all[te_idx]
            y_pred_h  = H[te_idx]  + reg_h.predict(Xh_te)
            y_pred_ha = H[te_idx]  + reg_ha.predict(Xha_te)
            r2_h  = r2_score(y_true_te, y_pred_h)
            r2_ha = r2_score(y_true_te, y_pred_ha)
            print(f"Linear f(h  )→h′   held-out R² : {r2_h:.3f} (α={alpha_h:g})")
            print(f"Linear f(h,a)→h′   held-out R² : {r2_ha:.3f} (α={alpha_ha:g})")
            print(f"ΔR² (add actions)                 : {r2_ha - r2_h:+.3f}")
        else:
            # predict Δh_k
            y_pred_h  = reg_h.predict(Xh_te)
            y_pred_ha = reg_ha.predict(Xha_te)
            r2_h  = r2_score(y_te, y_pred_h)
            r2_ha = r2_score(y_te, y_pred_ha)
            print(f"Linear f(h  )→Δh   held-out R² : {r2_h:.3f} (α={alpha_h:g})")
            print(f"Linear f(h,a)→Δh   held-out R² : {r2_ha:.3f} (α={alpha_ha:g})")
            print(f"ΔR² (add actions)                 : {r2_ha - r2_h:+.3f}")

            # Δh variance
            dh_std = np.std(Y, axis=0)
            print(f"Δh_k per-unit std    : mean {dh_std.mean():.4g}  median {np.median(dh_std):.4g}  "
                f"min {dh_std.min():.4g}  max {dh_std.max():.4g}")

        # tiny h_t → action probe (on the same split)
        try:
            sc_h_probe = StandardScaler().fit(H[tr_idx])
            reg_probe, alpha_probe, _ = ridge_cv_fit(sc_h_probe.transform(H[tr_idx]), A[tr_idx],
                                                    alphas, args.val_frac, SEED + k*13 + 5)
            r2_probe = reg_probe.score(sc_h_probe.transform(H[te_idx]), A[te_idx])
            print(f"h_t → action (Ridge)     held-out R² : {r2_probe:.3f}")
        except Exception:
            pass

        # Partial R² for actions given H (FWL) on Δh_k
        if not args.predict_hprime:
            fit_fwl = fwl_partial_r2(X_h, A, Y, alphas, SEED + k*17, args.val_frac)
            r2_partial, _ = fit_fwl(tr_idx, te_idx)
            print(f"Partial ΔR² (FWL: A | H)         : {r2_partial:.3f}")

        # -------- W_k analysis: h_{t+k} ≈ W_k h_t  vs  (I + W_1)^k ----------
        y_k = Htk_all  # true h_{t+k} as defined above
        sc_h_Wk = StandardScaler().fit(H[tr_idx])
        reg_Wk, alpha_Wk, _ = ridge_cv_fit(sc_h_Wk.transform(H[tr_idx]), y_k[tr_idx],
                                        alphas, args.val_frac, SEED + k*23)

        # Effective linear map from RAW H to y_k
        Wk_eff = reg_Wk.coef_ @ np.diag(1.0 / sc_h_Wk.scale_)

        # Build M_h = I + W_h(1) from the k=1 Δh model on RAW coords (already trained earlier as reg1)
        coef_1 = reg1.coef_
        D_H = sc_ha_roll.scale_[:hidden_sz]  # scaler from the k=1 [H,A] model
        B_H_eff = coef_1[:, :hidden_sz] @ np.diag(1.0 / D_H)
        M_h = np.eye(hidden_sz) + B_H_eff
        M_h_k = np.linalg.matrix_power(M_h, k)

        frob = np.linalg.norm(Wk_eff - M_h_k, ord="fro")
        rho_Wk = spectral_radius(Wk_eff)
        rho_Mh = spectral_radius(M_h)
        print(f"‖W_k − (I+W_1)^k‖_F : {frob:.3f}    ρ(W_k): {rho_Wk:.3f}    ρ(I+W_1): {rho_Mh:.3f}")

    # -------- Rollout report (from the 1-step model) ----------
    print("\n" + "=" * 90)
    print("Open-loop / Teacher-forcing rollouts using learned 1-step Δh model")
    print("(avg R² over test starts; horizons 1..max_k)")
    print("=" * 90)
    ol = rollout_scores["open_loop"]
    tf = rollout_scores["teacher_forcing"]
    ks = sorted(ol.keys())
    print("k\topen_loop_R2\tteacher_forcing_R2")
    for k in ks:
        print(f"{k}\t{ol[k]:.3f}\t\t{tf[k]:.3f}")



if __name__ == "__main__":
    main()
