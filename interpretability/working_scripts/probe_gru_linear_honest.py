#!/usr/bin/env python3
"""
GRU transition probe (step-gated capture, group split, stable ridge)

• capture exactly one hidden per *environment action* using a gated hook
• build pairs across consecutive steps within the same world only
• split by world when possible; fallback to random split
• standardize X on train only
• target = Δh (= h'−h) by default, or h' with --predict_hprime (with y-centering + intercept)
• report baselines + R²(h-only), R²([h,a]), ΔR², avg |corr|, action-probe
"""

import argparse, random, sys
from pathlib import Path
from typing import List, Tuple

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


def group_split_indices(groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a split by group labels (world IDs)."""
    uniq = np.unique(groups)
    n_groups = len(uniq)
    if n_groups < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_test = max(1, min(int(round(n_groups * test_frac)), n_groups - 1))
    rng_local = np.random.default_rng(seed)
    test_groups = rng_local.choice(uniq, size=n_test, replace=False)
    test_mask = np.isin(groups, test_groups)
    train_mask = ~test_mask
    return np.where(train_mask)[0], np.where(test_mask)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--n_worlds", type=int, default=20)
    ap.add_argument("--max_step", type=int, default=400)
    ap.add_argument("--test_frac", type=float, default=0.25,
                    help="fraction of *worlds* reserved for test (fallback to random pair split if <2 worlds)")
    ap.add_argument("--alphas", type=str, default="1e-3,1e-2,1e-1,1,10",
                    help="comma-separated ridge alphas to try")
    ap.add_argument("--deterministic", action="store_true",
                    help="use deterministic policy for action selection")
    ap.add_argument("--predict_hprime", action="store_true",
                    help="predict h' instead of Δh; when set, y is centered on train and fit_intercept=True")
    args = ap.parse_args()

    # ── build env / model (dummy for spaces/weights) ──
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    hidden_sz = model.ltl_net.rnn.hidden_size
    act_dim   = int(np.prod(dummy.action_space.shape))
    print("Observation shape:", dummy.observation_space.shape)
    print("Num LTLNet params :", sum(p.numel() for p in model.parameters()))
    print("GRU hidden        :", hidden_sz)
    print("Action dim        :", act_dim)
    dummy.close()

    # ── step-gated hook: capture the *last* forward inside get_action ──
    capture_flag = {"on": False}
    last_hidden  = {"h": None}
    current_world = {"id": -1}

    def rnn_hook(_, __, out):
        if capture_flag["on"]:
            # overwrite each time; after get_action returns, this is the final forward used
            last_hidden["h"] = out[1][-1].detach().cpu().numpy().squeeze()

    handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    # ── gather pairs by aligning across *consecutive* env steps ─────────
    H_t_buf, H_tp1_buf, A_buf, G_buf = [], [], [], []

    for wid in range(args.n_worlds):
        current_world["id"] = wid
        goal     = GOALS[wid % len(GOALS)]
        env      = make_env(ENV, FixedSampler.partial(goal), sequence=False)
        props    = set(env.get_propositions())
        planner  = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent    = Agent(model, planner, propositions=props)
        obs      = env.reset(seed=SEED + 100 * wid)
        agent.reset()

        prev_h   = None
        prev_act = None

        for step in range(args.max_step):
            # capture h for this obs/action selection
            capture_flag["on"] = True
            last_hidden["h"] = None
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=args.deterministic)
            capture_flag["on"] = False

            h_now = last_hidden["h"]
            if h_now is None:
                # If this ever happens, planner never called the RNN — extremely unlikely; skip step
                continue

            if prev_h is not None:
                # finalize previous pair with current h as h_{t+1}
                H_t_buf.append(prev_h)
                H_tp1_buf.append(h_now)
                A_buf.append(prev_act)
                G_buf.append(wid)

            # update for next step
            prev_h   = h_now
            prev_act = act.flatten()

            obs, *_ = env.step(prev_act)

        env.close()

    handle.remove()

    # ── arrays ─────────────────────────────────────────
    H_t   = np.asarray(H_t_buf)
    H_tp1 = np.asarray(H_tp1_buf)
    A_t   = np.asarray(A_buf)
    G     = np.asarray(G_buf, dtype=int)

    # world diagnostics
    uniq, counts = np.unique(G, return_counts=True)
    by_world = dict(zip(map(int, uniq), map(int, counts)))
    print("pairs collected     :", len(G))
    print("pairs by world      :", by_world)

    if len(G) == 0:
        raise RuntimeError("No (h,a,h') pairs were collected. Something's off with gating or get_action().")

    # targets and features
    predict_hprime = args.predict_hprime
    if predict_hprime:
        y_all = H_tp1
        y_label = "h′"
    else:
        y_all = H_tp1 - H_t
        y_label = "Δh"

    X_h  = H_t
    X_ha = np.hstack([H_t, A_t])

    # ── split by world when possible; otherwise fallback to random split ──
    tr_idx, te_idx = group_split_indices(G, args.test_frac, SEED)
    if tr_idx.size == 0 or te_idx.size == 0:
        print("⚠️  Not enough distinct worlds; falling back to random pair-level split.")
        n = len(X_ha)
        perm = rng.permutation(n)
        n_test = max(1, int(round(n * args.test_frac)))
        te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    # standardize X on train only
    sc_h  = StandardScaler().fit(X_h[tr_idx])
    sc_ha = StandardScaler().fit(X_ha[tr_idx])
    Xh_tr, Xh_te   = sc_h.transform(X_h[tr_idx]),  sc_h.transform(X_h[te_idx])
    Xha_tr, Xha_te = sc_ha.transform(X_ha[tr_idx]), sc_ha.transform(X_ha[te_idx])

    # targets (with y-centering + intercept for h′)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    if predict_hprime:
        y_tr_mean = y_tr.mean(axis=0)
        y_tr_c = y_tr - y_tr_mean
        y_te_c = y_te - y_tr_mean
        fit_intercept = True
    else:
        y_tr_mean = y_tr.mean(axis=0)  # baseline convenience
        y_tr_c = y_tr
        y_te_c = y_te
        fit_intercept = False

    # baselines
    if predict_hprime:
        # Baseline 1: copy h_t (centered to match y space)
        r2_copy = r2_score(y_te_c, X_h[te_idx] - y_tr_mean)
        # Baseline 2: mean(train) → zero vector in centered space
        r2_mean = r2_score(y_te_c, np.zeros_like(y_te_c))
        print(f"Baseline copy(h_t)  held-out R² : {r2_copy:.3f}")
        print(f"Baseline mean(train) held-out R² : {r2_mean:.3f}")
    else:
        # Baseline 1: Δh ≡ 0
        r2_zero = r2_score(y_te, np.zeros_like(y_te))
        # Baseline 2: Δh ≡ mean(train)
        r2_mean = r2_score(y_te, np.broadcast_to(y_tr_mean, y_te.shape))
        print(f"Baseline Δh≡0        held-out R² : {r2_zero:.3f}")
        print(f"Baseline Δh≡mean(tr) held-out R² : {r2_mean:.3f}")

    # ── small alpha sweep with SVD solver (stable) ────
    alphas = [float(a) for a in args.alphas.split(",")]
    def fit_score(Xtr, Xte, ytr, yte, fit_intercept: bool):
        best = None; best_r2 = -np.inf
        for a in alphas:
            reg = Ridge(alpha=a, fit_intercept=fit_intercept, solver="svd")
            reg.fit(Xtr, ytr)
            r2 = reg.score(Xte, yte)
            if r2 > best_r2:
                best_r2, best = r2, reg
        return best, best_r2

    reg_h,  r2_h  = fit_score(Xh_tr,  Xh_te,  y_tr_c, y_te_c, fit_intercept)
    reg_ha, r2_ha = fit_score(Xha_tr, Xha_te, y_tr_c, y_te_c, fit_intercept)

    arrow = "→"
    print(f"Linear f(h  ){arrow}{y_label}   held-out R² : {r2_h:.3f}")
    print(f"Linear f(h,a){arrow}{y_label}   held-out R² : {r2_ha:.3f}")
    print(f"ΔR² (add actions)                 : {r2_ha - r2_h:+.3f}")

    # ── diagnostic: avg |corr(h_t, h_{t+1})| ─────────
    if hidden_sz == 1:
        avg_corr = abs(np.corrcoef(H_t[:, 0], H_tp1[:, 0])[0, 1])
    else:
        avg_corr = np.mean([
            abs(np.corrcoef(H_t[:, i], H_tp1[:, i])[0, 1])
            for i in range(hidden_sz)
        ])
    print("avg |corr(h_t,h_tp1)|", avg_corr)

    # ── optional: predict action from h_t (quick linear probe) ─────────
    try:
        sc_act = StandardScaler().fit(X_h[tr_idx])
        Xh_tr_act = sc_act.transform(X_h[tr_idx])
        Xh_te_act = sc_act.transform(X_h[te_idx])
        reg_act, r2_act = fit_score(Xh_tr_act, Xh_te_act, A_t[tr_idx], A_t[te_idx], fit_intercept=True)
        print(f"h_t → action (Ridge)     held-out R² : {r2_act:.3f}")
    except Exception as e:
        print(f"h_t → action probe failed: {e}")


if __name__ == "__main__":
    main()
