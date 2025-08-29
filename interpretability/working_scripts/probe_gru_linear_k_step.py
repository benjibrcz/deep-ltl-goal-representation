#!/usr/bin/env python3
"""
GRU k-step transition probe (actor_state-first, robust splits)

• One hidden per env step via agent.actor_state (fallback to gated hook)
• Horizon: --k (Δh_k by default, or h' with --predict_hprime)
• Split choice: --split {world,random}
• Features: h_t vs [h_t, a_{t:t+k-1}] (concat window)
• Train-time standardization, alpha sweep (Ridge)
• Baselines & diagnostics
"""

import argparse, random, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

ENV, EXP, SEED = "PointLtl2-v0", "big_test", 0
GOALS          = [f"FG {c}" for c in ["blue", "green", "yellow", "magenta"]]
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


def split_indices(groups: np.ndarray, test_frac: float, seed: int, mode: str):
    n = len(groups)
    if mode == "random":
        rs = np.random.RandomState(seed)
        perm = rs.permutation(n)
        n_test = max(1, int(round(n * test_frac)))
        return perm[n_test:], perm[:n_test]
    # world split
    uniq = np.unique(groups)
    if len(uniq) < 2:
        print("⚠️  Not enough distinct worlds; falling back to random split.")
        return split_indices(groups, test_frac, seed, "random")
    rs = np.random.RandomState(seed)
    n_test_groups = max(1, int(round(len(uniq) * test_frac)))
    te_worlds = rs.choice(uniq, size=n_test_groups, replace=False)
    te_mask = np.isin(groups, te_worlds)
    tr, te = np.where(~te_mask)[0], np.where(te_mask)[0]
    if tr.size == 0 or te.size == 0:
        print("⚠️  Degenerate world split; falling back to random split.")
        return split_indices(groups, test_frac, seed, "random")
    return tr, te


def fit_ridge_sweep(Xtr, ytr, Xte, yte, alphas):
    best, best_r2, best_a = None, -np.inf, None
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=False, solver="auto")
        reg.fit(Xtr, ytr)
        r2 = reg.score(Xte, yte)
        if r2 > best_r2:
            best, best_r2, best_a = reg, r2, a
    return best, best_r2, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--n_worlds", type=int, default=20)
    ap.add_argument("--max_step", type=int, default=400)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--predict_hprime", action="store_true")
    ap.add_argument("--split", choices=["world", "random"], default="world")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--alphas", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    assert args.k >= 1, "--k must be ≥ 1"
    alphas = [float(a) for a in args.alphas.split(",")]

    # ── build env/model ──
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    hidden_sz = model.ltl_net.rnn.hidden_size
    print("Observation shape:", dummy.observation_space.shape)
    print("Num LTLNet params:", sum(p.numel() for p in model.parameters()))
    print("GRU hidden        :", hidden_sz)
    dummy.close()

    # ── buffers (actor_state-first) ──
    H_seq: List[np.ndarray] = []    # one hidden per env step (post get_action)
    A_seq: List[np.ndarray] = []
    W_seq: List[int]        = []    # world per hidden

    # gated hook fallback (only if actor_state unavailable)
    gated = {"enabled": False}
    current_world_id = -1
    def rnn_hook(_, __, out):
        if not gated["enabled"]:
            return
        h = out[1][-1].detach().cpu().numpy().squeeze()
        H_seq.append(h)
        W_seq.append(current_world_id)
        gated["enabled"] = False  # record exactly once

    hook_handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    action_dim = None
    used_actor_state = True

    for wid in range(args.n_worlds):
        current_world_id = wid
        goal     = GOALS[wid % len(GOALS)]
        env      = make_env(ENV, FixedSampler.partial(goal), sequence=False)
        props    = set(env.get_propositions())
        planner  = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent    = Agent(model, planner, propositions=props)
        obs      = env.reset(seed=SEED + 100 * wid)
        agent.reset()

        have_actor_state = isinstance(getattr(agent, "actor_state", None), torch.Tensor)

        # local counters for robust diagnostics
        h_before = len(W_seq)
        actions_this_world = 0

        for _ in range(args.max_step):
            # enable hook BEFORE forward if we don't have actor_state
            if not have_actor_state:
                used_actor_state = False
                gated["enabled"] = True

            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=args.deterministic)

            act = np.asarray(act).flatten()
            if action_dim is None:
                action_dim = act.shape[0]
                print("Action dim        :", action_dim)

            actions_this_world += 1

            # if actor_state path: record hidden now (post forward)
            if have_actor_state:
                h_post = agent.actor_state[-1].detach().cpu().numpy().squeeze()
                H_seq.append(h_post)
                W_seq.append(wid)

            A_seq.append(act)
            obs, *_ = env.step(act)

        h_logged = len(W_seq) - h_before
        print(f"world {wid}: actions={actions_this_world}, h_samples={h_logged}")
        env.close()

    hook_handle.remove()

    if len(H_seq) == 0:
        raise RuntimeError("No hidden states captured. (actor_state missing and gated hook didn’t fire)")

    H_seq = np.asarray(H_seq)   # (S, hidden)
    A_seq = np.asarray(A_seq)   # (S, act_dim)
    W_seq = np.asarray(W_seq)   # (S,)

    # ── build k-step tuples per world ──
    X_h_list, X_ha_list, y_list, G_list = [], [], [], []
    for w in np.unique(W_seq):
        idx = np.where(W_seq == w)[0]
        H_w = H_seq[idx]
        A_w = A_seq[idx]
        T = min(len(H_w) - args.k, len(A_w) - (args.k - 1))
        if T <= 0:
            continue

        H_t   = H_w[:T]
        H_tpk = H_w[args.k:args.k + T]

        # concat action window a_t..a_{t+k-1}
        A_win = []
        for t in range(T):
            window = A_w[t:t + args.k]
            if len(window) < args.k:
                break
            A_win.append(window.reshape(-1))
        if len(A_win) != T:
            T = len(A_win)
            H_t   = H_t[:T]
            H_tpk = H_tpk[:T]
        A_win = np.asarray(A_win) if T > 0 else np.zeros((0, args.k * action_dim))

        y = H_tpk if args.predict_hprime else (H_tpk - H_t)
        X_h  = H_t
        X_ha = np.hstack([H_t, A_win]) if A_win.size else H_t

        X_h_list.append(X_h)
        X_ha_list.append(X_ha)
        y_list.append(y)
        G_list.append(np.full(T, w, dtype=int))

    if not X_h_list:
        print("pairs collected     : 0")
        print("pairs by world      : {}")
        raise RuntimeError("No valid k-step pairs (try smaller --k or more steps).")

    X_h  = np.vstack(X_h_list)
    X_ha = np.vstack(X_ha_list)
    y    = np.vstack(y_list)
    G    = np.concatenate(G_list)

    uniq, counts = np.unique(G, return_counts=True)
    print("pairs collected     :", len(X_h))
    print("pairs by world      :", dict(zip(map(int, uniq), map(int, counts))))

    # diagnostics: avg |corr(h_t, h_{t+k})|
    def avg_abs_corr(H1, H2):
        vals = []
        for i in range(H1.shape[1]):
            c = np.corrcoef(H1[:, i], H2[:, i])[0, 1]
            if np.isfinite(c):
                vals.append(abs(c))
        return float(np.mean(vals)) if vals else float("nan")

    H_t_for_corr = X_h
    H_tpk_for_corr = y + X_h if not args.predict_hprime else y
    print("avg |corr(h_t,h_{t+k})|", avg_abs_corr(H_t_for_corr, H_tpk_for_corr))

    # ── split ──
    tr_idx, te_idx = split_indices(G, args.test_frac, SEED, args.split)

    # standardize on train only
    sc_h  = StandardScaler().fit(X_h[tr_idx])
    Xh_tr, Xh_te = sc_h.transform(X_h[tr_idx]), sc_h.transform(X_h[te_idx])

    sc_ha = StandardScaler().fit(X_ha[tr_idx])
    Xha_tr, Xha_te = sc_ha.transform(X_ha[tr_idx]), sc_ha.transform(X_ha[te_idx])

    y_tr, y_te = y[tr_idx], y[te_idx]

    # baselines
    def r2(ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - ytrue.mean(axis=0, keepdims=True)) ** 2) + 1e-12
        return 1.0 - ss_res / ss_tot

    if args.predict_hprime:
        yhat_copy = X_h[te_idx]            # predict h' ≈ h_t
        mu_tr = y_tr.mean(axis=0, keepdims=True)
        yhat_mean = np.repeat(mu_tr, repeats=len(te_idx), axis=0)
        print(f"Baseline copy(h_t)    held-out R² : {r2(y_te, yhat_copy):.3f}")
        print(f"Baseline mean(train)  held-out R² : {r2(y_te, yhat_mean):.3f}")
    else:
        zeros = np.zeros_like(y_te)
        mu_tr = y_tr.mean(axis=0, keepdims=True)
        yhat_mean = np.repeat(mu_tr, repeats=len(te_idx), axis=0)
        print(f"Baseline Δh≡0         held-out R² : {r2(y_te, zeros):.3f}")
        print(f"Baseline mean(train)  held-out R² : {r2(y_te, yhat_mean):.3f}")

    # ridge
    reg_h,  r2_h,  a_h  = fit_ridge_sweep(Xh_tr,  y_tr, Xh_te,  y_te, alphas)
    reg_ha, r2_ha, a_ha = fit_ridge_sweep(Xha_tr, y_tr, Xha_te, y_te, alphas)
    arrow = "→h′" if args.predict_hprime else "→Δh"
    print(f"Linear f(h  ){arrow}   held-out R² : {r2_h:.3f} (α={a_h:g})")
    print(f"Linear f(h,a){arrow}   held-out R² : {r2_ha:.3f} (α={a_ha:g})")
    print(f"ΔR² (add actions)                 : {r2_ha - r2_h:+.3f}")

    # optional: decode first action in window from h_t
    A_first = X_ha[:, hidden_sz:hidden_sz + (args.k * action_dim)] if X_ha.shape[1] > hidden_sz else np.zeros((len(X_ha), args.k * action_dim))
    A_first = A_first[:, :action_dim] if A_first.shape[1] >= action_dim else A_first
    sc_h2 = StandardScaler().fit(X_h[tr_idx])
    Xh_tr2, Xh_te2 = sc_h2.transform(X_h[tr_idx]), sc_h2.transform(X_h[te_idx])
    _, r2_a, _ = fit_ridge_sweep(Xh_tr2, A_first[tr_idx], Xh_te2, A_first[te_idx], alphas)
    print(f"h_t → action (Ridge)     held-out R² : {r2_a:.3f}")


if __name__ == "__main__":
    main()
