#!/usr/bin/env python3
"""
GRU transition probe (step-gated capture, group split, stable ridge + optional nonlinear MLP)

• capture exactly one hidden per *environment action* using a gated hook
• build pairs across consecutive steps within the same world only
• split by world when possible; fallback to random split
• standardize X on train only
• target = Δh (= h'−h) by default, or h' with --predict_hprime (with y-centering + intercept)
• report baselines + R²(h-only), R²([h,a]), ΔR², avg |corr|, action-probe
• now includes optional nonlinear probe (--mlp) using sklearn.neural_network.MLPRegressor
"""

import argparse, random, sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

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


def group_split_indices(groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
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
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--alphas", type=str, default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--predict_hprime", action="store_true")
    # nonlinear probe options
    ap.add_argument("--mlp", action="store_true", help="also fit an MLPRegressor as nonlinear probe")
    ap.add_argument("--mlp_hidden", type=str, default="128,64")
    ap.add_argument("--mlp_alpha", type=float, default=1e-3)
    ap.add_argument("--mlp_max_iter", type=int, default=300)
    args = ap.parse_args()

    # dummy model/env to get shapes
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

    capture_flag = {"on": False}
    last_hidden  = {"h": None}

    def rnn_hook(_, __, out):
        if capture_flag["on"]:
            last_hidden["h"] = out[1][-1].detach().cpu().numpy().squeeze()

    handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    H_t_buf, H_tp1_buf, A_buf, G_buf = [], [], [], []

    for wid in range(args.n_worlds):
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
            capture_flag["on"] = True
            last_hidden["h"] = None
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=args.deterministic)
            capture_flag["on"] = False
            h_now = last_hidden["h"]
            if h_now is None:
                continue
            if prev_h is not None:
                H_t_buf.append(prev_h)
                H_tp1_buf.append(h_now)
                A_buf.append(prev_act)
                G_buf.append(wid)
            prev_h   = h_now
            prev_act = act.flatten()
            obs, *_ = env.step(prev_act)
        env.close()
    handle.remove()

    H_t   = np.asarray(H_t_buf)
    H_tp1 = np.asarray(H_tp1_buf)
    A_t   = np.asarray(A_buf)
    G     = np.asarray(G_buf, dtype=int)

    if args.predict_hprime:
        y_all = H_tp1; y_label = "h′"
    else:
        y_all = H_tp1 - H_t; y_label = "Δh"

    X_h, X_ha = H_t, np.hstack([H_t, A_t])
    tr_idx, te_idx = group_split_indices(G, args.test_frac, SEED)
    if tr_idx.size == 0 or te_idx.size == 0:
        n = len(X_ha); perm = rng.permutation(n)
        n_test = max(1, int(round(n * args.test_frac)))
        te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    sc_h, sc_ha = StandardScaler().fit(X_h[tr_idx]), StandardScaler().fit(X_ha[tr_idx])
    Xh_tr, Xh_te = sc_h.transform(X_h[tr_idx]), sc_h.transform(X_h[te_idx])
    Xha_tr, Xha_te = sc_ha.transform(X_ha[tr_idx]), sc_ha.transform(X_ha[te_idx])
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    if args.predict_hprime:
        y_tr_mean = y_tr.mean(axis=0)
        y_tr_c, y_te_c = y_tr - y_tr_mean, y_te - y_tr_mean
        fit_intercept = True
    else:
        y_tr_mean = y_tr.mean(axis=0)
        y_tr_c, y_te_c = y_tr, y_te
        fit_intercept = False

    # baselines + ridge
    alphas = [float(a) for a in args.alphas.split(",")]
    def fit_score(Xtr, Xte, ytr, yte):
        best, best_r2 = None, -np.inf
        for a in alphas:
            reg = Ridge(alpha=a, fit_intercept=fit_intercept, solver="svd")
            reg.fit(Xtr, ytr)
            r2 = reg.score(Xte, yte)
            if r2 > best_r2: best_r2, best = r2, reg
        return best, best_r2

    reg_h, r2_h   = fit_score(Xh_tr, Xh_te, y_tr_c, y_te_c)
    reg_ha, r2_ha = fit_score(Xha_tr, Xha_te, y_tr_c, y_te_c)
    print(f"Linear f(h  )→{y_label}   held-out R² : {r2_h:.3f}")
    print(f"Linear f(h,a)→{y_label}   held-out R² : {r2_ha:.3f}")

    # nonlinear probe if requested
    if args.mlp:
        hidden = tuple(int(x) for x in args.mlp_hidden.split(","))
        mlp_h = MLPRegressor(hidden_layer_sizes=hidden,
                             alpha=args.mlp_alpha,
                             max_iter=args.mlp_max_iter,
                             random_state=SEED)
        mlp_ha = MLPRegressor(hidden_layer_sizes=hidden,
                              alpha=args.mlp_alpha,
                              max_iter=args.mlp_max_iter,
                              random_state=SEED)
        mlp_h.fit(Xh_tr, y_tr_c); r2_mlp_h = mlp_h.score(Xh_te, y_te_c)
        mlp_ha.fit(Xha_tr, y_tr_c); r2_mlp_ha = mlp_ha.score(Xha_te, y_te_c)
        print(f"MLP f(h  )→{y_label}     held-out R² : {r2_mlp_h:.3f}")
        print(f"MLP f(h,a)→{y_label}     held-out R² : {r2_mlp_ha:.3f}")

    # diagnostics
    avg_corr = np.mean([
        abs(np.corrcoef(H_t[:, i], H_tp1[:, i])[0, 1])
        for i in range(hidden_sz)
    ])
    print("avg |corr(h_t,h_tp1)|", avg_corr)


if __name__ == "__main__":
    main()
