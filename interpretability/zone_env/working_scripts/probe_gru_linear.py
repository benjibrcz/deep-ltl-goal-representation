#!/usr/bin/env python3
"""
Lightweight GRU-probe with transition analysis.

• collects (h_t, a_t, h_{t+1}) tuples
• fits a linear map  f(h, a) → h′
• reports held-out R² and per-unit |corr(h_t, h_{t+1})|
"""

import argparse, os, random, sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split   # NEW

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
N_WORLDS, N_ROLLOUT, MAX_STEP = 20, 20, 700
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
# -----------------------------------------------------


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--test_frac", type=float, default=0.20,
                    help="fraction of (h,a,h') pairs reserved for test")
    args = ap.parse_args()

    # ── build env / model ─────────────────────────────
    dummy_env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg       = model_configs[ENV]
    store     = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()
    hidden_sz = model.ltl_net.rnn.hidden_size
    print("Observation shape:", dummy_env.observation_space.shape)
    print("Num LTLNet params :", sum(p.numel() for p in model.parameters()))
    print("GRU hidden        :", hidden_sz)

    # ── hook to grab hidden after each forward ────────
    h_buf: List[np.ndarray] = []

    def rnn_hook(_, __, out):
        # out = (output, h_n); take last-layer hidden
        h_buf.append(out[1][-1].detach().cpu().numpy().squeeze())

    hook_handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    # ── gather (h_t, a_t, h_{t+1}) pairs ─────────────
    a_buf: List[np.ndarray] = []

    for wid in range(N_WORLDS):
        goal     = GOALS[wid % len(GOALS)]
        env      = make_env(ENV, FixedSampler.partial(goal), sequence=False)
        props    = set(env.get_propositions())
        planner  = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent    = Agent(model, planner, propositions=props)
        obs      = env.reset(seed=SEED + 100 * wid)
        agent.reset()

        for _ in range(MAX_STEP):
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=True)
            a_buf.append(act.flatten())                       # a_t

            obs, *_ = env.step(act.flatten())                 # env step

            # we now have a new hidden in h_buf (hook fired)
        env.close()

    hook_handle.remove()

    # ── align buffers ─────────────────────────────────
    H = np.asarray(h_buf)
    A = np.asarray(a_buf)
    T = min(len(A), len(H) - 1)                               # pairs count
    H_t, H_tp1, A = H[:T], H[1:T + 1], A[:T]
    print("pairs collected     :", T)

    # ── train / test split ────────────────────────────
    X = np.hstack([H_t, A])                                   # (T, hidden+act)
    X_train, X_test, y_train, y_test = train_test_split(
        X, H_tp1, test_size=args.test_frac, random_state=SEED, shuffle=True
    )

    reg = Ridge(alpha=1e-2, fit_intercept=False).fit(X_train, y_train)
    print(f"Linear f(h,a)->h'   held-out R² : {reg.score(X_test, y_test):.3f}")

    # ── diagnostic corr per hidden unit ───────────────
    if hidden_sz == 1:                                        # edge-case scalar
        avg_corr = abs(np.corrcoef(H_t[:, 0], H_tp1[:, 0])[0, 1])
    else:
        avg_corr = np.mean([
            abs(np.corrcoef(H_t[:, i], H_tp1[:, i])[0, 1])
            for i in range(hidden_sz)
        ])
    print("avg |corr(h_t,h_tp1)|", avg_corr)


if __name__ == "__main__":
    main()
