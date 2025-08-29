#!/usr/bin/env python3
# interpretability/working_scripts/world_model_probe_and_steer.py
"""
DeepLTL world-model: probe APs & automaton, predict next-AP with actions, and steer via AP directions.

This version fixes AP extraction by introspecting env/agent/planner/tracker for a labeler so
--do_ap_probe actually runs. It prints which label source it uses.
"""

import argparse, random, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from inspect import signature, isfunction, ismethod

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

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

# ======== helpers: splits, AP extractor, steering hook ========

def group_split_indices(groups: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_test = max(1, min(int(round(len(uniq) * test_frac)), len(uniq) - 1))
    te_groups = np.random.default_rng(seed).choice(uniq, size=n_test, replace=False)
    te_mask = np.isin(groups, te_groups)
    return np.where(~te_mask)[0], np.where(te_mask)[0]


LABEL_METHOD_NAMES = [
    "label", "label_state", "label_propositions", "get_labels", "get_label",
    "label_fn", "label_func", "ap", "ap_truth", "get_ap_truth",
    "ltl_label", "ltl_labels", "labels",
    "get_true_props", "true_props", "current_props", "ltl_true_props",
]

def _normalize_ap_generic(ret, props):
    """Accepts set/list/tuple/dict/bool-array and returns 0/1 vector or None if shape mismatched."""
    if ret is None:
        return None
    # set/list/tuple of names
    if isinstance(ret, (set, list, tuple)):
        s = set(map(str, ret))
        return np.array([1 if str(p) in s else 0 for p in props], dtype=np.int64)
    # dict name->bool
    if isinstance(ret, dict):
        return np.array([1 if ret.get(p, False) else 0 for p in props], dtype=np.int64)
    # array-like already aligned
    arr = np.asarray(ret)
    arr = arr.reshape(-1)
    if arr.size == len(props):
        return (arr > 0).astype(np.int64)
    return None

def make_auto_ap_extractor(env, agent, planner, props, verbose=False):
    """
    Returns (extractor_fn, source_desc). extractor_fn(obs) -> 0/1 vector or None.
    Searches env/agent/planner/tracker for *any* callable/attr that yields AP truth we can normalize.
    """
    LOOK_IN = [
        ("env", env),
        ("agent", agent),
        ("planner", planner),
        ("planner.tracker", getattr(planner, "tracker", None)),
        ("agent.tracker", getattr(agent, "tracker", None)),
    ]

    # Build candidate list:
    # 1) Any attribute whose name hints it's about labels/APs; 2) Any callable we can try.
    name_hints = ("label", "ap", "prop", "truth")
    candidates = []

    def add_candidate(desc, fn, takes_obs: bool):
        candidates.append((desc, fn, takes_obs))

    for obj_name, obj in LOOK_IN:
        if obj is None: 
            continue
        # Named attributes with hints
        for name in dir(obj):
            if not any(h in name.lower() for h in name_hints):
                continue
            thing = getattr(obj, name, None)
            if thing is None:
                continue
            # callable?
            if callable(thing):
                # Try to infer if it takes one argument (obs)
                takes_obs = False
                try:
                    sig = signature(thing)
                    takes_obs = (len(sig.parameters) >= 1)
                except Exception:
                    # if we can't inspect, try both later
                    takes_obs = True
                add_candidate(f"{obj_name}.{name}(obs?)", thing, takes_obs)
            else:
                # Non-callable attribute (maybe a set/dict/list of props or a ready-made vector)
                def getter(_obs, _obj=obj, _name=name):
                    return getattr(_obj, _name)
                add_candidate(f"{obj_name}.{name}", getter, False)

        # Any other callable attributes (fallback, very liberal)
        for name in dir(obj):
            thing = getattr(obj, name, None)
            if callable(thing) and (isfunction(thing) or ismethod(thing)):
                # Skip dunders and obvious non-relevant methods
                if name.startswith("__"):
                    continue
                add_candidate(f"{obj_name}.{name}(obs?)", thing, True)

    if verbose:
        print("AP candidate count:", len(candidates))

    def extractor(obs):
        # Try candidates in order: first with obs (if supported), then no-arg.
        for desc, fn, takes_obs in candidates:
            # Try with obs
            for try_obs in [True, False]:
                if try_obs and not takes_obs:
                    continue
                try:
                    out = fn(obs) if try_obs else fn()
                except TypeError:
                    # wrong arity; skip this mode
                    continue
                except Exception:
                    # runtime error inside candidate; skip
                    continue
                vec = _normalize_ap_generic(out, props)
                if vec is not None and vec.shape[0] == len(props):
                    extractor.source = f"{desc}  (called with {'obs' if try_obs else 'no-arg'})"
                    return vec
        return None

    extractor.source = None
    return extractor


class GRUSteerHook:
    """ Adds α·v to GRU output and last hidden (safe for batch=1, seq=1). """
    def __init__(self, v: np.ndarray):
        self.v = v.astype(np.float32)
        self.alpha = 0.0
        self.enabled = False
        self.handle = None
    def _hook(self, module, inputs, outputs):
        if not self.enabled or self.alpha == 0.0:
            return outputs
        out, h_n = outputs
        v = torch.from_numpy(self.v * self.alpha).to(h_n.device)
        h_n = h_n.clone(); h_n[-1, 0, :] = h_n[-1, 0, :] + v
        out = out.clone(); out[-1, 0, :] = out[-1, 0, :] + v
        return (out, h_n)
    def attach(self, gru_module: torch.nn.GRU):
        if self.handle is None:
            self.handle = gru_module.register_forward_hook(self._hook)
    def detach(self):
        if self.handle is not None:
            self.handle.remove(); self.handle = None

# ========================= main =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    # data collection
    ap.add_argument("--n_worlds", type=int, default=24)
    ap.add_argument("--max_step", type=int, default=400)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--ap_debug", action="store_true", help="print AP labeler candidates and chosen source")
    # probes
    ap.add_argument("--do_ap_probe", action="store_true", help="train AP probes h→L(s) and [h,a]→L(s′)")
    ap.add_argument("--do_q_probe", action="store_true", help="try to train automaton-state probe h→q")
    # steering (kept for later use)
    ap.add_argument("--do_steer", action="store_true", help="evaluate steering using AP-probe probabilities")
    ap.add_argument("--steer_prop", type=str, default="")
    ap.add_argument("--alphas", type=str, default="0.0,0.25,0.5,1.0,1.5")
    args = ap.parse_args()

    # ── build env / model ──
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    hidden_sz = model.ltl_net.rnn.hidden_size
    act_dim   = int(np.prod(dummy.action_space.shape))
    print("Observation shape:", dummy.observation_space.shape)
    print("Num LTLNet params:", sum(p.numel() for p in model.parameters()))
    print("GRU hidden       :", hidden_sz)
    print("Action dim       :", act_dim)
    dummy.close()

    # ── step-gated hidden capture ──
    capture_flag = {"on": False}
    last_hidden  = {"h": None}
    def rnn_capture_hook(_, __, out):
        if capture_flag["on"]:
            last_hidden["h"] = out[1][-1].detach().cpu().numpy().squeeze()
    cap_handle = model.ltl_net.rnn.register_forward_hook(rnn_capture_hook)

    # ── collect (h, a, h′) and APs ──
    H_t_buf, H_tp1_buf, A_buf, G_buf = [], [], [], []
    L_t_buf, L_tp1_buf = [], []
    Q_t_buf = []  # optional automaton-state ids

    props_ref: Optional[List[str]] = None
    ap_source_once: Optional[str] = None

    for wid in range(args.n_worlds):
        goal     = GOALS[wid % len(GOALS)]
        env      = make_env(ENV, FixedSampler.partial(goal), sequence=False)
        # props as strings to keep alignment stable
        props    = [str(p) for p in env.get_propositions()]
        if props_ref is None:
            props_ref = props
        else:
            assert props == props_ref, f"World {wid} propositions mismatch."

        planner  = ExhaustiveSearch(model, set(props), num_loops=args.num_loops)
        agent    = Agent(model, planner, propositions=set(props))
        obs      = env.reset(seed=SEED + 100 * wid)
        agent.reset()

        # find an AP extractor for THIS env/agent/planner
        # after you build env / agent / planner and have `props = list(env.get_propositions())`
        ap_extractor = make_auto_ap_extractor(env, agent, planner, props, verbose=args.ap_debug)

        # before stepping:
        prev_L = ap_extractor(obs)
        if ap_extractor.source and wid == 0:
            print("✔ AP label source:", ap_extractor.source)


        prev_h   = None
        prev_act = None

        # try to read automaton state id
        def read_q():
            for path in [
                ("planner", "tracker", "state_id"),
                ("planner", "automaton_state_id"),
                ("tracker", "state_id"),
                ("automaton_state_id",),
            ]:
                cur = agent
                ok = True
                for attr in path:
                    cur = getattr(cur, attr, None)
                    if cur is None:
                        ok = False; break
                if ok and isinstance(cur, (int, np.integer)):
                    return int(cur)
            return None

        for step in range(args.max_step):
            # capture h for this obs/action selection
            capture_flag["on"] = True
            last_hidden["h"] = None
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=args.deterministic)
            capture_flag["on"] = False

            h_now = last_hidden["h"]
            if h_now is None:
                continue

            if prev_h is not None:
                # finalize previous pair with current h as h_{t+1}
                H_t_buf.append(prev_h)
                H_tp1_buf.append(h_now)
                A_buf.append(prev_act)
                G_buf.append(wid)
                if prev_L is not None:
                    L_t_buf.append(prev_L)

            # env step
            prev_h   = h_now
            prev_act = act.flatten()
            obs, _, done, _ = env.step(prev_act)

            # labels after stepping
            L_now = ap_extractor(obs)
            if L_now is not None and prev_L is not None:
                L_tp1_buf.append(L_now)
            prev_L = L_now

            q_now = read_q()
            if q_now is not None:
                Q_t_buf.append(q_now)

            if done:
                break

        env.close()

    cap_handle.remove()

    # ── stack arrays ──
    H_t   = np.asarray(H_t_buf)
    H_tp1 = np.asarray(H_tp1_buf)
    A_t   = np.asarray(A_buf)
    G     = np.asarray(G_buf, dtype=int)

    have_L = len(L_t_buf) == len(H_t) and len(L_tp1_buf) == len(H_t) and (len(L_t_buf) > 0)
    if have_L:
        L_t   = np.asarray(L_t_buf, dtype=int)
        L_tp1 = np.asarray(L_tp1_buf, dtype=int)
        P     = len(props_ref)
    else:
        L_t = L_tp1 = None
        P = 0

    have_Q = len(Q_t_buf) == len(H_t) and len(Q_t_buf) > 0 and (len(set(Q_t_buf)) > 1)
    if have_Q:
        Q_t = np.asarray(Q_t_buf, dtype=int)
    else:
        Q_t = None

    print(f"pairs collected: {len(G)} | hidden={H_t.shape} action={A_t.shape}")
    if have_L:
        print(f"AP labels present: L_t={L_t.shape} L_tp1={L_tp1.shape} | props={props_ref}")
        if ap_source_once:
            print(f"AP label source example: {ap_source_once}")
    else:
        print("⚠️  Could not extract AP labels. Try rerun with --ap_debug to list candidates;")
        print("    if none work, add your env’s labeler to make_ap_extractor().")
        print("Done."); return

    # ── split by world ──
    tr_idx, te_idx = group_split_indices(G, args.test_frac, SEED)
    if tr_idx.size == 0 or te_idx.size == 0:
        print("⚠️  Not enough distinct worlds; using random split.")
        perm = rng.permutation(len(G))
        n_test = max(1, int(round(len(G) * args.test_frac)))
        te_idx = perm[:n_test]; tr_idx = perm[n_test:]

    # ── scalers ──
    sc_h  = StandardScaler().fit(H_t[tr_idx])
    sc_ha = StandardScaler().fit(np.hstack([H_t, A_t])[tr_idx])
    Htr, Hte     = sc_h.transform(H_t[tr_idx]),  sc_h.transform(H_t[te_idx])
    HAtr, HAte   = sc_ha.transform(np.hstack([H_t, A_t])[tr_idx]), sc_ha.transform(np.hstack([H_t, A_t])[te_idx])

    # ============ AP probes ============
    print("\n== AP probes: h→L(s) ==")
    ap_probes: List[Optional[LogisticRegression]] = []
    for i in range(P):
        ytr, yte = L_t[tr_idx, i], L_t[te_idx, i]
        if len(np.unique(ytr)) < 2:
            print(f"  prop[{i}] {props_ref[i]} is constant in train; skipping.")
            ap_probes.append(None); continue
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Htr, ytr)
        yhat = clf.predict(Hte)
        proba = clf.predict_proba(Hte)[:, 1]
        acc = accuracy_score(yte, yhat)
        try:
            auc = roc_auc_score(yte, proba)
        except ValueError:
            auc = float("nan")
        ap_probes.append(clf)
        print(f"  {props_ref[i]:>10s}: acc={acc:.3f} auc={auc:.3f}")

    print("\n== Next-AP probes: [h,a]→L(s′) ==")
    next_ap_probes: List[Optional[LogisticRegression]] = []
    for i in range(P):
        ytr, yte = L_tp1[tr_idx, i], L_tp1[te_idx, i]
        if len(np.unique(ytr)) < 2:
            print(f"  next prop[{i}] {props_ref[i]} constant; skipping.")
            next_ap_probes.append(None); continue
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(HAtr, ytr)
        yhat = clf.predict(HAte)
        proba = clf.predict_proba(HAte)[:, 1]
        acc = accuracy_score(yte, yhat)
        try:
            auc = roc_auc_score(yte, proba)
        except ValueError:
            auc = float("nan")
        next_ap_probes.append(clf)
        print(f"  {props_ref[i]:>10s}: acc={acc:.3f} auc={auc:.3f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
