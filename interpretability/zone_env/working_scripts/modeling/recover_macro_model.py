#!/usr/bin/env python3
"""
Recover a proposition-level macro world model from a DeepLTL agent.

We estimate P_hat(dst | src, go-to-dst) by:
  - getting the agent to visit `src` (pre-phase, budget pre_steps)
  - then asking for `dst` (attempt, budget horizon)

Two modes:
  • default (reset-per-link): reach `src` in one env, then attempt `dst` in a fresh env
  • --stateful: use ONE env per episode; keep physics; swap LDBA goal via env.set_goal(...)

Results are appended pair-by-pair to CSV (resumable).
"""

import argparse, sys, csv, re, gc
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch

# -------- repo imports --------
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

# gym / gymnasium spaces
try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces  # type: ignore

torch.set_grad_enabled(False)


# ---------- helpers ----------
def goal_for_search(phi: str) -> str:
    """Map 'F x' → 'FG x' for planners if needed; otherwise pass through."""
    m = re.match(r"^\s*F\s+([A-Za-z]+)\s*$", phi)
    return f"FG {m.group(1)}" if m else phi

def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}

def step_unpack(ret):
    if len(ret) == 5:
        obs, rew, terminated, truncated, info = ret
        return obs, rew, bool(terminated or truncated), info
    elif len(ret) == 4:
        obs, rew, done, info = ret
        return obs, rew, bool(done), info
    raise RuntimeError(f"Unexpected env.step return of length {len(ret)}")

def coerce_action(act, action_space):
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1: a = np.repeat(a, need)
        if a.size != need: raise ValueError(f"Action size {a.size} != {need}")
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        a = int(np.asarray(act).ravel()[0]) if isinstance(act,(np.ndarray,list,tuple)) else int(act)
        if not (0 <= a < action_space.n): raise ValueError("Discrete out of range")
        return a
    elif isinstance(action_space, gspaces.MultiDiscrete):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        if a.size != action_space.nvec.size: raise ValueError("MultiDiscrete size mismatch")
        return a
    elif isinstance(action_space, gspaces.MultiBinary):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size != need: raise ValueError("MultiBinary size mismatch")
        return a.reshape(action_space.shape)
    return act

def deep_accept(info, reward, done):
    """Recursive accept/satisfy scan + positive terminal reward heuristic."""
    def any_truthy_accept(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if ("accept" in lk) or ("satisf" in lk):
                    try:
                        if bool(v): return True
                    except Exception: pass
                if any_truthy_accept(v): return True
            return False
        elif isinstance(x, (list, tuple)):
            return any(any_truthy_accept(v) for v in x)
        return False
    if any_truthy_accept(info): return True
    if done and isinstance(reward, (int,float)) and reward > 0: return True
    return False

def saw_prop(info, name: str) -> bool:
    if "propositions" in info:
        return any(str(p) == name for p in info["propositions"])
    return False


# --- Env cache (keyed by goal string) to avoid repeated heavy construction ---
class EnvCache:
    def __init__(self, env_id: str):
        self.env_id = env_id
        self.cache: Dict[str, object] = {}

    def get(self, goal: str):
        if goal not in self.cache:
            self.cache[goal] = make_env(self.env_id, FixedSampler.partial(goal_for_search(goal)), sequence=False)
        return self.cache[goal]

    def get_env(self, goal: str):
        return self.get(goal)

    def reset(self, goal: str, **kwargs):
        env = self.get(goal)
        return reset_unpack(env, **kwargs)

    def step(self, goal: str, action):
        env = self.get(goal)
        return step_unpack(env.step(action))

    def action_space(self, goal: str):
        env = self.get(goal)
        return env.action_space

    def close_all(self):
        for env in self.cache.values():
            try:
                env.close()
            except Exception:
                pass
        self.cache.clear()


# --------- rollouts ----------
def run_policy_until_prop_cached(ec: EnvCache, goal: str, agent: Agent, target_prop: str,
                                 max_steps: int, deterministic: bool, seed=None) -> Tuple[bool, int]:
    obs, info = ec.reset(goal, seed=seed)
    agent.reset()
    for t in range(max_steps):
        with torch.no_grad():
            act = agent.get_action(obs, {}, deterministic=deterministic)
        act = coerce_action(act, ec.action_space(goal))
        obs, rew, done, info = ec.step(goal, act)
        if saw_prop(info, target_prop) or deep_accept(info, rew, done):
            return True, t+1
        if done:
            return False, t+1
    return False, max_steps

def run_policy_until_prop_same_env(env, agent: Agent, target_prop: str,
                                   max_steps: int, deterministic: bool) -> Tuple[bool, int]:
    """
    Version that uses the SAME env instance (no reset); assumes env.set_goal(...) already called.
    """
    # We do NOT reset here: caller keeps physics continuous and will call agent.reset() before use
    # Grab the last obs/info by calling a zero-length reset? Not needed; we'll just step.
    steps = 0
    obs, info = None, None
    while steps < max_steps:
        # Fetch a minimal observation via a no-op? We simply rely on last returned 'obs' in loop below
        # For the first step, we need an obs. We get it with a dummy zero-step by reading from env.unwrapped?
        # Simpler: perform one normal step using the agent on the previous obs held by wrapper.
        # LDBAWrapper keeps self.obs updated; its step() will complete observation before returning.
        if obs is None:
            # seed not used here; env already reset by caller
            obs = env.obs  # LDBAWrapper caches the last obs
            if obs is None:
                obs, info = reset_unpack(env)

        with torch.no_grad():
            act = agent.get_action(obs, {}, deterministic=deterministic)
        act = coerce_action(act, env.action_space)
        obs, rew, done, info = step_unpack(env.step(act))
        steps += 1

        if saw_prop(info, target_prop) or deep_accept(info, rew, done):
            return True, steps
        if done:
            return False, steps
    return False, steps


def estimate_transition_cached(ec: EnvCache, agent: Agent, src_prop: str, dst_prop: str,
                               episodes: int, horizon: int, pre_steps: int,
                               seed: int, deterministic: bool, stateful: bool) -> Tuple[int,int]:
    succ = 0; att = 0
    for k in range(episodes):
        if not stateful:
            # old behavior: possibly different env instances
            ok_src, _ = run_policy_until_prop_cached(ec, f"FG {src_prop}", agent, src_prop,
                                                     pre_steps, deterministic, seed + 31*k)
            if not ok_src:
                continue
            ok_dst, _ = run_policy_until_prop_cached(ec, f"FG {dst_prop}", agent, dst_prop,
                                                     horizon, deterministic, seed + 37*k)
        else:
            # NEW: one env, keep physics; swap goal via set_goal + replan
            env = ec.get_env(f"FG {src_prop}")
            reset_unpack(env, seed=seed + 31*k)

            if hasattr(env, "set_goal"):
                env.set_goal(f"FG {src_prop}")
            agent.reset()
            ok_src, _ = run_policy_until_prop_same_env(env, agent, src_prop, pre_steps, deterministic)
            if not ok_src:
                continue

            if hasattr(env, "set_goal"):
                env.set_goal(f"FG {dst_prop}")
            agent.reset()
            ok_dst, _ = run_policy_until_prop_same_env(env, agent, dst_prop, horizon, deterministic)

        att += 1
        succ += int(ok_dst)

        if (k+1) % 5 == 0:
            gc.collect()
    return succ, att


def jeffreys_rate(succ: int, n: int) -> float:
    return (succ + 0.5) / (n + 1.0)


# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--episodes_per_pair", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=300, help="budget for the dst attempt")
    ap.add_argument("--pre_steps", type=int, default=200, help="budget to get to src before attempting dst")
    ap.add_argument("--out_csv", type=str, default="interpretability/working_scripts/macro_model.csv")
    ap.add_argument("--stateful", action="store_true", help="Use one env per episode and swap goals via set_goal without resetting physics.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Build dummy env + model once
    first_goal = f"FG {args.colors.split(',')[0].strip()}"
    dummy = make_env(args.env_id, FixedSampler.partial(goal_for_search(first_goal)), sequence=False)
    cfg   = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    props = set(dummy.get_propositions())
    try:
        print("Observation space :", dummy.observation_space)
        print("Action space      :", dummy.action_space)
    except Exception:
        pass
    print(f"Num model params  : {sum(p.numel() for p in model.parameters())}")
    dummy.close()

    planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
    agent   = Agent(model, planner, propositions=props)

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    pairs = [(s,t) for s in colors for t in colors if s != t]

    # Load existing CSV to support resume
    out_path = Path(args.out_csv)
    done_pairs = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                # differentiate by mode so you can keep both versions if desired
                stateful_flag = str(int(args.stateful))
                if r.get("stateful", None) == stateful_flag:
                    done_pairs.add((r["src"], r["dst"]))

    # Open CSV for append (create header if new)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    fcsv = open(out_path, "a", newline="")
    writer = csv.DictWriter(fcsv,
        fieldnames=["env_id","exp","seed","src","dst","succ","att","p_hat",
                    "episodes_per_pair","horizon","pre_steps","stateful"])
    if new_file:
        writer.writeheader()

    ec = EnvCache(args.env_id)

    print("\nEstimating macro transitions P_hat[t | s, go-to-t] ...")
    try:
        for (s,t) in pairs:
            if (s,t) in done_pairs:
                print(f"    {s:>7s} -> {t:<7s}  (skipping; already in CSV for stateful={int(args.stateful)})")
                continue

            print(f"    {s:>7s} -> {t:<7s}  ({args.episodes_per_pair} trials; stateful={int(args.stateful)}) ...", end="", flush=True)
            succ, att = estimate_transition_cached(ec, agent, s, t,
                                                   args.episodes_per_pair, args.horizon, args.pre_steps,
                                                   seed=args.seed, deterministic=args.deterministic,
                                                   stateful=args.stateful)
            p = jeffreys_rate(succ, max(1,att))
            print(f"  succ={succ}/{att}  p̂={p:.3f}")

            writer.writerow({
                "env_id": args.env_id, "exp": args.exp, "seed": args.seed,
                "src": s, "dst": t, "succ": succ, "att": att, "p_hat": p,
                "episodes_per_pair": args.episodes_per_pair,
                "horizon": args.horizon, "pre_steps": args.pre_steps,
                "stateful": int(args.stateful)
            })
            fcsv.flush()
            gc.collect()

    finally:
        fcsv.close()
        ec.close_all()

    print(f"\nAppended results → {out_path}")
    print("You can re-run the same command; it will resume and skip completed pairs (per stateful mode).")

if __name__ == "__main__":
    main()
