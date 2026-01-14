#!/usr/bin/env python3
# scripts/compute_meg.py
"""
Compute a MEG-style goal-directedness score for DeepLTL agents.

For each goal φ:
  p_pi   = Pr[success within H | start ~ reset, policy = Agent]
  p_rand = Pr[success within H | start ~ reset, policy = uniform-random]
  GD(φ)  = log p_pi  −  log p_rand

Notes
- For reach-style goals we proxy "success" as "saw the target proposition at least once".
- Some sequence searchers don't like bare 'F x'; we map 'F x' -> 'FG x' for planning only
  but still score success as 'saw x' (so semantics are reach-like).
"""

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch

# ─────────── repo imports (match your probing script) ───────────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

# Gym/Gymnasium compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces as gspaces
    GYMN = True
except Exception:
    import gym  # type: ignore
    from gym import spaces as gspaces  # type: ignore
    GYMN = False


# -------- helpers: parsing, acceptance, IO --------
def prop_from_formula(phi: str) -> str | None:
    """Grab the last alphabetic token, e.g. 'FG green' -> 'green'."""
    m = re.findall(r"[A-Za-z]+", phi)
    return m[-1] if m else None


def goal_for_search(phi: str) -> str:
    """
    Map formulas the searcher might dislike to equivalents it handles.
    - 'F x' → plan with 'FG x' (while success is still 'saw x' at least once)
    """
    m = re.match(r"^\s*F\s+([A-Za-z]+)\s*$", phi)
    if m:
        return f"FG {m.group(1)}"
    return phi


def deep_accept(info, reward, done):
    """
    Recursively scan dict/list/tuple for any key containing 'accept' or 'satisf'
    (case-insensitive) that is truthy. Also fall back to positive terminal reward.
    """
    def any_truthy_accept(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if ("accept" in lk) or ("satisf" in lk):
                    try:
                        if bool(v):
                            return True
                    except Exception:
                        pass
                if any_truthy_accept(v):
                    return True
            return False
        elif isinstance(x, (list, tuple)):
            return any(any_truthy_accept(v) for v in x)
        else:
            return False

    if any_truthy_accept(info):
        return True
    # conservative fallback: positive terminal reward as success
    if done and isinstance(reward, (int, float)) and reward > 0:
        return True
    return False


# -------- shape-agnostic reset/step and action coercion --------
def step_unpack(ret):
    if len(ret) == 5:
        obs, rew, terminated, truncated, info = ret
        return obs, rew, bool(terminated or truncated), info
    elif len(ret) == 4:
        obs, rew, done, info = ret
        return obs, rew, bool(done), info
    else:
        raise RuntimeError(f"Unexpected env.step return of length {len(ret)}")


def reset_unpack(env, **kwargs):
    out = env.reset(**kwargs)
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1]
    return out, {}


def coerce_action(act, action_space):
    """Make `act` conform to the env's action_space."""
    if isinstance(action_space, gspaces.Box):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size == 1 and need > 1:
            a = np.repeat(a, need)
        if a.size != need:
            raise ValueError(f"Action size {a.size} != expected {need} for Box{action_space.shape}")
        a = np.clip(a, action_space.low, action_space.high)
        return a.reshape(action_space.shape)
    elif isinstance(action_space, gspaces.Discrete):
        a = int(np.asarray(act).ravel()[0]) if isinstance(act, (np.ndarray, list, tuple)) else int(act)
        if not (0 <= a < action_space.n):
            raise ValueError(f"Discrete action {a} outside [0,{action_space.n})")
        return a
    elif isinstance(action_space, gspaces.MultiDiscrete):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        if a.size != action_space.nvec.size:
            raise ValueError(f"MultiDiscrete size {a.size} != {action_space.nvec.size}")
        return a
    elif isinstance(action_space, gspaces.MultiBinary):
        a = np.asarray(act, dtype=action_space.dtype).ravel()
        need = int(np.prod(action_space.shape))
        if a.size != need:
            raise ValueError(f"MultiBinary size {a.size} != {need}")
        return a.reshape(action_space.shape)
    else:
        return act


# -------- rollout runners --------
def run_episode_with_agent(env, agent, horizon, deterministic, seed=None, prop_success: str | None = None) -> bool:
    obs, info = reset_unpack(env, seed=seed)
    agent.reset()
    seen_prop = False

    for _ in range(horizon):
        with torch.no_grad():
            act = agent.get_action(obs, {}, deterministic=deterministic)
        act = coerce_action(act, env.action_space)
        obs, rew, done, info = step_unpack(env.step(act))

        if prop_success and "propositions" in info:
            if any(str(p) == prop_success for p in info["propositions"]):
                seen_prop = True

        # accept via LDBA OR via prop_success
        if deep_accept(info, rew, done) or (prop_success and seen_prop):
            return True
        if done:
            return False
    return bool(prop_success and seen_prop)


def run_episode_random(env, horizon: int, seed=None, prop_success: str | None = None) -> bool:
    obs, info = reset_unpack(env, seed=seed)
    seen_prop = False

    for _ in range(horizon):
        act = env.action_space.sample()
        act = coerce_action(act, env.action_space)
        obs, rew, done, info = step_unpack(env.step(act))

        # one-time debug on first random episode
        if hasattr(run_episode_random, "_dbg") is False:
            try:
                interesting = {k: v for k, v in info.items() if any(s in k.lower() for s in ["ltl","ldba","accept","satisf","prop"])}
                print("[debug] info keys:", list(info.keys()))
                print("[debug] accept-ish subdict:", interesting)
            except Exception:
                pass
            run_episode_random._dbg = True  # set once

        if prop_success and "propositions" in info:
            if any(str(p) == prop_success for p in info["propositions"]):
                seen_prop = True

        if deep_accept(info, rew, done) or (prop_success and seen_prop):
            return True
        if done:
            return False
    return bool(prop_success and seen_prop)


def estimate_accept_successes(make_env_fn, agent, episodes: int, horizon: int, deterministic: bool,
                              base_seed: int, policy: str, prop_success: str | None = None, progress: bool = True):
    successes = 0
    for i in range(episodes):
        if progress and (i + 1) % max(1, episodes // 8) == 0:
            print(f"    [{policy}] {i + 1}/{episodes} episodes...", flush=True)
        env = make_env_fn()
        try:
            if policy == "agent":
                ok = run_episode_with_agent(env, agent, horizon, deterministic=deterministic,
                                            seed=base_seed + 17 * i, prop_success=prop_success)
            elif policy == "random":
                ok = run_episode_random(env, horizon, seed=base_seed + 17 * i + 7, prop_success=prop_success)
            else:
                raise ValueError(policy)
            successes += int(ok)
        finally:
            env.close()
    return successes


def smoothed_rate(successes: int, n: int) -> float:
    """Jeffreys prior Beta(1/2,1/2) smoothing."""
    return (successes + 0.5) / (n + 1.0)


def print_results_table(results, episodes, horizon):
    """Print a formatted table of all results."""
    print("\n" + "="*100)
    print("GOAL DIRECTEDNESS EVALUATION RESULTS")
    print("="*100)
    
    # Table header
    header = f"{'Goal':<15} {'p_pi':<10} {'p_rand':<10} {'GD':<10} {'Success Rate':<20} {'Raw Successes':<20}"
    print(header)
    print("-" * 100)
    
    # Table rows
    for goal, (p_pi, p_rand, gd, succ_pi, succ_rand) in results.items():
        success_rate_pi = f"{succ_pi}/{episodes}"
        success_rate_rand = f"{succ_rand}/{episodes}"
        raw_successes = f"Agent: {succ_pi}, Random: {succ_rand}"
        
        row = f"{goal:<15} {p_pi:<10.3f} {p_rand:<10.3f} {gd:<+10.3f} {success_rate_pi:<20} {raw_successes:<20}"
        print(row)
    
    print("-" * 100)
    
    # Summary statistics
    mean_gd = float(np.mean([v[2] for v in results.values()]))
    mean_p_pi = float(np.mean([v[0] for v in results.values()]))
    mean_p_rand = float(np.mean([v[1] for v in results.values()]))
    
    print(f"{'MEAN':<15} {mean_p_pi:<10.3f} {mean_p_rand:<10.3f} {mean_gd:<+10.3f}")
    print("="*100)
    print(f"Configuration: {episodes} episodes, horizon={horizon}")
    print("="*100)


# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--goals", type=str, default="FG blue,FG green,FG yellow,FG magenta")
    ap.add_argument("--prop_success", type=str, default="", help="If set (e.g. 'green'), success = saw this proposition at least once. If empty, infer from formula.")
    ap.add_argument("--out_csv", type=str, default="", help="Optional CSV path to append per-goal results.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Build dummy env + model (same as your probe flow)
    first_goal = args.goals.split(",")[0].strip()
    dummy = make_env(args.env_id, FixedSampler.partial(goal_for_search(first_goal)), sequence=False)
    cfg = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
    status = store.load_training_status(map_location="cpu")
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
    agent = Agent(model, planner, propositions=props)

    # Evaluate per goal
    results = {}
    goals = [g.strip() for g in args.goals.split(",") if g.strip()]
    print("\nEstimating acceptance probabilities:")
    for g in goals:
        env_goal = goal_for_search(g)
        prop = args.prop_success or prop_from_formula(g)

        def env_factory(goal=env_goal):
            return make_env(args.env_id, FixedSampler.partial(goal), sequence=False)

        succ_pi = estimate_accept_successes(env_factory, agent, args.episodes, args.horizon,
                                            args.deterministic, base_seed=args.seed,
                                            policy="agent", prop_success=prop, progress=True)
        succ_rd = estimate_accept_successes(env_factory, agent, args.episodes, args.horizon,
                                            args.deterministic, base_seed=args.seed + 10_000,
                                            policy="random", prop_success=prop, progress=True)

        p_pi = smoothed_rate(succ_pi, args.episodes)
        p_rd = smoothed_rate(succ_rd, args.episodes)
        gd = math.log(p_pi) - math.log(p_rd + 1e-12)

        results[g] = (p_pi, p_rd, gd, succ_pi, succ_rd)
        print(f"  {g:>10s} | p_pi={p_pi:.3f}  p_rand={p_rd:.3f}  GD={gd:+.3f}  (succ {succ_pi}/{args.episodes}, {succ_rd}/{args.episodes})")

    if results:
        # Print the formatted table
        print_results_table(results, args.episodes, args.horizon)
        
        # Keep the original summary output
        mean_gd = float(np.mean([v[2] for v in results.values()]))
        mean_p_pi = float(np.mean([v[0] for v in results.values()]))
        mean_p_rand = float(np.mean([v[1] for v in results.values()]))
        print(f"\nSummary (macro-average over listed goals):")
        print(f"  p_pi={mean_p_pi:.3f}  p_rand={mean_p_rand:.3f}  GD={mean_gd:+.3f}")

        if args.out_csv:
            rows = []
            ts = datetime.now().isoformat(timespec="seconds")
            for g, (ppi, prd, gd, s_pi, s_rd) in results.items():
                rows.append({
                    "timestamp": ts, "env_id": args.env_id, "exp": args.exp, "goal": g,
                    "episodes": args.episodes, "horizon": args.horizon,
                    "p_pi": ppi, "p_rand": prd, "GD": gd,
                    "succ_pi": s_pi, "succ_rand": s_rd
                })
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if f.tell() == 0:
                    w.writeheader()
                w.writerows(rows)
            print(f"Saved CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
