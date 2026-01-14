#!/usr/bin/env python3
"""
Evaluate recovered macro world model vs. goal depth.

Supports --stateful:
- If set, each chain runs inside ONE env instance without resetting between links.
- We call env.set_goal("FG <color>") + agent.reset() before each link.
- This preserves physics/poses/momentum across goal switches.

Writes optional per-depth CSV; can resume.
"""

import argparse, sys, csv, re, gc, random, copy
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from collections import Counter

# repo imports
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent

try:
    from gymnasium import spaces as gspaces
except Exception:
    from gym import spaces as gspaces  # type: ignore

torch.set_grad_enabled(False)

# --- helpers ---
def goal_for_search(phi: str) -> str:
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

def saw_prop(info, name: str) -> bool:
    if "propositions" in info:
        return any(str(p) == name for p in info["propositions"])
    return False

def override_goal_in_obs(obs, color: str):
    """Return a shallow-copied obs with goal text set to 'FG <color>'."""
    if not isinstance(obs, dict):
        return obs
    obs2 = dict(obs)
    obs2["goal"] = f"FG {color}"
    return obs2


# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--model_csv", type=str, required=True)
    ap.add_argument("--depths", type=str, default="1,2,3,4,5")
    ap.add_argument("--sequences_per_depth", type=int, default=40)
    ap.add_argument("--pre_steps", type=int, default=200)
    ap.add_argument("--per_step_budget", type=int, default=200)
    ap.add_argument("--stateful", action="store_true", help="Keep one env per chain and swap LDBA goal via env.set_goal.")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--log_fail_positions", action="store_true", help="Print histogram of first failing link index.")
    ap.add_argument("--no_auto_flush", action="store_true",
                help="Disable Agent's auto-flush-on-goal-change (for A/B comparison).")


    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # load macro model
    rows = list(csv.DictReader(open(args.model_csv, "r")))
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    idx = {c:i for i,c in enumerate(colors)}
    P = np.zeros((len(colors), len(colors)), dtype=float)
    for r in rows:
        s, t = r["src"], r["dst"]
        if s in idx and t in idx:
            P[idx[s], idx[t]] = float(r["p_hat"])

    # agent
    first_goal = f"FG {colors[0]}"
    dummy = make_env(args.env_id, FixedSampler.partial(goal_for_search(first_goal)), sequence=False)
    cfg = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    props = set(dummy.get_propositions())
    dummy.close()

    planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
    agent   = Agent(model, planner, propositions=props)
    # toggle planner auto-flush on goal change
    agent.auto_flush_on_goal = (not args.no_auto_flush)

    # CSV resume (by depth)
    done_depths = set()
    if args.out_csv and Path(args.out_csv).exists():
        with open(args.out_csv, "r") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if r.get("env_id")==args.env_id and r.get("exp")==args.exp and r.get("colors")==",".join(colors) and r.get("stateful")==str(int(args.stateful)):
                    done_depths.add(int(r["depth"]))

    fcsv = None; writer = None
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        new_file = not Path(args.out_csv).exists()
        fcsv = open(args.out_csv, "a", newline="")
        writer = csv.DictWriter(fcsv, fieldnames=[
            "env_id","exp","seed","colors","depth","N",
            "predicted_mean","empirical_mean","MAE",
            "pre_steps","per_step_budget","num_loops","deterministic","stateful"
        ])
        if new_file:
            writer.writeheader()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]

    print("\nEvaluating world-model vs. chain depth:")
    try:
        for n in depths:
            if n in done_depths:
                print(f"  depth={n:2d} | (skipping; already in {args.out_csv})")
                continue

            # deterministic sampling per depth
            local_rng = np.random.default_rng(args.seed + n*1009)
            trials = []
            for _ in range(args.sequences_per_depth):
                s0 = colors[local_rng.integers(len(colors))]
                chain = [s0]
                for _k in range(n):
                    choices = [c for c in colors if c != chain[-1]]
                    chain.append(choices[local_rng.integers(len(choices))])
                trials.append(chain)

            # predicted success
            preds = []
            for ch in trials:
                prob = 1.0
                for k in range(1, len(ch)):
                    i, j = idx[ch[k-1]], idx[ch[k]]
                    prob *= P[i, j]
                preds.append(prob)
            pred_mean = float(np.mean(preds))

            # empirical success
            succ = 0
            fails_at = []
            for i, ch in enumerate(trials, 1):
                if not args.stateful:
                    # reset-per-link behavior (kept for comparison)
                    env = make_env(args.env_id, FixedSampler.partial(goal_for_search(f"FG {ch[0]}")), sequence=False)
                    try:
                        obs, info = reset_unpack(env, seed=args.seed+17*i)
                        agent.reset()
                        ok_chain = True
                        # reach s0
                        ok0 = False
                        for _ in range(args.pre_steps):
                            obs2 = override_goal_in_obs(obs, ch[0])
                            with torch.no_grad():
                                act = agent.get_action(obs2, {}, deterministic=args.deterministic)
                            act = coerce_action(act, env.action_space)
                            obs, rew, done, info = step_unpack(env.step(act))
                            if saw_prop(info, ch[0]): ok0=True; break
                            if done: break
                        if not ok0:
                            ok_chain = False
                            fails_at.append(0)

                        # links
                        for k in range(1, len(ch)):
                            if not ok_chain: break
                            envk = make_env(args.env_id, FixedSampler.partial(goal_for_search(f"FG {ch[k]}")), sequence=False)
                            try:
                                obs, info = reset_unpack(envk, seed=args.seed+23*i+7*k)
                                agent.reset()
                                ok = False
                                for _ in range(args.per_step_budget):
                                    obs2 = override_goal_in_obs(obs, ch[k])
                                    with torch.no_grad():
                                        act = agent.get_action(obs2, {}, deterministic=args.deterministic)
                                    act = coerce_action(act, envk.action_space)
                                    obs, rew, done, info = step_unpack(envk.step(act))
                                    if saw_prop(info, ch[k]): ok=True; break
                                    if done: break
                                if not ok:
                                    fails_at.append(k)
                                ok_chain = ok_chain and ok
                            finally:
                                envk.close()
                        succ += int(ok_chain)
                    finally:
                        env.close()

                else:
                    # stateful chaining inside a single env (no resets between links)
                    env = make_env(args.env_id, FixedSampler.partial(goal_for_search(f"FG {ch[0]}")), sequence=False)
                    try:
                        obs, info = reset_unpack(env, seed=args.seed+17*i)

                        # first node in SAME env
                        if hasattr(env, "set_goal"):
                            env.set_goal(f"FG {ch[0]}")
                        agent.reset()

                        ok_chain = True
                        ok0 = False
                        for _ in range(args.pre_steps):
                            # (obs returned by step already contains updated LDBA via wrapper)
                            with torch.no_grad():
                                act = agent.get_action(obs, {}, deterministic=args.deterministic)
                            act = coerce_action(act, env.action_space)
                            obs, rew, done, info = step_unpack(env.step(act))
                            if saw_prop(info, ch[0]): ok0=True; break
                            if done: break
                        if not ok0:
                            ok_chain = False
                            fails_at.append(0)

                        # successive links use SAME env; swap goal + replan
                        for k in range(1, len(ch)):
                            if not ok_chain: break
                            if hasattr(env, "set_goal"):
                                env.set_goal(f"FG {ch[k]}")
                            agent.reset()

                            ok = False
                            for _ in range(args.per_step_budget):
                                with torch.no_grad():
                                    act = agent.get_action(obs, {}, deterministic=args.deterministic)
                                act = coerce_action(act, env.action_space)
                                obs, rew, done, info = step_unpack(env.step(act))
                                if saw_prop(info, ch[k]): ok=True; break
                                if done: break
                            if not ok:
                                fails_at.append(k)
                            ok_chain = ok_chain and ok

                        succ += int(ok_chain)

                    finally:
                        env.close()

                if i % max(1, args.sequences_per_depth//8) == 0:
                    print(f"    depth={n}  progress {i}/{args.sequences_per_depth}", flush=True)
                if i % 10 == 0:
                    gc.collect()

            emp_mean = succ / max(1, len(trials))
            mae = abs(emp_mean - pred_mean)
            print(f"  depth={n:2d} | predicted={pred_mean:.3f}  empirical={emp_mean:.3f}  MAE={mae:.3f}  (N={len(trials)})")
            if args.log_fail_positions and fails_at:
                print("  Failure locations (first failing link index; 0 = couldn't reach s0):", Counter(fails_at))

            if writer:
                writer.writerow({
                    "env_id": args.env_id, "exp": args.exp, "seed": args.seed,
                    "colors": ",".join(colors),
                    "depth": n, "N": len(trials),
                    "predicted_mean": pred_mean, "empirical_mean": emp_mean, "MAE": mae,
                    "pre_steps": args.pre_steps, "per_step_budget": args.per_step_budget,
                    "num_loops": args.num_loops, "deterministic": int(args.deterministic),
                    "stateful": int(args.stateful),
                })
                fcsv.flush()

    finally:
        if fcsv: fcsv.close()

if __name__ == "__main__":
    main()
