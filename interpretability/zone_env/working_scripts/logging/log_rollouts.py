#!/usr/bin/env python3
"""
Log stateful goal-switch rollouts with per-step hidden states.

What gets logged (per step):
  - chain_id, step_idx, link_idx (0 = pre-phase to reach s0, then 1..depth)
  - color (active target color at this step)
  - h: hidden state vector (from model.ltl_net.rnn)
  - a: action taken
  - success_step: True if target prop was hit on this step
  - propositions: semicolon-joined set from info["propositions"]
  - accepting: info.get('accepting', False)
  - done: whether env terminated/truncated
  - world_seed: seed used for this chain

Outputs:
  - Parquet table with above fields
  - JSONL metadata file with run args + summary

Example:
  python interpretability/working_scripts/log_rollouts.py \
    --env_id PointLtl2-v0 --exp big_test --seed 0 --deterministic \
    --colors "green,blue,yellow,magenta" \
    --num_chains 20 --depth 3 --pre_steps 200 --per_link_steps 200 \
    --out_parquet interpretability/working_scripts/rollouts_stateful.parquet
"""
import argparse, sys, re, json, gzip, copy
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime

# ---- repo imports ----
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

def props_str(info) -> str:
    if "propositions" in info and isinstance(info["propositions"], (set, list, tuple)):
        return ";".join(sorted(map(str, info["propositions"])))
    return ""

def extract_xy(env=None, obs=None, info=None):
    """
    Best-effort (x,y) as plain floats every step.
    Priority:
      1) info[...] (common envs put agent position here)
      2) obs['features'][0:2]
      else -> (nan, nan)
    """
    # 1) Try common keys in info
    if isinstance(info, dict):
        for k in ("agent_pos", "position", "pos", "agent_position", "state_pos"):
            if k in info:
                v = info[k]
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    try:
                        x, y = float(v[0]), float(v[1])
                        if np.isfinite(x) and np.isfinite(y):
                            return x, y
                    except Exception:
                        pass
                if isinstance(v, dict) and "x" in v and "y" in v:
                    try:
                        x, y = float(v["x"]), float(v["y"])
                        if np.isfinite(x) and np.isfinite(y):
                            return x, y
                    except Exception:
                        pass

    # 2) Try obs['features'] first two entries
    if isinstance(obs, dict) and "features" in obs:
        try:
            f = np.asarray(obs["features"]).ravel()
            if f.size >= 2:
                x, y = float(f[0]), float(f[1])
                if np.isfinite(x) and np.isfinite(y):
                    return x, y
        except Exception:
            pass

    # 3) Give up
    return float("nan"), float("nan")


def get_obs_features(obs):
    """Consistently extract obs features as a flat list or None."""
    if isinstance(obs, dict) and "features" in obs:
        try:
            return np.asarray(obs["features"]).ravel().tolist()
        except Exception:
            return None
    return None


def to_list_or_none(x):
    try:
        arr = np.asarray(x).ravel()
        return arr.tolist()
    except Exception:
        return None


def _action_dim(action_space):
    from gymnasium import spaces as gspaces
    if isinstance(action_space, gspaces.Discrete):
        return int(action_space.n)
    if isinstance(action_space, gspaces.MultiDiscrete):
        return int(action_space.nvec.size)
    if isinstance(action_space, gspaces.MultiBinary):
        return int(np.prod(action_space.shape))
    if isinstance(action_space, gspaces.Box):
        return int(np.prod(action_space.shape))
    # fallback
    return None

def find_action_head_module(model: nn.Module, action_space) -> tuple[nn.Module, str]:
    """
    Heuristic: pick the *last* nn.Linear whose out_features matches action_dim.
    Prefer names containing 'actor'/'policy'/'pi'/'head'/'logits'/'mean'.
    Falls back to the last nn.Linear if nothing matches.
    """
    adim = _action_dim(action_space)
    candidates = []
    all_linears = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            all_linears.append((name, m))
            if adim is None or getattr(m, "out_features", None) == adim:
                candidates.append((name, m))

    def pref_score(name: str) -> int:
        n = name.lower()
        score = 0
        for key in ("actor", "policy", "pi", "head", "logits", "mean"):
            if key in n:
                score += 1
        return score

    if candidates:
        candidates.sort(key=lambda nm: (pref_score(nm[0]), len(nm[0])), reverse=True)
        return candidates[0][1], candidates[0][0]
    # fallback: last linear by definition order
    if all_linears:
        return all_linears[-1][1], all_linears[-1][0]
    raise RuntimeError("Could not find any nn.Linear to hook as action head.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="PointLtl2-v0")
    ap.add_argument("--exp", default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--num_chains", type=int, default=20)
    ap.add_argument("--depth", type=int, default=3, help="number of goal switches (links); chain length = depth+1 (including s0)")
    ap.add_argument("--pre_steps", type=int, default=200)
    ap.add_argument("--per_link_steps", type=int, default=200)
    ap.add_argument("--out_parquet", type=str, default="interpretability/working_scripts/rollouts_stateful.parquet")
    ap.add_argument("--meta_jsonl", type=str, default="interpretability/working_scripts/rollouts_stateful.meta.jsonl.gz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Build env + agent
    first_goal = f"FG {args.colors.split(',')[0].strip()}"
    dummy = make_env(args.env_id, FixedSampler.partial(goal_for_search(first_goal)), sequence=False)
    cfg   = model_configs[args.env_id]
    store = ModelStore(args.env_id, args.exp, args.seed); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    props = set(dummy.get_propositions())
    action_space = dummy.action_space
    hidden_sz = model.ltl_net.rnn.hidden_size
    print("Observation shape:", dummy.observation_space.shape if hasattr(dummy.observation_space, "shape") else None)
    print("Num LTLNet params:", sum(p.numel() for p in model.parameters()))
    print("GRU hidden       :", hidden_sz)
    dummy.close()

    planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
    agent   = Agent(model, planner, propositions=props)

    # Hook to capture the last GRU hidden used anywhere (we keep the most recent)
    last_hidden = {"h": None, "n_events": 0}

    def rnn_hook(_, __, out):
        try:
            h = out[1]  # GRU returns (output, h_n)
            if isinstance(h, (tuple, list)):
                h = h[-1]
            if isinstance(h, torch.Tensor):
                h = h.detach().cpu().numpy().squeeze()
            last_hidden["h"] = h
            last_hidden["n_events"] += 1
        except Exception:
            try:
                if isinstance(out, (tuple, list)):
                    h = out[-1]
                    if isinstance(h, torch.Tensor):
                        h = h.detach().cpu().numpy().squeeze()
                    last_hidden["h"] = h
                    last_hidden["n_events"] += 1
            except Exception:
                pass

    handle = model.ltl_net.rnn.register_forward_hook(rnn_hook)

    last_actor = {"penult": None, "n_events": 0}

    head_module, head_name = find_action_head_module(model, action_space)

    def actor_head_hook(module, inputs, output):
        # inputs[0] is the tensor fed into the final action projection
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            last_actor["penult"] = x.detach().cpu().numpy().ravel()
            last_actor["n_events"] += 1

    actor_handle = head_module.register_forward_hook(actor_head_hook)
    print(f"[actor-hook] attached to: {head_name}")

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    print("\nLogging stateful chains...")
    rows: List[Dict[str, Any]] = []

    for chain_id in range(args.num_chains):
        # sample chain colors: s0 then args.depth switches (total length = depth+1)
        s0 = colors[rng.integers(len(colors))]
        chain = [s0]
        for _ in range(args.depth):
            choices = [c for c in colors if c != chain[-1]]
            chain.append(choices[rng.integers(len(choices))])

        env = make_env(args.env_id, FixedSampler.partial(goal_for_search(f"FG {chain[0]}")), sequence=False)
        try:
            world_seed = int(args.seed + 97*chain_id)
            obs, info = reset_unpack(env, seed=world_seed)

            # Reach s0 (pre-phase, link_idx=0). We do NOT mark this as a "switch".
            if hasattr(env, "set_goal"):
                env.set_goal(f"FG {chain[0]}")
            agent.reset()

            steps = 0
            hit = False
            while steps < args.pre_steps and not hit:
                last_hidden["h"] = None
                last_actor["penult"] = None
                with torch.no_grad():
                    act = agent.get_action(obs, {}, deterministic=args.deterministic)
                act = coerce_action(act, env.action_space)
                obs, rew, done, info = step_unpack(env.step(act))
                steps += 1
                h = None if last_hidden["h"] is None else last_hidden["h"].copy()
                actor_penult = None if last_actor["penult"] is None else last_actor["penult"].copy()

                rows.append({
                    "chain_id": chain_id,
                    "world_seed": world_seed,
                    "step_idx": steps,
                    "link_idx": 0,
                    "color": chain[0],                                 # active target for this link
                    "goal_text": f"FG {chain[0]}",                      # write explicitly
                    "switch_flag": 0,                                   # not a switch (start-of-episode)
                    "h": None if h is None else h.tolist(),
                    "actor_penult": None if actor_penult is None else actor_penult.tolist(),
                    "a": np.asarray(act).ravel().tolist(),
                    "obs_features": to_list_or_none(obs.get("features") if isinstance(obs, dict) else None),
                    "pos_x": extract_xy(env=env, obs=obs, info=info)[0],
                    "pos_y": extract_xy(env=env, obs=obs, info=info)[1],
                    "success_step": bool(saw_prop(info, chain[0])),
                    "propositions": props_str(info),
                    "accepting": bool(info.get("accepting", False)),
                    "done": bool(done),
                })
                if saw_prop(info, chain[0]):
                    hit = True
                if done:
                    break

            # Successive links (link_idx = 1..depth). Mark the first step after each switch.
            for k in range(1, len(chain)):
                if hasattr(env, "set_goal"):
                    env.set_goal(f"FG {chain[k]}")
                agent.reset()

                first_step_of_link = True
                steps_on_link = 0
                hit = False
                while steps_on_link < args.per_link_steps and not hit:
                    last_hidden["h"] = None
                    last_actor["penult"] = None
                    with torch.no_grad():
                        act = agent.get_action(obs, {}, deterministic=args.deterministic)
                    act = coerce_action(act, env.action_space)
                    obs, rew, done, info = step_unpack(env.step(act))
                    steps += 1
                    steps_on_link += 1
                    h = None if last_hidden["h"] is None else last_hidden["h"].copy()
                    actor_penult = None if last_actor["penult"] is None else last_actor["penult"].copy()
                    rows.append({
                        "chain_id": chain_id,
                        "world_seed": world_seed,
                        "step_idx": steps,
                        "link_idx": k,
                        "color": chain[k],                               # active target for this link
                        "obs_features": np.asarray(obs["features"]).ravel().tolist(),
                        "goal_text": f"FG {chain[k]}",                   # write explicitly (do not rely on obs['goal'])
                        "switch_flag": int(first_step_of_link),          # <-- mark switch boundary
                        "h": None if h is None else h.tolist(),
                        "actor_penult": None if actor_penult is None else actor_penult.tolist(),
                        "a": np.asarray(act).ravel().tolist(),
                        "pos_x": extract_xy(env=env, obs=obs, info=info)[0],
                        "pos_y": extract_xy(env=env, obs=obs, info=info)[1],
                        "success_step": bool(saw_prop(info, chain[k])),
                        "propositions": props_str(info),
                        "accepting": bool(info.get("accepting", False)),
                        "done": bool(done),
                    })
                    first_step_of_link = False

                    if saw_prop(info, chain[k]):
                        hit = True
                    if done:
                        break

        finally:
            env.close()

        print(f"  chain {chain_id+1}/{args.num_chains}  sequence={chain}")

    handle.remove()
    actor_handle.remove()
    print(f"Hook events captured: GRU={last_hidden['n_events']}  ACTOR={last_actor['n_events']}")

    non_null = sum(1 for _ in filter(lambda r: isinstance(r.get('h'), list) and len(r['h'])>0, rows))
    print(f"Rows with non-empty h: {non_null} / {len(rows)}")

    # Save
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Basic sanity columns
    df["h_len"] = df["h"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["a_len"] = df["a"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Debug: how many switch rows did we record?
    num_switch = int(df.get("switch_flag", pd.Series([0]*len(df))).sum())
    print(f"\nSwitch rows recorded: {num_switch}")
    if num_switch > 0:
        print(df.loc[df["switch_flag"] == 1, ["chain_id","step_idx","link_idx","goal_text"]].head())

    print("\nSummary:")
    print(df.groupby("link_idx")["success_step"].mean().rename("per-link hit-rate"))

    print(f"\nWriting Parquet → {out_path}")
    df.to_parquet(out_path, index=False)

    # meta
    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "args": vars(args),
        "num_rows": int(len(df)),
        "hidden_size": int(hidden_sz),
        "env_id": args.env_id,
        "exp": args.exp,
        "seed": args.seed,
    }
    meta_path = Path(args.meta_jsonl)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(meta_path, "at", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    print(f"Wrote meta → {meta_path}")


if __name__ == "__main__":
    main()
