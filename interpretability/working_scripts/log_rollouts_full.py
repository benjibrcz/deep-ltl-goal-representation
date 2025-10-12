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

def _np2(x):
    a = np.asarray(x, dtype=float).ravel()
    return a[:2] if a.size >= 2 else None

def discover_zone_centers(env) -> dict[str, np.ndarray] | None:
    """
    Best-effort spelunking for zone centers in Safety-Gym*/custom envs.
    Returns {color: np.array([x,y])} or None.
    """
    cand = []

    # unwrap chain
    e = env
    for _ in range(5):
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
        elif hasattr(e, "env"):
            e = e.env
        else:
            break

    # common spots
    for attr in ("zone_centers", "zones", "objects", "world", "task", "scene"):
        if hasattr(e, attr):
            cand.append(getattr(e, attr))

    # dive one level
    more = []
    for c in cand:
        if isinstance(c, dict):
            for v in c.values():
                more.append(v)
        else:
            for name in ("zones", "zone_centers", "objects"):
                if hasattr(c, name):
                    more.append(getattr(c, name))
    cand.extend(more)

    out = {}
    def maybe_add(name, val):
        p = _np2(val)
        if p is not None and np.all(np.isfinite(p)):
            out[str(name)] = np.asarray(p, dtype=float)

    # try dicts like {'green': {'center':[x,y], ...}}
    for c in cand:
        if isinstance(c, dict):
            # direct color -> center
            ok = True
            for k,v in c.items():
                if isinstance(v, (list, tuple, np.ndarray)):
                    maybe_add(k, v)
                elif isinstance(v, dict) and "center" in v:
                    maybe_add(k, v["center"])
                else:
                    ok = False
            if ok and out:
                return out

    # try objects with attributes
    for c in cand:
        if hasattr(c, "__dict__"):
            d = c.__dict__
            # nested dict by color
            if any(isinstance(v, dict) for v in d.values()):
                for k,v in d.items():
                    if isinstance(v, dict) and "center" in v:
                        maybe_add(k, v["center"])
                if out:
                    return out
            # flat center with a color field
            if "center" in d and ("color" in d or "name" in d):
                maybe_add(d.get("color", d.get("name", "zone")), d["center"])
                return out if out else None
    return out or None

def maybe_add_zone_centers(env, info: dict):
    try:
        if "zone_centers" not in info or not info["zone_centers"]:
            zc = discover_zone_centers(env)
            if zc:
                # cast to plain lists for JSON/Parquet safety
                info["zone_centers"] = {k: v.astype(float).tolist() for k,v in zc.items()}
    except Exception:
        pass


# --- NEW: helpers for NPZ logging & labels ---

def flatten_features(obs):
    """
    Return a 1D float vector for the current observation.
    Supports either a flat numpy array observation OR a dict with 'features'.
    """
    # Case A: the env already returns a flat vector (e.g., shape (80,))
    if isinstance(obs, np.ndarray):
        try:
            return np.asarray(obs, dtype=np.float32).ravel()
        except Exception:
            return None

    # Case B: dict observations with an inner 'features' vector
    if isinstance(obs, dict):
        for key in ("features", "obs", "state", "flat"):
            if key in obs:
                try:
                    return np.asarray(obs[key], dtype=np.float32).ravel()
                except Exception:
                    pass
    return None


def build_prop_index(props: set[str]) -> dict[str,int]:
    return {p:i for i,p in enumerate(sorted(props))}

def ap_vec_from_info(info: dict, prop_index: dict[str,int]) -> np.ndarray:
    """Binary AP vector for current step from info['propositions']."""
    v = np.zeros(len(prop_index), dtype=np.float32)
    if "propositions" in info:
        for p in info["propositions"]:
            i = prop_index.get(str(p))
            if i is not None: v[i] = 1.0
    return v

def color_next_positives(goal_text: str, prop_index: dict[str,int]) -> np.ndarray:
    """
    For FG <color> links, 'next positives' is just that color one-hot.
    goal_text is like 'FG green' from this logger.
    """
    v = np.zeros(len(prop_index), dtype=np.float32)
    try:
        color = goal_text.split()[-1].strip()
        i = prop_index.get(color)
        if i is not None: v[i] = 1.0
    except Exception:
        pass
    return v

def get_zone_center_from_info(info: dict) -> dict[str, np.ndarray] | None:
    """
    Best-effort: look for any zone center hints in info.
    Expected formats (we try several):
      - info['zone_centers'] = {'green': [x,y], ...}
      - info['zones'] = {'green': {'center':[x,y]}, ...}
      - info['target_zone_center'] = [x,y]
    Returns: dict color->np.array([x,y]) or None.
    """
    try:
        if "zone_centers" in info and isinstance(info["zone_centers"], dict):
            return {k: np.asarray(v, dtype=float).ravel()[:2] for k,v in info["zone_centers"].items()}
        if "zones" in info and isinstance(info["zones"], dict):
            out = {}
            for k,v in info["zones"].items():
                if isinstance(v, dict) and "center" in v:
                    out[k] = np.asarray(v["center"], dtype=float).ravel()[:2]
            if out: return out
    except Exception:
        pass
    return None

def vector_to_goal_explicit(color: str, agent_xy: tuple[float,float], zone_positions: dict) -> np.ndarray:
    """
    Compute agent -> current goal-color zone center using explicit zone positions.
    Uses the nearest zone center for the target color.
    """
    ax, ay = agent_xy
    
    # Find zone centers for this color
    color_zones = []
    for zone_key, zone_pos in zone_positions.items():
        if zone_key.startswith(f"{color}_zone"):
            color_zones.append(zone_pos)
    
    if not color_zones:
        # No zones found for this color
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    # Find the nearest zone center
    min_dist = float('inf')
    nearest_center = None
    
    for zone_pos in color_zones:
        cx, cy = zone_pos[0], zone_pos[1]
        dist = np.sqrt((ax - cx)**2 + (ay - cy)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_center = (cx, cy)
    
    if nearest_center is None:
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    cx, cy = nearest_center
    return np.array([cx - ax, cy - ay], dtype=np.float32)


def vector_to_goal(color: str, agent_xy: tuple[float,float], info: dict) -> np.ndarray:
    """
    Try to compute agent -> current goal-color zone center.
    Falls back to NaNs if we cannot find a center.
    """
    centers = get_zone_center_from_info(info)
    if centers and color in centers:
        cx, cy = centers[color]
        ax, ay = agent_xy
        return np.array([cx - ax, cy - ay], dtype=np.float32)
    # fallback: unknown -> NaNs (actor alignment probe will skip)
    return np.array([np.nan, np.nan], dtype=np.float32)


def vector_to_goal_from_env(color: str, agent_xy: tuple[float,float], env) -> np.ndarray:
    """
    Compute agent -> current goal-color zone center using environment's zone_positions.
    Uses the nearest zone center for the target color.
    """
    ax, ay = agent_xy
    
    # Get zone positions directly from environment
    if not hasattr(env, 'zone_positions') or not env.zone_positions:
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    # Find zone centers for this color
    color_zones = []
    for zone_key, zone_pos in env.zone_positions.items():
        if zone_key.startswith(f"{color.lower()}_zone"):
            color_zones.append(zone_pos)
    
    if not color_zones:
        # No zones found for this color
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    # Find the nearest zone center
    min_dist = float('inf')
    nearest_center = None
    
    for zone_pos in color_zones:
        cx, cy = zone_pos[0], zone_pos[1]
        dist = np.sqrt((ax - cx)**2 + (ay - cy)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_center = (cx, cy)
    
    if nearest_center is None:
        return np.array([np.nan, np.nan], dtype=np.float32)
    
    cx, cy = nearest_center
    return np.array([cx - ax, cy - ay], dtype=np.float32)


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
        for key in ("actor", "policy", "pi", "mu", "mean", "head", "logits"):
            if key in n: score += 1
        # discourage picking std if both match action_dim
        if "std" in n: score -= 1
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

    # --- capture env_net MLP layer outputs ---
    env_hooks_cache = {"hook_env_mlp1": None, "hook_env_mlp3": None, "n_events": 0}

    def _make_env_hook(key):
        def _hook(_module, _inputs, output):
            try:
                x = output.detach().cpu().numpy().ravel()
            except Exception:
                x = None
            env_hooks_cache[key] = x
            env_hooks_cache["n_events"] += 1
        return _hook

    env_handles = []
    try:
        # Adjust indices if your env_net MLP depth differs
        env_handles.append(model.env_net.mlp[1].register_forward_hook(_make_env_hook("hook_env_mlp1")))
        env_handles.append(model.env_net.mlp[3].register_forward_hook(_make_env_hook("hook_env_mlp3")))
        print("[env-hooks] attached: env_net.mlp[1], env_net.mlp[3]")
    except Exception as e:
        print("[warn] could not attach env_net hooks:", e)


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

    # Hook to capture critic/value network outputs and internal representations
    last_critic = {"value": None, "hook_critic_mlp0": None, "hook_critic_mlp2": None, "n_events": 0}

    def critic_hook(module, inputs, output):
        try:
            if isinstance(output, torch.Tensor):
                value = output.detach().cpu().numpy().ravel()
                last_critic["value"] = value
                last_critic["n_events"] += 1
        except Exception as e:
            pass

    def critic_mlp0_hook(module, inputs, output):
        try:
            if isinstance(output, torch.Tensor):
                hidden = output.detach().cpu().numpy().ravel()
                last_critic["hook_critic_mlp0"] = hidden
        except Exception as e:
            pass

    def critic_mlp2_hook(module, inputs, output):
        try:
            if isinstance(output, torch.Tensor):
                hidden = output.detach().cpu().numpy().ravel()
                last_critic["hook_critic_mlp2"] = hidden
        except Exception as e:
            pass

    # Hook into critic network layers
    critic_handle = model.critic.register_forward_hook(critic_hook)
    critic_mlp0_handle = model.critic[0].register_forward_hook(critic_mlp0_hook)  # First linear layer
    critic_mlp2_handle = model.critic[2].register_forward_hook(critic_mlp2_hook)  # Second linear layer
    print(f"[critic-hooks] attached to: critic, critic[0], critic[2]")

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    print("\nLogging stateful chains...")
    rows: List[Dict[str, Any]] = []

        # --- NPZ collectors (only append rows where all needed entries exist) ---
    prop_index = build_prop_index(props)
    npz_obs      = []  # s_t flattened features
    npz_next_obs = []  # s_{t+1} flattened features
    npz_action   = []  # a_t
    npz_env1     = []  # hook_env_mlp1
    npz_env3     = []  # hook_env_mlp3
    npz_ltl_h    = []  # hook_ltl_rnn_h
    npz_actor_h5 = []  # hook_actor_h5 (penultimate before action head)
    npz_next_pos = []  # next_positives (one-hot over props)
    npz_vec_next = []  # vec_to_next_pos (agent -> target color zone center, NaNs if unknown)
    npz_vec_curr = []  # vec_to_goal_t (current-step agent -> target color center)
    npz_ap       = []  # AP(s_t) one-hot
    npz_next_ap  = []  # AP(s_{t+1}) one-hot
    npz_critic   = []  # critic value estimate
    npz_critic_mlp0 = []  # critic first layer output
    npz_critic_mlp2 = []  # critic second layer output
    npz_traj_id  = []  # trajectory identifier for consecutive step alignment
    npz_reward   = []  # environment reward
    npz_done     = []  # done flag
    npz_link_idx = []  # current link index within trajectory
    npz_success_step = []  # success flag per step


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

            # ---------------- PRE-PHASE (link_idx = 0) ----------------
            current_link_idx = 0
            current_goal_color = chain[0]
            if hasattr(env, "set_goal"):
                env.set_goal(f"FG {current_goal_color}")
            agent.reset()

            steps = 0
            hit = False
            while steps < args.pre_steps and not hit:
                # Reset per-step caches so hooks reflect THIS forward pass
                last_hidden["h"] = None
                last_actor["penult"] = None
                last_critic["value"] = None
                last_critic["hook_critic_mlp0"] = None
                last_critic["hook_critic_mlp2"] = None
                env_hooks_cache["hook_env_mlp1"] = None
                env_hooks_cache["hook_env_mlp3"] = None

                # prev state labels
                prev_obs_flat = flatten_features(obs)
                prev_ap_vec   = ap_vec_from_info(info if isinstance(info, dict) else {}, prop_index)

                # forward (hooks fire)
                with torch.no_grad():
                    act = agent.get_action(obs, {}, deterministic=args.deterministic)
                act = coerce_action(act, env.action_space)

                # env step
                obs_next, rew, done, info_next = step_unpack(env.step(act))
                maybe_add_zone_centers(env, info_next)

                steps += 1

                # hooks/hidden
                h_gru  = None if last_hidden["h"] is None else np.asarray(last_hidden["h"]).ravel()
                h_act  = None if last_actor["penult"] is None else np.asarray(last_actor["penult"]).ravel()
                h_env1 = env_hooks_cache["hook_env_mlp1"]
                h_env3 = env_hooks_cache["hook_env_mlp3"]

                # next-state labels
                next_obs_flat = flatten_features(obs_next)
                next_ap_vec   = ap_vec_from_info(info_next if isinstance(info_next, dict) else {}, prop_index)

                # vec to target zone center for current (pre-step) and next (post-step)
                agent_xy_curr = extract_xy(env=env, obs=obs, info=info)
                agent_xy_next = extract_xy(env=env, obs=obs_next, info=info_next)
                vec_curr = vector_to_goal_from_env(current_goal_color, agent_xy_curr, env)
                vec_next = vector_to_goal_from_env(current_goal_color, agent_xy_next, env)

                # next-positives one-hot for FG <color>
                next_pos_oh = color_next_positives(f"FG {current_goal_color}", prop_index)

                # NPZ row (only if everything numeric is present)
                if (prev_obs_flat is not None and next_obs_flat is not None and
                    isinstance(h_gru, np.ndarray) and h_gru.size and
                    isinstance(h_act, np.ndarray) and h_act.size and
                    isinstance(h_env1, np.ndarray) and h_env1.size and
                    isinstance(h_env3, np.ndarray) and h_env3.size):

                    npz_obs.append(prev_obs_flat)
                    npz_next_obs.append(next_obs_flat)
                    npz_action.append(np.asarray(act, dtype=np.float32).ravel())
                    npz_env1.append(h_env1.astype(np.float32))
                    npz_env3.append(h_env3.astype(np.float32))
                    npz_ltl_h.append(h_gru.astype(np.float32))
                    npz_actor_h5.append(h_act.astype(np.float32))
                    npz_next_pos.append(next_pos_oh.astype(np.float32))
                    npz_vec_next.append(vec_next.astype(np.float32))
                    npz_vec_curr.append(vec_curr.astype(np.float32))
                    npz_ap.append(prev_ap_vec.astype(np.float32))
                    npz_next_ap.append(next_ap_vec.astype(np.float32))
                    
                    # Add critic value if available
                    if last_critic["value"] is not None:
                        npz_critic.append(last_critic["value"].astype(np.float32))
                    else:
                        npz_critic.append(np.array([0.0], dtype=np.float32))
                    
                    # Add critic layer activations if available
                    if last_critic["hook_critic_mlp0"] is not None:
                        npz_critic_mlp0.append(last_critic["hook_critic_mlp0"].astype(np.float32))
                    else:
                        npz_critic_mlp0.append(np.zeros(64, dtype=np.float32))
                    
                    if last_critic["hook_critic_mlp2"] is not None:
                        npz_critic_mlp2.append(last_critic["hook_critic_mlp2"].astype(np.float32))
                    else:
                        npz_critic_mlp2.append(np.zeros(64, dtype=np.float32))
                    
                    # Add trajectory tracking data
                    traj_id = chain_id * 10000 + current_link_idx
                    npz_traj_id.append(traj_id)
                    npz_reward.append(rew)
                    npz_done.append(done)
                    npz_link_idx.append(current_link_idx)
                    npz_success_step.append(1.0 if rew > 0 else 0.0)  # success if reward > 0

                # Keep your Parquet row append (unchanged except 'color'/link_idx use locals)
                rows.append({
                    "chain_id": chain_id,
                    "world_seed": world_seed,
                    "step_idx": steps,
                    "link_idx": current_link_idx,
                    "color": current_goal_color,
                    "goal_text": f"FG {current_goal_color}",
                    "switch_flag": 0,
                    "h": None if last_hidden["h"] is None else last_hidden["h"].tolist(),
                    "actor_penult": None if last_actor["penult"] is None else last_actor["penult"].tolist(),
                    "a": np.asarray(act).ravel().tolist(),
                    "obs_features": to_list_or_none(prev_obs_flat),
                    "pos_x": agent_xy_next[0],
                    "pos_y": agent_xy_next[1],
                    "success_step": bool(saw_prop(info_next, current_goal_color)),
                    "propositions": props_str(info_next),
                    "accepting": bool(info_next.get("accepting", False)),
                    "done": bool(done),
                })

                # step loop state
                hit = saw_prop(info_next, current_goal_color)
                obs, info = obs_next, info_next
                if done:
                    break


            # ---------------- LINKS (k = 1..depth) ----------------
            for k in range(1, len(chain)):
                current_link_idx = k
                current_goal_color = chain[k]
                if hasattr(env, "set_goal"):
                    env.set_goal(f"FG {current_goal_color}")
                agent.reset()

                first_step_of_link = True
                steps_on_link = 0
                hit = False
                while steps_on_link < args.per_link_steps and not hit:
                    last_hidden["h"] = None
                    last_actor["penult"] = None
                    last_critic["value"] = None
                    last_critic["hook_critic_mlp0"] = None
                    last_critic["hook_critic_mlp2"] = None
                    env_hooks_cache["hook_env_mlp1"] = None
                    env_hooks_cache["hook_env_mlp3"] = None

                    prev_obs_flat = flatten_features(obs)
                    prev_ap_vec   = ap_vec_from_info(info if isinstance(info, dict) else {}, prop_index)

                    with torch.no_grad():
                        act = agent.get_action(obs, {}, deterministic=args.deterministic)
                    act = coerce_action(act, env.action_space)

                    obs_next, rew, done, info_next = step_unpack(env.step(act))
                    steps += 1
                    steps_on_link += 1

                    h_gru  = None if last_hidden["h"] is None else np.asarray(last_hidden["h"]).ravel()
                    h_act  = None if last_actor["penult"] is None else np.asarray(last_actor["penult"]).ravel()
                    h_env1 = env_hooks_cache["hook_env_mlp1"]
                    h_env3 = env_hooks_cache["hook_env_mlp3"]

                    next_obs_flat = flatten_features(obs_next)
                    next_ap_vec   = ap_vec_from_info(info_next if isinstance(info_next, dict) else {}, prop_index)

                    agent_xy_curr = extract_xy(env=env, obs=obs, info=info)
                    agent_xy_next = extract_xy(env=env, obs=obs_next, info=info_next)
                    vec_curr = vector_to_goal_from_env(current_goal_color, agent_xy_curr, env)
                    vec_next = vector_to_goal_from_env(current_goal_color, agent_xy_next, env)
                    next_pos_oh = color_next_positives(f"FG {current_goal_color}", prop_index)

                    if (prev_obs_flat is not None and next_obs_flat is not None and
                        isinstance(h_gru, np.ndarray) and h_gru.size and
                        isinstance(h_act, np.ndarray) and h_act.size and
                        isinstance(h_env1, np.ndarray) and h_env1.size and
                        isinstance(h_env3, np.ndarray) and h_env3.size):

                        npz_obs.append(prev_obs_flat)
                        npz_next_obs.append(next_obs_flat)
                        npz_action.append(np.asarray(act, dtype=np.float32).ravel())
                        npz_env1.append(h_env1.astype(np.float32))
                        npz_env3.append(h_env3.astype(np.float32))
                        npz_ltl_h.append(h_gru.astype(np.float32))
                        npz_actor_h5.append(h_act.astype(np.float32))
                        npz_next_pos.append(next_pos_oh.astype(np.float32))
                        npz_vec_next.append(vec_next.astype(np.float32))
                        npz_vec_curr.append(vec_curr.astype(np.float32))
                        npz_ap.append(prev_ap_vec.astype(np.float32))
                        npz_next_ap.append(next_ap_vec.astype(np.float32))
                        
                        # Add critic value if available
                        if last_critic["value"] is not None:
                            npz_critic.append(last_critic["value"].astype(np.float32))
                        else:
                            npz_critic.append(np.array([0.0], dtype=np.float32))
                        
                        # Add critic layer activations if available
                        if last_critic["hook_critic_mlp0"] is not None:
                            npz_critic_mlp0.append(last_critic["hook_critic_mlp0"].astype(np.float32))
                        else:
                            npz_critic_mlp0.append(np.zeros(64, dtype=np.float32))
                        
                        if last_critic["hook_critic_mlp2"] is not None:
                            npz_critic_mlp2.append(last_critic["hook_critic_mlp2"].astype(np.float32))
                        else:
                            npz_critic_mlp2.append(np.zeros(64, dtype=np.float32))
                        
                        # Add trajectory tracking data
                        traj_id = chain_id * 10000 + current_link_idx
                        npz_traj_id.append(traj_id)
                        npz_reward.append(rew)
                        npz_done.append(done)
                        npz_link_idx.append(current_link_idx)
                        npz_success_step.append(1.0 if rew > 0 else 0.0)  # success if reward > 0

                    rows.append({
                        "chain_id": chain_id,
                        "world_seed": world_seed,
                        "step_idx": steps,
                        "link_idx": current_link_idx,
                        "color": current_goal_color,
                        "goal_text": f"FG {current_goal_color}",
                        "switch_flag": int(first_step_of_link),
                        "h": None if last_hidden["h"] is None else last_hidden["h"].tolist(),
                        "actor_penult": None if last_actor["penult"] is None else last_actor["penult"].tolist(),
                        "a": np.asarray(act).ravel().tolist(),
                        "obs_features": to_list_or_none(prev_obs_flat),
                        "pos_x": agent_xy_next[0],
                        "pos_y": agent_xy_next[1],
                        "success_step": bool(saw_prop(info_next, current_goal_color)),
                        "propositions": props_str(info_next),
                        "accepting": bool(info_next.get("accepting", False)),
                        "done": bool(done),
                    })
                    first_step_of_link = False

                    hit = saw_prop(info_next, current_goal_color)
                    obs, info = obs_next, info_next
                    if done:
                        break


        finally:
            env.close()

        print(f"  chain {chain_id+1}/{args.num_chains}  sequence={chain}")

    handle.remove()
    actor_handle.remove()
    critic_handle.remove()
    critic_mlp0_handle.remove()
    critic_mlp2_handle.remove()
    for h in env_handles:
        try: h.remove()
        except Exception: pass

    print(f"Hook events captured: GRU={last_hidden['n_events']}  ACTOR={last_actor['n_events']}")

    non_null = sum(1 for _ in filter(lambda r: isinstance(r.get('h'), list) and len(r['h'])>0, rows))
    print(f"Rows with non-empty h: {non_null} / {len(rows)}")

    # Save
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

        # --- NEW: write NPZ for probe_suite_simple.py ---
    npz_ok_len = len(npz_obs)
    if npz_ok_len > 0:
        # Make arrays
        A_obs      = np.stack(npz_obs, axis=0)
        A_next_obs = np.stack(npz_next_obs, axis=0)
        A_action   = np.stack(npz_action, axis=0)
        A_env1     = np.stack(npz_env1, axis=0)
        A_env3     = np.stack(npz_env3, axis=0)
        A_ltl_h    = np.stack(npz_ltl_h, axis=0)
        A_actor_h5 = np.stack(npz_actor_h5, axis=0)
        A_next_pos = np.stack(npz_next_pos, axis=0)
        A_vec_next = np.stack(npz_vec_next, axis=0)
        A_vec_curr = np.stack(npz_vec_curr, axis=0)
        A_ap       = np.stack(npz_ap, axis=0)
        A_next_ap  = np.stack(npz_next_ap, axis=0)
        A_critic   = np.stack(npz_critic, axis=0)
        A_critic_mlp0 = np.stack(npz_critic_mlp0, axis=0)
        A_critic_mlp2 = np.stack(npz_critic_mlp2, axis=0)
        A_traj_id  = np.stack(npz_traj_id, axis=0)
        A_reward   = np.stack(npz_reward, axis=0)
        A_done     = np.stack(npz_done, axis=0)
        A_link_idx = np.stack(npz_link_idx, axis=0)
        A_success_step = np.stack(npz_success_step, axis=0)

        out_npz = Path(args.out_parquet).with_suffix(".npz")
        np.savez_compressed(
            out_npz,
            # Keys expected by probe_suite_simple.py (you can change KEYMAP there if needed)
            obs=A_obs,
            next_obs=A_next_obs,
            action=A_action,
            hook_env_mlp1=A_env1,
            hook_env_mlp3=A_env3,
            hook_ltl_rnn_h=A_ltl_h,
            hook_actor_h5=A_actor_h5,
            next_positives=A_next_pos,
            vec_to_next_pos=A_vec_next,
            vec_to_goal_t=A_vec_curr,
            ap=A_ap,
            next_ap=A_next_ap,
            critic=A_critic,
            hook_critic_mlp0=A_critic_mlp0,
            hook_critic_mlp2=A_critic_mlp2,
            traj_id=A_traj_id,
            reward=A_reward,
            done_flag=A_done,
            link_idx=A_link_idx,
            success_step=A_success_step,
            meta=np.array([{
                "env_id": args.env_id,
                "exp": args.exp,
                "seed": args.seed,
                "props": sorted(list(props)),
            }], dtype=object),
        )
        print(f"Wrote NPZ → {out_npz}  (rows={npz_ok_len})")
    else:
        print("NPZ not written (no complete rows captured for hooks).")


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
