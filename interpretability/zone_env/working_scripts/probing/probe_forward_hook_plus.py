#!/usr/bin/env python3
"""
Forward-look Probe (v2): episode-safe horizons + critic Bellman check

What’s new vs your v1 JSON:
  • Episode-aware indexing: never crosses episode boundaries for k-step targets and Ψ_k.
  • Critic hooks: penultimate vs value head captured separately (automatically resolved).
  • Bellman consistency: V(s) ≈ r + γ V(s'), TD-error correlations, and penult→{V,y} R².

Still includes:
  • Multi-horizon (k in {1,5,20} by default) Δxy and direction prediction
  • Time-shuffle control
  • Successor features Ψ_k over a simple geometry φ(s) (nearest-zone features)

Run:
  python interpretability/working_scripts/probe_forward_look_plus_v2.py --episodes 200 --horizons 1,5,20
"""

import argparse, sys, math, random, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm.auto import trange
import torch

# ───── repo imports ─────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
import gymnasium as gym
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
# ────────────────────────

COLOURS = ["blue","green","yellow","magenta"]
BASE_HOOKS = ["env_net.mlp", "ltl_net.rnn", "actor.enc"]  # critic handled specially (penult + value head)

def nearest_zone_features(agent_xy: np.ndarray, ztab: List[Dict]) -> np.ndarray:
    """φ(s): for each colour, features to NEAREST zone: [cos θ, sin θ, 1/(1+dist)]."""
    feats = []
    for c in COLOURS:
        zs = [z for z in ztab if z["colour"] == c]
        if not zs:
            feats.extend([0.0, 0.0, 0.0]); continue
        best = min(zs, key=lambda z: ((z["center"][0]-agent_xy[0])**2 + (z["center"][1]-agent_xy[1])**2))
        dx, dy = best["center"][0]-agent_xy[0], best["center"][1]-agent_xy[1]
        d = (dx*dx+dy*dy)**0.5; ang = math.atan2(dy, dx)
        feats.extend([math.cos(ang), math.sin(ang), 1.0/(1.0+d)])
    return np.array(feats, dtype=np.float32)

def unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-8
    return v / n


# ---- helpers to identify A+ events (accept) ----
def is_accept_event(info: dict) -> bool:
    if not isinstance(info, dict):
        return False
    for k in ("ltl_accept", "accept", "goal_reached", "spec_satisfied"):
        if k in info and bool(info[k]):
            return True
    return False

# ==== LTL ETA helpers (two-zone ambiguity) ====
def two_zone_event(pt: np.ndarray, zone_xy: Dict[int, np.ndarray], cand_ids: List[int], dist_tol: float) -> bool:
    if len(cand_ids) != 2:
        return False
    p1 = zone_xy.get(cand_ids[0]); p2 = zone_xy.get(cand_ids[1])
    if p1 is None or p2 is None:
        return False
    d1 = float(np.linalg.norm(pt - p1))
    d2 = float(np.linalg.norm(pt - p2))
    return abs(d1 - d2) <= dist_tol

def steps_until_enter(traj_positions: np.ndarray, cand_ids: List[int], zone_xy: Dict[int, np.ndarray], enter_tol: float) -> int | None:
    for i in range(traj_positions.shape[0]):
        p = traj_positions[i]
        ok = False
        for zid in cand_ids:
            z = zone_xy.get(zid)
            if z is None:
                continue
            if float(np.linalg.norm(p - z)) <= enter_tol:
                ok = True
                break
        if ok:
            return i + 1
    return None

def get_zone_table(env) -> List[Dict]:
    # unwrap wrappers
    def unwrap(e, depth=5):
        u = e
        for _ in range(depth):
            if hasattr(u, "unwrapped") and u.unwrapped is not None:
                u = u.unwrapped
            elif hasattr(u, "env") and u.env is not None:
                u = u.env
            else:
                break
        return u

    e = unwrap(env)

    # 1) Direct API
    if hasattr(e, "get_zone_info"):
        try:
            out = []
            for z in e.get_zone_info():
                cid = z.get("id", z.get("zone_id"))
                col = (z.get("colour") or z.get("color") or "").lower()
                ctr = z.get("center") or z.get("centre") or z.get("pos")
                if cid is None or not col or ctr is None:
                    continue
                ctr = np.asarray(ctr, dtype=float).ravel()
                if ctr.size >= 2:
                    out.append({"id": cid, "colour": col, "center": (float(ctr[0]), float(ctr[1]))})
            if out:
                return out
        except Exception:
            pass

    # 2) zone_positions dict
    if hasattr(e, "zone_positions") and isinstance(getattr(e, "zone_positions"), dict):
        out = []
        n = 0
        for key, val in e.zone_positions.items():
            try:
                name = str(key).lower()
                col = name.split("_zone", 1)[0] if "_zone" in name else name.split("_")[0]
                ctr = np.asarray(val, dtype=float).ravel()
                if ctr.size >= 2 and col:
                    out.append({"id": n, "colour": col, "center": (float(ctr[0]), float(ctr[1]))})
                    n += 1
            except Exception:
                continue
        if out:
            return out

    # 3) world.zones list
    if hasattr(e, "world") and hasattr(e.world, "zones"):
        out = []
        for idx, z in enumerate(e.world.zones):
            try:
                col = getattr(z, "colour", getattr(z, "color", "")).lower()
                cen = getattr(z, "center", None)
                if cen is None:
                    cx = getattr(z, "cx", getattr(z, "x", None))
                    cy = getattr(z, "cy", getattr(z, "y", None))
                    if cx is not None and cy is not None:
                        cen = (float(cx), float(cy))
                if not col or cen is None:
                    continue
                cen = np.asarray(cen, dtype=float).ravel()
                if cen.size >= 2:
                    out.append({"id": getattr(z, "id", idx), "colour": col, "center": (float(cen[0]), float(cen[1]))})
            except Exception:
                continue
        if out:
            return out

    # 4) try common containers
    for attr in ("zones", "zone_centers", "objects", "scene", "task", "world"):
        try:
            c = getattr(e, attr)
        except Exception:
            continue
        if isinstance(c, dict):
            out = []
            n = 0
            for k, v in c.items():
                try:
                    col = str(k).lower()
                    if isinstance(v, dict) and "center" in v:
                        cen = np.asarray(v["center"], dtype=float).ravel()
                    else:
                        cen = np.asarray(v, dtype=float).ravel()
                    if cen.size >= 2 and col:
                        out.append({"id": n, "colour": col, "center": (float(cen[0]), float(cen[1]))})
                        n += 1
                except Exception:
                    continue
            if out:
                return out

    raise RuntimeError("Could not obtain zone info")

def resolve_critic_layers(model):
    """
    Try to locate critic penultimate (pre-final) and final value head inside model.critic.
    Returns (penult_module, value_module, penult_name, value_name). Falls back gracefully.
    """
    crit = dict(model.named_modules()).get("critic", None)
    if crit is None:
        return None, None, None, None
    # Collect submodules under "critic" with their qualified names
    subs = [(n, m) for (n, m) in model.named_modules() if n.startswith("critic.") and n != "critic"]
    # Prefer Linear modules for value head
    linear_subs = [(n, m) for (n, m) in subs if isinstance(m, torch.nn.Linear)]
    if linear_subs:
        value_name, value_mod = linear_subs[-1]
        # penult = the previous module that produces input to value_mod
        # If critic is Sequential, grab the layer before the last Linear.
        penult_name, penult_mod = None, None
        parts = value_name.split(".")
        try:
            idx = int(parts[-1])
            parent_name = ".".join(parts[:-1])
            parent = dict(model.named_modules())[parent_name]
            # take previous module in that Sequential as penult if exists
            prev = getattr(parent, str(idx-1), None)
            if prev is not None:
                penult_name = f"{parent_name}.{idx-1}"
                penult_mod = prev
        except Exception:
            pass
        return penult_mod, value_mod, penult_name, value_name
    # Fallback: treat "critic" itself as penult; no explicit value head
    return crit, None, "critic", None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--num-loops", type=int, default=2)
    ap.add_argument("--colours", type=str, default="blue,green,yellow,magenta")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--horizons", type=str, default="1,5,20")
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_json", type=str, default="interpretability/working_scripts/forward_look_v2_results.json")
    ap.add_argument("--save_npz", action="store_true")
    ap.add_argument("--npz_path", type=str, default="interpretability/working_scripts/forward_look_v2_rollouts.npz")
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    ENV, EXP, BASE_SEED = args.env_id, args.exp, args.seed
    COLOUR_LIST = [c.strip().lower() for c in args.colours.split(",") if c.strip()]
    HORIZONS = [int(x) for x in args.horizons.split(",") if x.strip()]
    rng = np.random.default_rng(BASE_SEED)
    torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED); random.seed(BASE_SEED)
    torch.set_grad_enabled(False)

    # model
    dummy_env = make_env(ENV, FixedSampler.partial(f"FG {COLOUR_LIST[0]}"), sequence=args.sequence)
    cfg       = model_configs[ENV]
    store     = ModelStore(ENV, EXP, BASE_SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()

    # hooks: base + critic penult/value
    acts: Dict[str, List[np.ndarray]] = {k: [] for k in BASE_HOOKS}
    handles = []
    def mk_hook(name):
        def _hook(_, __, out):
            if isinstance(out, (tuple, list)):  # GRU: (seq_out, h_n)
                t = out[1][-1] if isinstance(out[1], torch.Tensor) else out[1][0]
            else:
                t = out
            acts[name].append(t.detach().cpu().numpy().squeeze())
        return _hook
    for name, mod in model.named_modules():
        if name in BASE_HOOKS:
            handles.append(mod.register_forward_hook(mk_hook(name)))

    # critic hooks (robust): capture penult as INPUT to final value layer via pre-hook
    critic_value_mod = None
    for name, mod in model.named_modules():
        if name.startswith("critic") and isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == 1:
            critic_value_mod = mod
    if critic_value_mod is None:
        linear_under_critic = [(n, m) for (n, m) in model.named_modules() if n.startswith("critic") and isinstance(m, torch.nn.Linear)]
        critic_value_mod = linear_under_critic[-1][1] if linear_under_critic else None
    crit_pen_buf: List[np.ndarray] = []
    crit_val_buf: List[np.ndarray] = []
    def _prehook_penult(_mod, inputs):
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            crit_pen_buf.append(x.detach().cpu().numpy().squeeze())
        else:
            crit_pen_buf.append(np.array([]))
    def _hook_value(_mod, _inputs, out):
        y = out if isinstance(out, torch.Tensor) else out[0]
        crit_val_buf.append(y.detach().cpu().numpy().squeeze())
    if critic_value_mod is not None:
        handles.append(critic_value_mod.register_forward_pre_hook(_prehook_penult))
        handles.append(critic_value_mod.register_forward_hook(_hook_value))
    else:
        print("[warn] could not locate critic value head; Bellman tests may be uninformative")

    # buffers (episode-aware)
    world_ids: List[int] = []
    colours_arr: List[str] = []
    pos: List[np.ndarray] = []
    hand_geom: List[np.ndarray] = []
    sensors_feat: List[np.ndarray] = []
    actions_feat: List[np.ndarray] = []
    hooks_feat: Dict[str, List[np.ndarray]] = {ln: [] for ln in BASE_HOOKS}
    critic_penult_feat: List[np.ndarray] = []
    critic_value_scalar: List[float] = []
    rewards: List[float] = []
    ep_bounds: List[Tuple[int,int]] = []  # (start_idx, end_idx_exclusive)
    # ETA probe buffers (one sample per pursue start)
    ETA_feats: List[np.ndarray] = []
    ETA_steps: List[int] = []
    ETA_groups: List[int] = []  # world ids for grouping
    # LTL-ETA (two-zone short-horizon) buffers
    LTL_ETA_feats: List[np.ndarray] = []
    LTL_ETA_steps: List[int] = []
    LTL_groups: List[int] = []
    # Choice-axis buffers for actor
    AX_actor: List[np.ndarray] = []
    AY_choice: List[int] = []
    AG_group: List[int] = []

    # rollouts
    idx_counter = 0
    for ep in (trange(args.episodes, desc="episodes", leave=True) if not args.no_progress else range(args.episodes)):
        colour = COLOUR_LIST[ep % len(COLOUR_LIST)]
        spec   = f"FG {colour}"
        world_seed = BASE_SEED + 555 * ep

        env   = make_env(ENV, FixedSampler.partial(spec), sequence=args.sequence)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent   = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=world_seed)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        agent.reset()

        ztab = get_zone_table(env)

        ep_start = idx_counter
        t = 0
        # ETA local state
        last_colour = None
        waiting_for_accept = False
        steps_since_snapshot = 0
        while t < args.max_steps:
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=True)

            a_arr = np.asarray(act).flatten()
            step_action = int(a_arr[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arr.copy()

            step_out = env.step(step_action)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs_next, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, info = step_out

            # position
            ax, ay = np.nan, np.nan
            try:
                if isinstance(obs, dict) and 'features' in obs:
                    feats = np.asarray(obs['features']).ravel()
                    if feats.size >= 2: ax, ay = float(feats[0]), float(feats[1])
                elif hasattr(env, 'agent_pos'):
                    ap = np.asarray(env.agent_pos).ravel()
                    if ap.size >= 2: ax, ay = float(ap[0]), float(ap[1])
            except Exception:
                pass
            if np.isnan(ax):
                obs = obs_next; t += 1
                if done: break
                continue
            p_t = np.array([ax, ay], dtype=np.float32)

            # features
            hand = nearest_zone_features(p_t, ztab)
            sens = np.asarray(obs['features']).ravel().astype(np.float32) if isinstance(obs, dict) and 'features' in obs else np.zeros((0,), dtype=np.float32)

            # stash
            pos.append(p_t)
            hand_geom.append(hand)
            sensors_feat.append(sens)
            actions_feat.append(np.atleast_1d(a_arr).astype(np.float32))
            for ln in BASE_HOOKS:
                if acts[ln]:
                    hooks_feat[ln].append(np.array(acts[ln][-1]).flatten().astype(np.float32))
                else:
                    hooks_feat[ln].append(None)
            if crit_pen_buf:
                critic_penult_feat.append(np.atleast_1d(crit_pen_buf[-1]).astype(np.float32))
            else:
                critic_penult_feat.append(None)
            if crit_val_buf:
                v = crit_val_buf[-1]
                v = float(np.asarray(v).squeeze())
                critic_value_scalar.append(v)
            else:
                critic_value_scalar.append(np.nan)

            rewards.append(float(reward))
            world_ids.append(world_seed)
            colours_arr.append(colour)

            # --- LTL ETA snapshot: when episode colour (goal) changes (once per episode here) ---
            ep_colour = colour
            if last_colour != ep_colour:
                if acts["ltl_net.rnn"]:
                    ETA_feats.append(np.array(acts["ltl_net.rnn"][-1]).flatten())
                    ETA_steps.append(0)
                    ETA_groups.append(world_seed)
                    waiting_for_accept = True
                    steps_since_snapshot = 0
                last_colour = ep_colour

            if waiting_for_accept:
                steps_since_snapshot += 1
                if is_accept_event(info):
                    ETA_steps[-1] = steps_since_snapshot
                    waiting_for_accept = False

            # --- Two-zone ambiguous ETA + choice-axis ---
            try:
                ztab_local = get_zone_table(env)
                zone_xy = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab_local}
                cands = [z["id"] for z in ztab_local if z.get("colour") == ep_colour]
            except Exception:
                ztab_local, zone_xy, cands = [], {}, []

            enter_tol = 1.0
            dist_tol  = 0.8
            if len(cands) == 2 and two_zone_event(p_t, zone_xy, cands, dist_tol):
                ltl_vec = np.array(acts["ltl_net.rnn"][-1]).flatten() if acts["ltl_net.rnn"] else None
                actor_vec_t0 = np.array(acts["actor.enc"][-1]).flatten() if acts["actor.enc"] else None
                # short lookahead on current env (limited steps)
                short_traj = []
                obs_k = obs
                kk = 0
                while kk < 12:
                    with torch.no_grad():
                        act_k = agent.get_action(obs_k, {}, deterministic=True)
                    a_arr_k = np.asarray(act_k).flatten()
                    step_action_k = int(a_arr_k[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arr_k
                    step_out_k = env.step(step_action_k)
                    if isinstance(step_out_k, (tuple, list)) and len(step_out_k) == 5:
                        obs_k, _, term_k, trunc_k, _ = step_out_k
                        done_k = bool(term_k or trunc_k)
                    else:
                        obs_k, _, done_k, _ = step_out_k
                    # pos
                    axk, ayk = np.nan, np.nan
                    if isinstance(obs_k, dict) and 'features' in obs_k:
                        feats_k = np.asarray(obs_k['features']).ravel()
                        if feats_k.size >= 2: axk, ayk = float(feats_k[0]), float(feats_k[1])
                    elif hasattr(env, 'agent_pos'):
                        apk = np.asarray(env.agent_pos).ravel()
                        if apk.size >= 2: axk, ayk = float(apk[0]), float(apk[1])
                    if not np.isnan(axk):
                        short_traj.append(np.array([axk, ayk], dtype=np.float32))
                    kk += 1
                    if done_k: break

                if short_traj:
                    short_arr = np.vstack(short_traj)
                    eta2 = steps_until_enter(short_arr, cands, zone_xy, enter_tol)
                    if eta2 is not None and eta2 > 0 and ltl_vec is not None:
                        LTL_ETA_feats.append(ltl_vec)
                        LTL_ETA_steps.append(int(eta2))
                        LTL_groups.append(world_seed)
                    # label choice
                    def min_future_dist(zid: int) -> float:
                        z = zone_xy.get(zid)
                        if z is None:
                            return 1e9
                        return float(np.min(np.linalg.norm(short_arr - z, axis=1)))
                    z_sorted = sorted(cands)
                    chosen = z_sorted[0] if min_future_dist(z_sorted[0]) < min_future_dist(z_sorted[1]) else z_sorted[1]
                    label = 1 if chosen == z_sorted[0] else 0
                    if actor_vec_t0 is not None:
                        AX_actor.append(actor_vec_t0)
                        AY_choice.append(label)
                        AG_group.append(world_seed)

            obs = obs_next
            idx_counter += 1
            t += 1
            if done: break

        ep_end = idx_counter
        ep_bounds.append((ep_start, ep_end))
        env.close()

    # materialize arrays with padding for missing hooks/sensors
    P = np.vstack(pos)
    H = np.vstack(hand_geom)
    A = np.vstack(actions_feat)
    # sensors pad
    S_dim = max((s.shape[0] for s in sensors_feat), default=0)
    S = np.zeros((len(sensors_feat), S_dim), dtype=np.float32)
    for i, s in enumerate(sensors_feat):
        if s.shape[0] > 0: S[i, :s.shape[0]] = s
    # base hooks
    X_hooks = {}
    for ln in BASE_HOOKS:
        rows = []
        dim = None
        for v in hooks_feat[ln]:
            if v is None:
                if dim is None: dim = 0
                rows.append(np.zeros((dim,), dtype=np.float32))
            else:
                if dim is None: dim = v.shape[0]
                rows.append(v if v.shape[0]==dim else np.pad(v, (0, max(0,dim-v.shape[0]))))
        X_hooks[ln] = np.vstack(rows)
    # critic penult
    rows = []
    dimc = None
    for v in critic_penult_feat:
      if v is None:
          if dimc is None: dimc = 0
          rows.append(np.zeros((dimc,), dtype=np.float32))
      else:
          if dimc is None: dimc = v.shape[0]
          rows.append(v if v.shape[0]==dimc else np.pad(v, (0, max(0,dimc-v.shape[0]))))
    X_hooks["critic.penult"] = np.vstack(rows)
    V = np.array(critic_value_scalar, dtype=np.float32)
    R = np.array(rewards, dtype=np.float32)
    world_ids = np.asarray(world_ids); colours_arr = np.asarray(colours_arr)

    # episode-aware valid (t, t+k) pairs
    def episode_pairs(k: int) -> np.ndarray:
        idxs = []
        for (s, e) in ep_bounds:
            if e - s <= k: continue
            idxs.extend(range(s, e - k))
        return np.array(idxs, dtype=np.int64)

    # sklearn helpers
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, roc_auc_score, log_loss
    from sklearn.compose import TransformedTargetRegressor

    def build_X(key: str, idx: np.ndarray):
        if key == "hand": return H[idx]
        if key == "sensors": return S[idx]
        if key == "hand+sensors": return np.hstack([H[idx], S[idx]])
        if key == "hand+action": return np.hstack([H[idx], A[idx]])
        if key.startswith("hook:"):
            name = key.split(":",1)[1]
            return X_hooks[name][idx]
        if key.startswith("hook+action:"):
            name = key.split(":",1)[1]
            return np.hstack([X_hooks[name][idx], A[idx]])
        raise ValueError(key)

    feature_sets = [
        "hand",
        "sensors",
        "hand+sensors",
        "hand+action",
        "hook:env_net.mlp",
        "hook:ltl_net.rnn",
        "hook:actor.enc",
        "hook:critic.penult",
        "hook+action:env_net.mlp",
        "hook+action:ltl_net.rnn",
        "hook+action:actor.enc",
        "hook+action:critic.penult",
    ]

    results = {
        "meta": {
            "env": ENV, "exp": EXP, "episodes": args.episodes,
            "horizons": HORIZONS, "gamma": args.gamma,
            "features": feature_sets,
            "critic_penult_name": "prehook(input to value)",
        },
        "multi_horizon": {},
        "successor": {},
        "bellman": {}
    }

    # world split
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=BASE_SEED)
    all_idx = np.arange(P.shape[0])
    tr_idx, te_idx = next(gss.split(all_idx, groups=world_ids))

    # ========== A) Multi-horizon (episode-safe) ==========
    rng_local = np.random.default_rng(BASE_SEED+1)
    for k in sorted(set([h for h in HORIZONS if h >= 1])):
        base = episode_pairs(k)
        tr = np.intersect1d(tr_idx, base); te = np.intersect1d(te_idx, base)
        if len(tr) < 200 or len(te) < 100: continue

        dxy_tr = P[tr + k] - P[tr]; dxy_te = P[te + k] - P[te]
        dir_tr = np.vstack([unit_vec(v) for v in dxy_tr]); dir_te = np.vstack([unit_vec(v) for v in dxy_te])
        results["multi_horizon"][k] = {}

        for feat in feature_sets:
            Xtr = build_X(feat, tr); Xte = build_X(feat, te)
            # Δxy_k
            pipe = Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                             ("rg", Ridge(alpha=1e-2, fit_intercept=False))]).fit(Xtr, dxy_tr)
            r2 = r2_score(dxy_te, pipe.predict(Xte))
            r2_ts = r2_score(dxy_te, pipe.predict(Xte[rng_local.permutation(len(Xte))]))
            results["multi_horizon"][k][f"{feat}::delta_xy"] = {"r2": float(r2), "r2_time_shuffle": float(r2_ts)}
            # dir_k
            pipe2 = Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                              ("rg", Ridge(alpha=1e-2, fit_intercept=False))]).fit(Xtr, dir_tr)
            r2d = r2_score(dir_te, pipe2.predict(Xte))
            r2d_ts = r2_score(dir_te, pipe2.predict(Xte[rng_local.permutation(len(Xte))]))
            results["multi_horizon"][k][f"{feat}::direction"] = {"r2": float(r2d), "r2_time_shuffle": float(r2d_ts)}

    # ========== B) Successor features Ψ_k for GOAL colour only (episode-safe, with target scaling + intercept) ==========
    color_to_idx = {c:i for i,c in enumerate(COLOURS)}
    Phi_goal = np.zeros((H.shape[0], 3), dtype=np.float32)
    for i, col in enumerate(colours_arr):
        idxc = color_to_idx[col]
        Phi_goal[i] = H[i, 3*idxc:3*idxc+3]

    for k in sorted(set([h for h in HORIZONS if h >= 1])):
        base = episode_pairs(k)
        tr = np.intersect1d(tr_idx, base); te = np.intersect1d(te_idx, base)
        if len(tr) < 200 or len(te) < 100:
            continue

        Psi = np.zeros_like(Phi_goal)
        for (s, e) in ep_bounds:
            for i in range(0, k+1):
                Psi[s:e-i] += (args.gamma ** i) * Phi_goal[s+i:e]

        results["successor"][f"goal_only_k={k}"] = {}
        for feat in feature_sets:
            Xtr = build_X(feat, tr); Xte = build_X(feat, te)
            base_reg = Pipeline([
                ("sc", StandardScaler(with_mean=True, with_std=True)),
                ("rg", Ridge(alpha=1e-2, fit_intercept=True)),
            ])
            reg = TransformedTargetRegressor(
                regressor=base_reg,
                transformer=StandardScaler(with_mean=True, with_std=True),
            ).fit(Xtr, Psi[tr])
            pred = reg.predict(Xte)
            r2 = r2_score(Psi[te], pred)
            pred_ts = reg.predict(Xte[rng_local.permutation(len(Xte))])
            r2_ts = r2_score(Psi[te], pred_ts)
            results["successor"][f"goal_only_k={k}"][feat] = {"r2": float(r2), "r2_time_shuffle": float(r2_ts)}

    # ========== C) Bellman consistency on critic ==========
    # valid one-step pairs inside episodes
    base1 = episode_pairs(1)
    tr_b = np.intersect1d(tr_idx, base1); te_b = np.intersect1d(te_idx, base1)
    # Targets
    y_tr = R[tr_b] + args.gamma * V[tr_b + 1]
    y_te = R[te_b] + args.gamma * V[te_b + 1]
    v_tr = V[tr_b]; v_te = V[te_b]
    Hc_tr = X_hooks["critic.penult"][tr_b]; Hc_te = X_hooks["critic.penult"][te_b]

    bell = {}
    # penult -> V and penult -> y (with target scaling + intercept)
    base_reg = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("rg", Ridge(alpha=1e-2, fit_intercept=True)),
    ])
    regV = TransformedTargetRegressor(
        regressor=base_reg,
        transformer=StandardScaler(with_mean=True, with_std=True),
    ).fit(Hc_tr, v_tr)
    regY = TransformedTargetRegressor(
        regressor=base_reg,
        transformer=StandardScaler(with_mean=True, with_std=True),
    ).fit(Hc_tr, y_tr)
    bell["penult_to_V_R2"] = float(r2_score(v_te, regV.predict(Hc_te)))
    bell["penult_to_y_R2"] = float(r2_score(y_te, regY.predict(Hc_te)))
    bell["delta_R2_y_minus_V"] = bell["penult_to_y_R2"] - bell["penult_to_V_R2"]

    # TD error correlation with penult changes
    delta_pen = X_hooks["critic.penult"][tr_b + 1] - X_hooks["critic.penult"][tr_b]
    td_err = (y_tr - v_tr)
    # use norms and dot with a whitened axis
    bell["corr_td_err_with_penult_norm"] = float(np.corrcoef(np.linalg.norm(delta_pen, axis=1), td_err)[0,1])
    results["bellman"] = bell

    # ========== Actor-reads-critic test: linear maps to action ==========
    try:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        X_actor_tr = X_hooks["actor.enc"][tr_idx]; X_actor_te = X_hooks["actor.enc"][te_idx]
        X_crit_tr  = X_hooks["critic.penult"][tr_idx]; X_crit_te  = X_hooks["critic.penult"][te_idx]
        Y_act_tr   = A[tr_idx]; Y_act_te   = A[te_idx]

        mk = lambda: Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                               ("rg", Ridge(alpha=1e-2, fit_intercept=True))])
        reg_actor = mk().fit(X_actor_tr, Y_act_tr)
        reg_crit  = mk().fit(X_crit_tr,  Y_act_tr)
        r2_actor  = float(r2_score(Y_act_te, reg_actor.predict(X_actor_te)))
        r2_crit   = float(r2_score(Y_act_te, reg_crit.predict(X_crit_te)))

        X_both_tr = np.hstack([X_crit_tr, X_actor_tr]); X_both_te = np.hstack([X_crit_te, X_actor_te])
        reg_both  = mk().fit(X_both_tr, Y_act_tr)
        r2_both   = float(r2_score(Y_act_te, reg_both.predict(X_both_te)))
        results["actor_reads_critic"] = {
            "r2_action_given_critic": r2_crit,
            "r2_action_given_actor": r2_actor,
            "delta_r2_plus_actor_given_critic": r2_both - r2_crit,
        }
    except Exception as e:
        results["actor_reads_critic"] = {"error": str(e)}

    # ========== LTL subgoal ETA probe ==========
    try:
        ETA_feats_arr = np.asarray(ETA_feats, dtype=np.float32)
        ETA_steps_arr = np.asarray(ETA_steps, dtype=np.float32)
        ETA_groups_arr= np.asarray(ETA_groups)
        mask = (ETA_steps_arr > 0) & np.isfinite(ETA_steps_arr)
        ETA_feats_arr = ETA_feats_arr[mask]
        ETA_steps_arr = ETA_steps_arr[mask]
        ETA_groups_arr= ETA_groups_arr[mask]
        r2_eta = None
        n_samples = int(len(ETA_steps_arr))
        if n_samples >= 50 and ETA_feats_arr.shape[0] == n_samples:
            tr_eta, te_eta = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                                  .split(ETA_feats_arr, groups=ETA_groups_arr))
            reg_eta = Pipeline([("sc", StandardScaler()),
                                ("rg", Ridge(alpha=1e-2, fit_intercept=True))]).fit(ETA_feats_arr[tr_eta], ETA_steps_arr[tr_eta])
            r2_eta = float(r2_score(ETA_steps_arr[te_eta], reg_eta.predict(ETA_feats_arr[te_eta])))
        results["ltl_eta"] = {"r2": r2_eta if r2_eta is not None else None, "n": n_samples}
    except Exception as e:
        results["ltl_eta"] = {"error": str(e)}

    # dump
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {outp.resolve()}")

    if args.save_npz:
        np.savez_compressed(args.npz_path,
            P=P, H=H, A=A, S=S, V=V, R=R,
            world_ids=world_ids, colours=colours_arr, ep_bounds=np.array(ep_bounds, dtype=np.int64),
            **{f"X_{k.replace('.','_')}": X_hooks[k] for k in X_hooks}
        )
        print(f"[saved] {Path(args.npz_path).resolve()}")

    for h in handles: h.remove()

if __name__ == "__main__":
    main()
