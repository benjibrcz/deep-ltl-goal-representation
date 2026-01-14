#!/usr/bin/env python3
"""
Probe latent goals via pairwise zone-choice residuals.

Idea:
  1) For steps where ≥2 zones match the current required colour, build pairwise
     choice examples (i vs j): label which zone the agent actually heads toward.
  2) Fit a baseline logistic model using simple hand features (Δdistance, Δbearing).
  3) Add actor penultimate features; if AUC/Acc improves, hidden state encodes
     extra preferences (safety/clearance/smoothness) → evidence for latent goals.

Assumptions:
  - Env exposes zone centres/colours (try env.get_zone_info() or env.world.zones).
  - Info dict has agent_pos; if not, we approximate from obs (left as TODO).
  - Current required colour read from info["current_colour"] if available,
    else parsed from --spec (e.g., "FG blue").
"""

import argparse, sys, re, math, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

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

def get_zone_table(env) -> List[Dict]:
    """Return [{id, colour, center:(x,y)}]. Adapt this to your env if needed."""
    # Try method 1: env.get_zone_info()
    if hasattr(env, "get_zone_info"):
        zs = env.get_zone_info()
        out = []
        for z in zs:
            # Expect z like {'id':..,'colour':..,'center':(..,..)}; fallback keys:
            cid = z.get("id", z.get("zone_id", None))
            col = z.get("colour", z.get("color", None))
            ctr = z.get("center", z.get("centre", z.get("pos", None)))
            if cid is None or col is None or ctr is None: continue
            out.append({"id": cid, "colour": str(col), "center": tuple(ctr)})
        if out: return out

    # Method 2: FlatWorld static definition
    try:
        from envs.flatworld.flatworld import FlatWorld
        out = []
        for idx, c in enumerate(FlatWorld.CIRCLES):
            out.append({"id": idx, "colour": str(getattr(c, 'color', getattr(c, 'colour', ''))),
                        "center": (float(c.center[0]), float(c.center[1]))})
        if out:
            return out
    except Exception:
        pass

    raise RuntimeError("Could not obtain zone info. Please adapt get_zone_table(env).")

def parse_colour_from_spec(spec: str) -> str:
    m = re.search(r"\b(blue|green|yellow|magenta)\b", spec, re.IGNORECASE)
    return m.group(1).lower() if m else "blue"

def bearing(agent_xy: np.ndarray, zone_xy: np.ndarray, agent_heading: float = None) -> float:
    """Return absolute bearing angle (rad) between agent->zone vector and agent heading if provided, else just angle from x-axis."""
    dx, dy = zone_xy[0]-agent_xy[0], zone_xy[1]-agent_xy[1]
    ang_to_zone = math.atan2(dy, dx)
    if agent_heading is None:
        return abs(ang_to_zone)
    # wrap to [-pi,pi]
    d = (ang_to_zone - agent_heading + math.pi) % (2*math.pi) - math.pi
    return abs(d)

def find_chosen_zone_id(traj_xy: List[Tuple[float,float]],
                        candidate_ids: List[int],
                        zone_lookup: Dict[int, Tuple[float,float]],
                        horizon: int = 40,
                        eps: float = 0.5) -> int:
    """
    Choose the candidate whose centre the agent approaches closest within H steps (or enters).
    """
    T = min(horizon, len(traj_xy))
    best = None; best_d = 1e9
    for zid in candidate_ids:
        zx, zy = zone_lookup[zid]
        dmin = min(( (traj_xy[t][0]-zx)**2 + (traj_xy[t][1]-zy)**2 )**0.5 for t in range(T))
        if dmin < best_d:
            best_d = dmin; best = zid
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--num-loops", type=int, default=2)
    ap.add_argument("--spec", type=str, default="FG blue")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--horizon", type=int, default=40, help="steps to look ahead for choice label")
    ap.add_argument("--out", type=str, default="interpretability/zone_env/results/zone_choice_residuals.npz")
    args = ap.parse_args()

    ENV, EXP, SEED = args.env_id, args.exp, args.seed
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    # ── env/model/agent ──
    dummy_env = make_env(ENV, FixedSampler.partial(args.spec), sequence=args.sequence)
    cfg       = model_configs[ENV]
    store     = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()
    torch.set_grad_enabled(False)

    # Hook actor penultimate (best guess: model.actor.enc)
    actor_feats: List[np.ndarray] = []
    def _actor_hook(_, __, out):
        # out is tensor (B, D)
        actor_feats.append(out.detach().cpu().numpy().squeeze())
    hook_handle = None
    if hasattr(model, "actor") and hasattr(model.actor, "enc"):
        hook_handle = model.actor.enc.register_forward_hook(_actor_hook)
    else:
        # Fallback: search for a module named 'actor.enc' or last Linear before heads
        for name, m in model.named_modules():
            if name.endswith("actor.enc"):
                hook_handle = m.register_forward_hook(_actor_hook); break
    if hook_handle is None:
        print("[warn] Could not hook actor penultimate; continuing without hidden-state features.")

    # Containers for dataset
    X_hand, X_actor, y = [], [], []

    # ── rollouts ──
    for ep in range(args.episodes):
        env   = make_env(ENV, FixedSampler.partial(args.spec), sequence=args.sequence)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent   = Agent(model, planner, propositions=props)
        reset_out = env.reset(seed=SEED + 10_000 * ep)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        agent.reset()

        # Cache zone info
        ztab = get_zone_table(env)
        zone_lookup = {z["id"]: z["center"] for z in ztab}
        colour_lookup = {z["id"]: z["colour"].lower() for z in ztab}

        # Rolling buffers to label choice
        traj_xy: List[Tuple[float,float]] = []
        step = 0
        while step < args.max_steps:
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=True)

            a_arr = np.asarray(act).flatten()
            if isinstance(env.action_space, gym.spaces.Discrete):
                step_action = int(a_arr[0])
            else:
                step_action = a_arr

            step_out = env.step(step_action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            # Agent pos (prefer obs features; fallback to env.agent_pos)
            ax, ay = np.nan, np.nan
            try:
                if isinstance(obs, dict) and 'features' in obs:
                    feats = np.asarray(obs['features']).ravel()
                    if feats.size >= 2:
                        ax, ay = float(feats[0]), float(feats[1])
                elif hasattr(env, 'agent_pos'):
                    ap = np.asarray(getattr(env, 'agent_pos'))
                    if ap.size >= 2:
                        ax, ay = float(ap[0]), float(ap[1])
            except Exception:
                pass
            if not np.isnan(ax):
                traj_xy.append((ax, ay))

            # Current required colour — derive from agent sequence if available; else from spec
            cur_col = parse_colour_from_spec(args.spec)
            try:
                if getattr(agent, 'sequence', None):
                    current_goal_assignment_set, _ = agent.sequence[0]
                    if isinstance(current_goal_assignment_set, (set, frozenset)):
                        for frozen_assignment in current_goal_assignment_set:
                            if hasattr(frozen_assignment, 'assignment'):
                                props_true = {p for p, v in frozen_assignment.assignment if v}
                                if len(props_true) == 1:
                                    cur_col = next(iter(props_true)).lower()
                                    break
            except Exception:
                pass

            # Candidate zones of current colour (need ≥2). If not enough, fallback to nearest zones overall.
            cands = [z["id"] for z in ztab if colour_lookup[z["id"]] == cur_col]
            agent_xy = np.array(traj_xy[-1]) if len(traj_xy) else None
            if len(cands) < 2 and agent_xy is not None:
                dists = sorted(((float(np.linalg.norm(agent_xy - np.array(zone_lookup[z["id"]]))), z["id"]) for z in ztab))
                # ensure at least 2 candidates by taking nearest 3 overall
                cands = [zid for _, zid in dists[:3]]
            if len(cands) >= 2 and len(traj_xy) >= 1:
                # Label: which candidate appears most approached within last H observed positions
                chosen = find_chosen_zone_id(traj_xy[-args.horizon:], cands, zone_lookup, horizon=args.horizon, eps=0.5)

                # Build pairwise examples (i vs j)
                agent_xy = np.array(traj_xy[-1])
                # Optional heading if env provides it
                agent_heading = None
                if isinstance(info, dict) and "agent_heading" in info:
                    agent_heading = float(info["agent_heading"])

                # Precompute features per candidate
                feats = {}
                for zid in cands:
                    zx, zy = zone_lookup[zid]
                    dist = float(np.linalg.norm(agent_xy - np.array([zx, zy])))
                    bear = bearing(agent_xy, np.array([zx, zy]), agent_heading)
                    feats[zid] = (dist, bear)

                for i_idx in range(len(cands)):
                    for j_idx in range(i_idx + 1, len(cands)):
                        i, j = cands[i_idx], cands[j_idx]
                        # Δ features: i - j (sign encodes preference direction)
                        d_dist   = feats[i][0] - feats[j][0]
                        d_bear   = feats[i][1] - feats[j][1]
                        # label = 1 if chose i else 0 (if chose j)
                        y_ij = 1 if chosen == i else (0 if chosen == j else None)
                        if y_ij is None:  # if chosen neither (e.g., unclear), skip
                            continue

                        X_hand.append([d_dist, d_bear])

                        if actor_feats:
                            X_actor.append(actor_feats[-1].copy())
                        else:
                            X_actor.append(np.zeros(1, dtype=np.float32))  # placeholder

                        y.append(y_ij)

            step += 1
            if (isinstance(step_out, (tuple, list)) and len(step_out) == 5 and (terminated or truncated)) or \
               (not isinstance(step_out, (tuple, list)) and done):
                break

        env.close()

    if hook_handle is not None:
        hook_handle.remove()

    X_hand = np.asarray(X_hand, dtype=np.float32)
    X_actor = np.asarray(X_actor, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    # Normalize shapes for empty cases
    if X_hand.ndim == 1:
        X_hand = X_hand.reshape(-1, 2) if X_hand.size > 0 else np.zeros((0, 2), dtype=np.float32)
    if X_actor.ndim == 1:
        X_actor = X_actor.reshape(-1, 1) if X_actor.size > 0 else np.zeros((0, 0), dtype=np.float32)
    print(f"[data] pairs: {len(y)}  hand_dim={X_hand.shape[1] if X_hand.ndim==2 else 0}  actor_dim={X_actor.shape[1] if X_actor.ndim==2 else 0}")
    if len(y) == 0:
        print("[warn] No pairwise examples collected. Increase --episodes/--max-steps or adjust --spec.")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X_hand=X_hand, X_actor=X_actor, y=y)
        print(f"[saved] {out.resolve()}")
        return
    # Guard: need both classes for meaningful logistic regression
    uniq = np.unique(y)
    if uniq.size < 2:
        print("[warn] Only one class present in labels; need more diverse choices. Try more episodes/steps or a different spec.")
        # Save minimal artifacts and exit gracefully
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X_hand=X_hand, X_actor=X_actor, y=y)
        print(f"[saved] {out.resolve()}")
        return

    # Baseline: hand features only
    pipe_hand = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))
    ]).fit(X_hand, y)
    p_hand = pipe_hand.predict_proba(X_hand)[:, 1]
    auc_hand = roc_auc_score(y, p_hand); acc_hand = accuracy_score(y, (p_hand >= 0.5).astype(int))
    print(f"[baseline] hand-only  AUC={auc_hand:.3f}  Acc={acc_hand:.3f}")

    # Extended: hand + actor features
    X_ext = np.hstack([X_hand, X_actor]) if X_actor.shape[1] > 0 else X_hand
    pipe_ext = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))
    ]).fit(X_ext, y)
    p_ext = pipe_ext.predict_proba(X_ext)[:, 1]
    auc_ext = roc_auc_score(y, p_ext); acc_ext = accuracy_score(y, (p_ext >= 0.5).astype(int))
    print(f"[extended] hand+actor AUC={auc_ext:.3f}  Acc={acc_ext:.3f}  ΔAUC={auc_ext-auc_hand:.3f}  ΔAcc={acc_ext-acc_hand:.3f}")

    # Actor-only sanity
    pipe_actor = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))
    ]).fit(X_actor, y)
    p_actor = pipe_actor.predict_proba(X_actor)[:, 1]
    auc_actor = roc_auc_score(y, p_actor); acc_actor = accuracy_score(y, (p_actor >= 0.5).astype(int))
    print(f"[actor-only] AUC={auc_actor:.3f}  Acc={acc_actor:.3f}")

    # Save artifacts for later analysis
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        X_hand=X_hand, X_actor=X_actor, y=y,
        p_hand=p_hand, p_ext=p_ext, p_actor=p_actor,
        auc_hand=auc_hand, auc_ext=auc_ext, auc_actor=auc_actor,
        acc_hand=acc_hand, acc_ext=acc_ext, acc_actor=acc_actor
    )
    print(f"[saved] {out.resolve()}")

if __name__ == "__main__":
    main()
