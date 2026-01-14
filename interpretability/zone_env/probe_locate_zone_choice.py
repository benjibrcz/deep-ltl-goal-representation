#!/usr/bin/env python3
"""
Early-commit zone-choice probe (one snapshot per pursue event).

Upgrades:
- Multi-colour sampling per episode (FG {blue,green,yellow,magenta} by default)
- Optional leave-one-colour-out (LOCO) evaluation with --holdout-colour
- Test worlds guaranteed different via GroupShuffleSplit(groups=world_seed)
- Actor vs Critic: incremental ΔAUC over hand+env baseline (+actor, +critic, +actor+critic)

Event logic:
- Trigger when EXACTLY two zones of the required colour are present.
- Take ONE snapshot at event start (t=0) when |Δdist| ≤ dist_tol (ambiguous-ish).
- Use a SHORT look-ahead (k steps) ONLY to decide the pursued zone label.
"""

import argparse, sys, math, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
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

HOOK_NAMES = ["env_net.mlp", "ltl_net.rnn", "actor.enc", "critic"]

def parse_colour_from_spec(spec: str) -> str:
    for c in ["blue","green","yellow","magenta"]:
        if c in spec.lower(): return c
    return "blue"

def get_zone_table(env) -> List[Dict]:
    if hasattr(env, "get_zone_info"):
        zs = env.get_zone_info()
        out = []
        for z in zs:
            cid = z.get("id", z.get("zone_id"))
            col = (z.get("colour") or z.get("color") or "").lower()
            ctr = z.get("center") or z.get("centre") or z.get("pos")
            if cid is None or not col or ctr is None: continue
            out.append({"id": cid, "colour": col, "center": tuple(ctr)})
        if out: return out
    if hasattr(env, "world") and hasattr(env.world, "zones"):
        out = []
        for idx, z in enumerate(env.world.zones):
            col = getattr(z, "colour", getattr(z, "color", "")).lower()
            cen = getattr(z, "center", None)
            if cen is None:
                cx = getattr(z, "cx", getattr(z, "x", None))
                cy = getattr(z, "cy", getattr(z, "y", None))
                if cx is not None and cy is not None: cen = (float(cx), float(cy))
            if not col or cen is None: continue
            out.append({"id": getattr(z, "id", idx), "colour": col, "center": tuple(cen)})
        if out: return out
    # FlatWorld fallback
    try:
        from envs.flatworld.flatworld import FlatWorld
        out = []
        for idx, c in enumerate(FlatWorld.CIRCLES):
            out.append({"id": idx, "colour": str(getattr(c, 'color', '')).lower(),
                        "center": (float(c.center[0]), float(c.center[1]))})
        if out: return out
    except Exception:
        pass
    raise RuntimeError("Could not obtain zone info — adapt get_zone_table(env).")

def unit_vec_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.asarray(b) - np.asarray(a)
    n = np.linalg.norm(v) + 1e-8
    return v / n  # [cos, sin]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--num-loops", type=int, default=2)
    ap.add_argument("--colours", type=str, default="blue,green,yellow,magenta",
                    help="comma-separated list of colours to sample per-episode")
    ap.add_argument("--holdout-colour", type=str, default="",
                    help="if set (e.g., 'yellow'), train on other colours and test on this one")
    ap.add_argument("--sequence", action="store_true")
    ap.add_argument("--lookahead", type=int, default=8, help="k steps to decide label (short!)")
    ap.add_argument("--dist_tol", type=float, default=1.0, help="max |d1-d2| at t=0 to accept event")
    ap.add_argument("--test_size", type=float, default=0.25, help="test fraction for world-split")
    ap.add_argument("--out", type=str, default="interpretability/zone_env/results/zone_choice_early_commit.npz")
    args = ap.parse_args()

    ENV, EXP, BASE_SEED = args.env_id, args.exp, args.seed
    COLOURS = [c.strip().lower() for c in args.colours.split(",") if c.strip()]
    HOLDOUT = args.holdout_colour.strip().lower()
    rng = np.random.default_rng(BASE_SEED)
    torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED); random.seed(BASE_SEED)
    torch.set_grad_enabled(False)

    # model
    dummy_env = make_env(ENV, FixedSampler.partial(f"FG {COLOURS[0]}"), sequence=args.sequence)
    cfg       = model_configs[ENV]
    store     = ModelStore(ENV, EXP, BASE_SEED); store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    model     = build_model(dummy_env, status, cfg).eval()

    # hooks
    acts: Dict[str, List[np.ndarray]] = {k: [] for k in HOOK_NAMES}
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
        if name in HOOK_NAMES:
            handles.append(mod.register_forward_hook(mk_hook(name)))

    # Dataset
    X_hand, y_bin, y_vec = [], [], []
    X_by_layer = {ln: [] for ln in HOOK_NAMES}
    world_ids: List[int] = []
    colours_arr: List[str] = []

    # Episodes across colours (cycle deterministically)
    for ep in range(args.episodes):
        colour = COLOURS[ep % len(COLOURS)]
        spec   = f"FG {colour}"
        world_seed = BASE_SEED + 777 * ep

        env   = make_env(ENV, FixedSampler.partial(spec), sequence=args.sequence)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=args.num_loops)
        agent   = Agent(model, planner, propositions=props)

        reset_out = env.reset(seed=world_seed)
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        agent.reset()

        ztab = get_zone_table(env)
        zone_xy = {z["id"]: np.array(z["center"], dtype=np.float32) for z in ztab}

        step = 0
        event_active = False
        buffer_positions: List[np.ndarray] = []

        while step < args.max_steps:
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=True)

            # step env (discrete vs continuous)
            a_arr = np.asarray(act).flatten()
            step_action = int(a_arr[0]) if isinstance(env.action_space, gym.spaces.Discrete) else a_arr
            step_out = env.step(step_action)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out

            # agent position
            ax, ay = np.nan, np.nan
            try:
                if isinstance(obs, dict) and 'features' in obs:
                    feats = np.asarray(obs['features']).ravel()
                    if feats.size >= 2: ax, ay = float(feats[0]), float(feats[1])
                elif hasattr(env, 'agent_pos'):
                    pos = np.asarray(env.agent_pos).ravel()
                    if pos.size >= 2: ax, ay = float(pos[0]), float(pos[1])
            except Exception:
                pass
            if np.isnan(ax):
                step += 1
                if done: break
                continue
            p_t = np.array([ax, ay], dtype=np.float32)

            # two candidate zones of *this episode's* colour
            cands = [z["id"] for z in ztab if z["colour"] == colour]
            if len(cands) == 2:
                d1 = np.linalg.norm(p_t - zone_xy[cands[0]])
                d2 = np.linalg.norm(p_t - zone_xy[cands[1]])
                if not event_active and abs(d1 - d2) <= args.dist_tol:
                    event_active = True
                    buffer_positions = []

                    # hand features at t=0
                    v1 = zone_xy[cands[0]] - p_t
                    v2 = zone_xy[cands[1]] - p_t
                    d_dist = float(np.linalg.norm(v1) - np.linalg.norm(v2))
                    ang1 = math.atan2(v1[1], v1[0]); ang2 = math.atan2(v2[1], v2[0])
                    hand = [d_dist, math.cos(ang1) - math.cos(ang2), math.sin(ang1) - math.sin(ang2)]
                    # stash activations
                    feat_by_layer = {ln: (np.array(acts[ln][-1]).flatten() if acts[ln] else None)
                                     for ln in HOOK_NAMES}
                elif event_active:
                    pass
            else:
                event_active = False

            if event_active:
                buffer_positions.append(p_t.copy())
                if len(buffer_positions) >= args.lookahead:
                    # label from short horizon
                    def min_future_dist(zid):
                        z = zone_xy[zid]
                        return min(np.linalg.norm(pp - z) for pp in buffer_positions)
                    z_sorted = sorted(cands)
                    chosen = z_sorted[0] if min_future_dist(z_sorted[0]) < min_future_dist(z_sorted[1]) else z_sorted[1]

                    # targets at t=0
                    y_bin.append(1 if chosen == z_sorted[0] else 0)
                    y_vec.append(unit_vec_to(p_t, zone_xy[chosen]))

                    # features
                    X_hand.append(hand)
                    for ln in HOOK_NAMES:
                        if feat_by_layer.get(ln) is not None:
                            X_by_layer[ln].append(feat_by_layer[ln])

                    # meta
                    world_ids.append(world_seed)
                    colours_arr.append(colour)

                    event_active = False

            step += 1
            if done: break

        env.close()

    # Arrays
    X_hand = np.asarray(X_hand)
    y_bin = np.asarray(y_bin, dtype=np.int64)
    y_vec = np.asarray(y_vec, dtype=np.float32)
    world_ids = np.asarray(world_ids)
    colours_arr = np.asarray(colours_arr)

    print(f"[data] events={len(y_bin)}  hand_dim={X_hand.shape[1] if len(X_hand) else 0}")
    for ln in HOOK_NAMES:
        n = len(X_by_layer[ln])
        d = (np.vstack(X_by_layer[ln]).shape[1] if n else 0)
        print(f"  {ln:12s}: n={n} dim={d}")

    # ---- helpers for evaluation ----
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

    def fit_eval_auc(X_train, y_train, X_test, y_test, max_iter=800):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=max_iter, class_weight="balanced"))]).fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:,1]
        return roc_auc_score(y_test, p), accuracy_score(y_test, (p>=0.5).astype(int)), log_loss(y_test, p)

    have_env   = len(X_by_layer["env_net.mlp"]) > 0
    have_actor = len(X_by_layer["actor.enc"]) > 0
    have_critic= len(X_by_layer["critic"]) > 0
    X_env   = np.vstack(X_by_layer["env_net.mlp"]) if have_env else None
    X_actor = np.vstack(X_by_layer["actor.enc"])   if have_actor else None
    X_critic= np.vstack(X_by_layer["critic"])      if have_critic else None

    # -------------------------------
    # 1) World-split evaluation
    # -------------------------------
    print("\n=== World-split (different test worlds) ===")
    if len(y_bin) < 50:
        print("[warn] Very few events; consider increasing --episodes.")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=BASE_SEED)
    tr_idx, te_idx = next(gss.split(X_hand, y_bin, groups=world_ids))

    # baselines
    auc_hand, acc_hand, _ = fit_eval_auc(X_hand[tr_idx], y_bin[tr_idx], X_hand[te_idx], y_bin[te_idx])
    print(f"[choice] hand                  AUC={auc_hand:.3f}  Acc={acc_hand:.3f}")

    if have_env:
        X_hand_env = np.hstack([X_hand, X_env])
        auc_env, acc_env, _ = fit_eval_auc(X_hand_env[tr_idx], y_bin[tr_idx], X_hand_env[te_idx], y_bin[te_idx])
        print(f"[choice] hand+env             AUC={auc_env:.3f}  Acc={acc_env:.3f}  Δ(hand→+env)={auc_env-auc_hand:+.3f}")
    else:
        auc_env, acc_env = auc_hand, acc_hand
        X_hand_env = X_hand

    # add actor / critic incrementally over hand+env
    if have_actor:
        X_env_actor = np.hstack([X_hand_env, X_actor])
        auc_env_actor, acc_env_actor, _ = fit_eval_auc(X_env_actor[tr_idx], y_bin[tr_idx], X_env_actor[te_idx], y_bin[te_idx])
        print(f"[choice] hand+env+actor       AUC={auc_env_actor:.3f}  Acc={acc_env_actor:.3f}  Δ(+actor|hand+env)={auc_env_actor-auc_env:+.3f}")
    if have_critic:
        X_env_critic = np.hstack([X_hand_env, X_critic])
        auc_env_critic, acc_env_critic, _ = fit_eval_auc(X_env_critic[tr_idx], y_bin[tr_idx], X_env_critic[te_idx], y_bin[te_idx])
        print(f"[choice] hand+env+critic      AUC={auc_env_critic:.3f}  Acc={acc_env_critic:.3f}  Δ(+critic|hand+env)={auc_env_critic-auc_env:+.3f}")
    if have_actor and have_critic:
        X_env_both = np.hstack([X_hand_env, X_actor, X_critic])
        auc_env_both, acc_env_both, _ = fit_eval_auc(X_env_both[tr_idx], y_bin[tr_idx], X_env_both[te_idx], y_bin[te_idx])
        print(f"[choice] hand+env+actor+critic AUC={auc_env_both:.3f}  Acc={acc_env_both:.3f}  Δ(+both|hand+env)={auc_env_both-auc_env:+.3f}")

    # Ambiguity binning (|Δdist| bins) in world-split
    abs_dd = np.abs(X_hand[:, 0])
    bins = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    print("\n[binning by |Δdist| at t=0] (world-split)")
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        mask = (abs_dd >= lo) & (abs_dd < hi)
        tr = np.intersect1d(tr_idx, np.where(mask)[0])
        te = np.intersect1d(te_idx, np.where(mask)[0])
        if len(tr) < 40 or len(te) < 20:
            print(f"  bin {labels[i]:>10s}: n_tr={len(tr):<3d} n_te={len(te):<3d} (skip)")
            continue
        a_hand, _, _ = fit_eval_auc(X_hand[tr], y_bin[tr], X_hand[te], y_bin[te])
        if have_env:
            a_env, _, _ = fit_eval_auc(X_hand_env[tr], y_bin[tr], X_hand_env[te], y_bin[te])
        else:
            a_env = a_hand
        line = f"  bin {labels[i]:>10s}: AUC_hand={a_hand:.3f}  AUC_hand+env={a_env:.3f}"
        if have_actor:
            a_env_actor, _, _ = fit_eval_auc(np.hstack([X_hand_env[tr], X_actor[tr]]), y_bin[tr],
                                             np.hstack([X_hand_env[te], X_actor[te]]), y_bin[te])
            line += f"  AUC+actor={a_env_actor:.3f}  Δ(actor|env)={a_env_actor-a_env:+.3f}"
        if have_critic:
            a_env_critic, _, _ = fit_eval_auc(np.hstack([X_hand_env[tr], X_critic[tr]]), y_bin[tr],
                                              np.hstack([X_hand_env[te], X_critic[te]]), y_bin[te])
            line += f"  AUC+critic={a_env_critic:.3f}  Δ(critic|env)={a_env_critic-a_env:+.3f}"
        print(line)

    # -------------------------------
    # 2) Leave-one-colour-out (optional)
    # -------------------------------
    if HOLDOUT and HOLDOUT in COLOURS:
        print(f"\n=== LOCO: train on {set(COLOURS)-{HOLDOUT}}  test on {HOLDOUT} ===")
        tr_mask = colours_arr != HOLDOUT
        te_mask = colours_arr == HOLDOUT

        # ensure different worlds too (we already used disjoint seeds per episode)
        Xh_tr, Xh_te = X_hand[tr_mask], X_hand[te_mask]
        y_tr,  y_te  = y_bin[tr_mask],  y_bin[te_mask]

        auc_hand_loco, acc_hand_loco, _ = fit_eval_auc(Xh_tr, y_tr, Xh_te, y_te)
        print(f"[LOCO] hand                  AUC={auc_hand_loco:.3f}  Acc={acc_hand_loco:.3f}")

        if have_env:
            Xe_tr = np.hstack([Xh_tr, X_env[tr_mask]])
            Xe_te = np.hstack([Xh_te, X_env[te_mask]])
            auc_env_loco, acc_env_loco, _ = fit_eval_auc(Xe_tr, y_tr, Xe_te, y_te)
            print(f"[LOCO] hand+env             AUC={auc_env_loco:.3f}  Acc={acc_env_loco:.3f}  Δ(hand→+env)={auc_env_loco-auc_hand_loco:+.3f}")
        else:
            Xe_tr, Xe_te = Xh_tr, Xh_te
            auc_env_loco = auc_hand_loco

        if have_actor:
            Xea_tr = np.hstack([Xe_tr, X_actor[tr_mask]])
            Xea_te = np.hstack([Xe_te, X_actor[te_mask]])
            auc_ea_loco, acc_ea_loco, _ = fit_eval_auc(Xea_tr, y_tr, Xea_te, y_te)
            print(f"[LOCO] hand+env+actor       AUC={auc_ea_loco:.3f}  Acc={acc_ea_loco:.3f}  Δ(+actor|hand+env)={auc_ea_loco-auc_env_loco:+.3f}")

        if have_critic:
            Xec_tr = np.hstack([Xe_tr, X_critic[tr_mask]])
            Xec_te = np.hstack([Xe_te, X_critic[te_mask]])
            auc_ec_loco, acc_ec_loco, _ = fit_eval_auc(Xec_tr, y_tr, Xec_te, y_te)
            print(f"[LOCO] hand+env+critic      AUC={auc_ec_loco:.3f}  Acc={acc_ec_loco:.3f}  Δ(+critic|hand+env)={auc_ec_loco-auc_env_loco:+.3f}")

        if have_actor and have_critic:
            Xeb_tr = np.hstack([Xe_tr, X_actor[tr_mask], X_critic[tr_mask]])
            Xeb_te = np.hstack([Xe_te, X_actor[te_mask], X_critic[te_mask]])
            auc_eb_loco, acc_eb_loco, _ = fit_eval_auc(Xeb_tr, y_tr, Xeb_te, y_te)
            print(f"[LOCO] hand+env+actor+critic AUC={auc_eb_loco:.3f}  Acc={acc_eb_loco:.3f}  Δ(+both|hand+env)={auc_eb_loco-auc_env_loco:+.3f}")

    # -------------------------------
    # (Optional) Direction regression at t=0 (angular MAE)
    # -------------------------------
    def angular_mae(pred_vec, true_vec):
        pv = pred_vec / (np.linalg.norm(pred_vec, axis=1, keepdims=True)+1e-8)
        tv = true_vec / (np.linalg.norm(true_vec, axis=1, keepdims=True)+1e-8)
        dots = np.clip(np.sum(pv*tv, axis=1), -1.0, 1.0)
        return float(np.mean(np.degrees(np.arccos(dots))))

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if len(y_vec) > 20:
        print("\n[dir] angular MAE at t=0 (world-split)")
        # world split again
        tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=BASE_SEED)
                              .split(X_hand, y_bin, groups=world_ids))
        for ln in HOOK_NAMES:
            if len(X_by_layer[ln]) == 0: 
                print(f"  {ln:12s}: (no data)")
                continue
            X_layer = np.vstack(X_by_layer[ln])
            reg = Pipeline([("sc", StandardScaler(with_mean=True, with_std=True)),
                            ("rg", Ridge(alpha=1e-2, fit_intercept=False))]).fit(X_layer[tr_idx], y_vec[tr_idx])
            mae = angular_mae(reg.predict(X_layer[te_idx]), y_vec[te_idx])
            print(f"  {ln:12s}: MAE={mae:.1f}°")

    # Save for later
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        X_hand=X_hand, y_bin=y_bin, y_vec=y_vec,
        world_ids=world_ids, colours=colours_arr,
        **{f"X_{ln.replace('.','_')}": (np.vstack(X_by_layer[ln]) if X_by_layer[ln] else np.zeros((0,))) for ln in HOOK_NAMES}
    )
    print(f"\n[saved] {out.resolve()}")

if __name__ == "__main__":
    main()
