#!/usr/bin/env python3
import os
import re
from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from preprocessing.preprocessing import preprocess_obss
from preprocessing.vocab import VOCAB
from utils.model_store.model_store import ModelStore
from config import model_configs
from ltl.logic.assignment import FrozenAssignment
from sequence.search.exhaustive_search import ExhaustiveSearch


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x

def to_numpy_any(x) -> np.ndarray:
    """Best-effort conversion to a 1D numpy array for printing/stacking.
    Handles torch.Tensor, PackedSequence, tuples/lists.
    """
    try:
        if isinstance(x, PackedSequence):
            x = x.data
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple)):
            # Take first element if it's a 1-length container
            if len(x) == 1:
                return to_numpy_any(x[0])
            x = np.array(x, dtype=object)
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr
    except Exception:
        try:
            arr = np.array(x)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            return arr
        except Exception:
            return np.array([], dtype=float)

def pca_project(X: np.ndarray, n: int = 2, whiten: bool = False) -> np.ndarray:
    Xs = StandardScaler(with_mean=True, with_std=whiten).fit_transform(X)
    pcs = PCA(n_components=n, random_state=0).fit_transform(Xs)
    return pcs


def plot_trajectory(XY: np.ndarray, title: str, out_png: str, annotate: bool = True) -> None:
    plt.figure(figsize=(7, 6), dpi=150)
    xs, ys = XY[:, 0], XY[:, 1]
    plt.plot(xs, ys, '-o', alpha=0.8, markersize=3)
    if annotate:
        for i, (x, y) in enumerate(XY):
            plt.text(x, y, str(i), fontsize=7, ha='left', va='bottom')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def register_hooks(model, capture: dict, log_all_env: bool = False, log_all_ltl: bool = False) -> list:
    hooks = []

    # env_net output (grid representation)
    try:
        if hasattr(model, 'env_net') and model.env_net is not None:
            def env_hook(m, inp, out):
                try:
                    arr = out.detach().cpu().numpy()
                except Exception:
                    arr = np.asarray(out)
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    capture['hook_env_mlp3'] = arr
            hooks.append(model.env_net.register_forward_hook(lambda m, i, o: env_hook(m, i, o)))
            if log_all_env:
                for name, mod in model.env_net.named_modules():
                    if name == '':
                        continue
                    if any(True for _ in mod.children()):
                        continue
                    def make_hook(key):
                        def _hook(m, inp, out):
                            val = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                            try:
                                arr = val.detach().cpu().numpy()
                            except Exception:
                                arr = np.asarray(val)
                            capture[key] = arr
                        return _hook
                    hooks.append(mod.register_forward_hook(make_hook(f"hook_env.{name}")))
    except Exception:
        pass

    # actor penultimate (optional alternative for grid)
    try:
        if hasattr(model, 'actor') and hasattr(model.actor, 'model'):
            seq = model.actor.model
            if len(seq) >= 2:
                penultimate = seq[-2]
                def actor_mid_hook(m, i, o):
                    arr = o.detach().cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    capture['actor_mid'] = arr
                hooks.append(penultimate.register_forward_hook(lambda m, i, o: actor_mid_hook(m, i, o)))
    except Exception:
        pass

    # ltl_net RNN hidden (goal representation)
    try:
        if hasattr(model, 'ltl_net') and model.ltl_net is not None:
            rnn_mod = getattr(model.ltl_net, 'rnn', None)
            if rnn_mod is None:
                rnn_mod = getattr(model.ltl_net, 'gru', None)
            if rnn_mod is not None:
                def ltl_rnn_hook(m, inp, out):
                    try:
                        if isinstance(out, (tuple, list)) and len(out) >= 2:
                            gru_out, h_n = out[0], out[1]
                        else:
                            gru_out, h_n = out, getattr(m, 'last_h', None)
                        # Save sequence output and final hidden
                        # Ensure PackedSequence is handled
                        if isinstance(gru_out, PackedSequence):
                            go = gru_out.data.detach().cpu().numpy()
                        elif isinstance(gru_out, torch.Tensor):
                            go = gru_out.detach().cpu().numpy()
                        else:
                            go = np.asarray(gru_out)
                        capture['hook_ltl_gru_out'] = go
                        if h_n is not None:
                            try:
                                h_np = h_n[-1].detach().cpu().numpy() if hasattr(h_n, 'detach') else np.asarray(h_n)
                            except Exception:
                                h_np = np.asarray(h_n)
                            capture['hook_ltl_gru_h'] = h_np
                            capture['hook_ltl_rnn_h'] = h_np
                    except Exception:
                        pass
                hooks.append(rnn_mod.register_forward_hook(lambda m, i, o: ltl_rnn_hook(m, i, o)))
            if log_all_ltl:
                for name, mod in model.ltl_net.named_modules():
                    if name == '':
                        continue
                    if any(True for _ in mod.children()):
                        continue
                    def make_hook(key):
                        def _hook(m, inp, out):
                            val = out
                            if isinstance(m, nn.GRU):
                                if isinstance(out, (tuple, list)) and len(out) >= 2:
                                    val = out[1][-1]
                            elif isinstance(out, (tuple, list)) and len(out) > 0:
                                val = out[0]
                            try:
                                arr = val.detach().cpu().numpy()
                            except Exception:
                                arr = np.asarray(val)
                            capture[key] = arr
                        return _hook
                    hooks.append(mod.register_forward_hook(make_hook(f"hook_ltl.{name}")))
    except Exception:
        pass
    return hooks


def resolve_letter_token(letter: Optional[str], vocab_keys=None) -> Optional[FrozenAssignment]:
    if letter is None:
        return None
    target = str(letter).strip().lower()
    if vocab_keys is None:
        return None
    for k in vocab_keys:
        if isinstance(k, FrozenAssignment):
            s = str(k).strip()
            if s.lower() == target or s.upper() == target.upper():
                return k
            if s.lower().endswith(target) or s.lower().startswith(target):
                return k
    return None


def parse_formula_letters(formula: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Very small regex-based parser for forms like 'F a', 'F a & F b', 'G ! c'.
    Returns (reach, reach2, avoid) letters (lowercase) or None.
    """
    s = formula
    # reach letters: all occurrences of F <letter>
    reach = re.findall(r'[Ff]\s*([a-zA-Z])', s)
    reach = [r.lower() for r in reach]
    reach1 = reach[0] if len(reach) >= 1 else None
    reach2 = reach[1] if len(reach) >= 2 else None
    # avoid letter: first occurrence of G ! <letter>
    m = re.search(r'[Gg]\s*[!¬]\s*([a-zA-Z])', s)
    avoid = m.group(1).lower() if m else None
    return reach1, reach2, avoid


def build_goal_seq(reach_tok, reach2_tok, avoid_tok, include_reach2: bool) -> tuple:
    seq = []
    if reach_tok is not None:
        av = (tuple([avoid_tok]) if avoid_tok is not None else tuple())
        seq.append((tuple([reach_tok]), av))
    if include_reach2 and reach2_tok is not None:
        av2 = (tuple([avoid_tok]) if avoid_tok is not None else tuple())
        seq.append((tuple([reach2_tok]), av2))
    return tuple(seq)


@dataclass
class Args:
    # Edit these defaults (no CLI needed)
    env: str = 'LetterEnv-v0'
    exp: str = 'test'
    model_seed: int = 1            # model checkpoint (experiments/.../{model_seed})
    rollout_seed: int = 43         # env RNG for this episode
    formula: str = 'F (a & (!d U c))'
    steps: int = 20
    goal_hook: str = 'hook_ltl_gru_h'
    grid_hook: str = 'hook_env_mlp3'
    actor_hook: Optional[str] = None
    dynamic_goal: bool = True
    no_early_stop: bool = False     # stop when success or violation (default behavior)
    whiten: bool = True
    use_exhaustive_agent: bool = True  # use Agent with ExhaustiveSearch for action selection
    agent_verbose: bool = True         # print selected sequences from the agent
    log_all_env: bool = True
    log_all_ltl: bool = True
    print_hooks: bool = False
    out_dir: str = 'interpretability/letter_world/results/single_ep'
    annotate: bool = True
    save_path_png: Optional[str] = 'interpretability/letter_world/results/single_ep/path_overlay.png'


def main():
    args = Args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Env + model
    # If we want to save a path image, ask renderer for an RGB array
    render_mode = 'rgb_array' if args.save_path_png else None
    env = make_env(args.env, FixedSampler.partial(args.formula), render_mode=render_mode)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.model_seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    # Seed rollout RNGs (not model weights)
    random.seed(args.rollout_seed)
    np.random.seed(args.rollout_seed)
    torch.random.manual_seed(args.rollout_seed)

    # Propositions and vocab keys
    try:
        props = set(env.get_propositions())
    except Exception:
        props = set()
    vocab_keys = set(VOCAB.keys())

    # Prepare goal tokens from formula
    reach_ch, reach2_ch, avoid_ch = parse_formula_letters(args.formula)
    reach_tok = resolve_letter_token(reach_ch, vocab_keys)
    reach2_tok = resolve_letter_token(reach2_ch, vocab_keys)
    avoid_tok = resolve_letter_token(avoid_ch, vocab_keys)
    include_reach2 = reach2_tok is not None

    # Hooks
    capture: Dict[str, np.ndarray] = {}
    hooks = register_hooks(model, capture, log_all_env=args.log_all_env, log_all_ltl=args.log_all_ltl)

    # Buffers
    grid_seq = []
    goal_seq = []
    actor_seq = []
    actions = []
    positions = []            # positions before taking the action at time t
    letters = []              # letter under agent before action
    hit_letters = []          # letters observed in info['propositions'] after step (like simulate_letter)
    positions_after = []      # positions after step
    letters_after = []        # letter under agent after step
    # Aggregate all captured env/ltl hooks
    env_all: Dict[str, list] = {}
    ltl_all: Dict[str, list] = {}

    # Rollout
    out = env.reset(seed=args.rollout_seed)
    # Optionally disable early termination on acceptance in the LDBA wrapper
    if args.no_early_stop:
        try:
            w = env
            # unwrap to find LDBAWrapper
            from envs.ldba_wrapper import LDBAWrapper
            while hasattr(w, 'env') and not isinstance(w, LDBAWrapper):
                w = w.env
            if isinstance(w, LDBAWrapper):
                w.terminate_on_acceptance = False
        except Exception:
            pass
    if isinstance(out, (tuple, list)) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    done = False
    t = 0
    # Unwrap to base env for path/position introspection
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    consumed_first = False
    # Optional exhaustive-search agent
    agent = None
    if args.use_exhaustive_agent:
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=args.agent_verbose)
        agent.reset()
        if isinstance(info, dict):
            info.setdefault('ldba_state_changed', True)
    while not done and t < args.steps:
        capture.clear()
        # Choose action either via exhaustive-search agent (preferred) or direct greedy policy
        if agent is not None:
            a = agent.get_action(obs, info, deterministic=True)
            try:
                act = int(np.array(a).flatten()[0])
            except Exception:
                act = int(a)
        else:
            obs_fwd = dict(obs)
            # Optionally update the goal sequence once the first letter is reached
            inc_reach2 = include_reach2
            if args.dynamic_goal and consumed_first:
                inc_reach2 = True  # only the second reach remains
            obs_fwd['goal'] = build_goal_seq(None if (args.dynamic_goal and consumed_first) else reach_tok,
                                             reach2_tok,
                                             avoid_tok,
                                             inc_reach2)
            dl = preprocess_obss([obs_fwd], props)
            with torch.inference_mode():
                dist, _ = model(dl)
            # Greedy action
            try:
                if hasattr(dist, 'mode'):
                    a_t = dist.mode()
                elif hasattr(dist, 'logits'):
                    a_t = torch.argmax(dist.logits, dim=-1)
                else:
                    a_t = dist.sample()
                act = int(a_t.flatten()[0].item())
            except Exception:
                act = int(np.argmax(dist.logits.detach().cpu().numpy()[0])) if hasattr(dist, 'logits') else 0

        # Collect features
        if args.grid_hook in capture:
            arr = capture[args.grid_hook]
            grid_seq.append(to_numpy_any(arr))
        elif args.grid_hook == 'actor_mid' and 'actor_mid' in capture:
            arr = capture['actor_mid']
            grid_seq.append(to_numpy_any(arr))
        else:
            # Defer error until after step if both grid & goal absent
            pass

        if args.goal_hook in capture:
            arr = capture[args.goal_hook]
            goal_seq.append(to_numpy_any(arr))
        elif 'hook_ltl_rnn_h' in capture:
            arr = capture['hook_ltl_rnn_h']
            goal_seq.append(to_numpy_any(arr))

        if args.actor_hook is not None and args.actor_hook in capture:
            arr = capture[args.actor_hook]
            actor_seq.append(to_numpy_any(arr))

        # Record all env/ltl hooks
        for k, v in list(capture.items()):
            arr = to_numpy_any(v)
            if k.startswith('hook_env'):
                env_all.setdefault(k, []).append(arr)
            if k.startswith('hook_ltl'):
                ltl_all.setdefault(k, []).append(arr)

        # Optional prints
        if args.print_hooks:
            ltl_keys = sorted([k for k in capture.keys() if k.startswith('hook_ltl')])
            env_keys = sorted([k for k in capture.keys() if k.startswith('hook_env')])
            if ltl_keys:
                print(f"[t={t}] LTL hooks:")
            for k in ltl_keys:
                a = capture[k]
                vec = to_numpy_any(a)
                shp = np.asarray(to_numpy_any(a)).shape
                try:
                    head = np.round(vec[:6], 4)
                except Exception:
                    head = vec[:6]
                print(f"  - {k}: shape={shp}, head={head}")
            if env_keys:
                print(f"[t={t}] ENV hooks:")
            for k in env_keys:
                a = capture[k]
                vec = to_numpy_any(a)
                shp = np.asarray(to_numpy_any(a)).shape
                try:
                    head = np.round(vec[:6], 4)
                except Exception:
                    head = vec[:6]
                print(f"  - {k}: shape={shp}, head={head}")

        # Track current base env position and letter
        agent_xy = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
        positions.append(agent_xy)
        cur_letter = None
        if hasattr(base_env, 'map') and agent_xy in getattr(base_env, 'map'):
            cur_letter = str(base_env.map[agent_xy]).lower()
        letters.append(cur_letter)
        # Update dynamic-goal consumption when we step onto the first reach letter
        if args.dynamic_goal and (not consumed_first) and reach_ch is not None and cur_letter == reach_ch:
            consumed_first = True

        actions.append(act)
        step_out = env.step(act)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, reward, term, trunc, info = step_out
            done = bool(term or trunc)
        else:
            obs, reward, done, info = step_out

        # Track post-step position and letter, and propositions (letters hit)
        agent_xy_after = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
        positions_after.append(agent_xy_after)
        cur_letter_after = None
        if hasattr(base_env, 'map') and agent_xy_after in getattr(base_env, 'map'):
            cur_letter_after = str(base_env.map[agent_xy_after]).lower()
        letters_after.append(cur_letter_after)
        if isinstance(info, dict) and 'propositions' in info and len(info['propositions']) > 0:
            # LetterWorld has at most one true proposition at a time
            try:
                hit = next(iter(info['propositions']))
                hit_letters.append(str(hit))
            except Exception:
                pass

        # Update dynamic-goal consumption after stepping onto the first reach letter
        if args.dynamic_goal and (not consumed_first) and reach_ch is not None and isinstance(info, dict):
            if reach_ch in info.get('propositions', set()):
                consumed_first = True
        t += 1

    # Validate captures
    if not goal_seq:
        raise RuntimeError(f"No goal representations captured with key '{args.goal_hook}'. Captured: {list(capture.keys())}")
    if not grid_seq:
        raise RuntimeError(f"No grid representations captured with key '{args.grid_hook}'. Captured: {list(capture.keys())}")

    # Prepare matrices
    Ggrid = ensure_2d(np.asarray(grid_seq))
    Ggoal = ensure_2d(np.asarray(goal_seq))
    # PCA projections
    XY_grid = pca_project(Ggrid, n=2, whiten=args.whiten)
    XY_goal = pca_project(Ggoal, n=2, whiten=args.whiten)

    # Save plots
    plot_trajectory(XY_goal, title=f"Goal representation trajectory\n{args.formula}",
                    out_png=os.path.join(args.out_dir, 'goal_ts_pca.png'), annotate=args.annotate)
    plot_trajectory(XY_grid, title=f"Grid representation trajectory\n{args.formula}",
                    out_png=os.path.join(args.out_dir, 'grid_ts_pca.png'), annotate=args.annotate)

    # Optional actor
    if actor_seq:
        Gactor = ensure_2d(np.asarray(actor_seq))
        XY_actor = pca_project(Gactor, n=2, whiten=args.whiten)
        plot_trajectory(XY_actor, title=f"Actor representation trajectory\n{args.formula}",
                        out_png=os.path.join(args.out_dir, 'actor_ts_pca.png'), annotate=args.annotate)

    # Save raw arrays for inspection
    np.savez_compressed(os.path.join(args.out_dir, 'traj_raw.npz'),
                        grid=Ggrid, goal=Ggoal,
                        actor=(np.asarray(actor_seq) if actor_seq else None),
                        actions=np.asarray(actions, dtype=np.int64),
                        positions=np.asarray(positions, dtype=np.int64),
                        letters=np.asarray(letters, dtype=object),
                        positions_after=np.asarray(positions_after, dtype=np.int64),
                        letters_after=np.asarray(letters_after, dtype=object),
                        hit_letters=np.asarray(hit_letters, dtype=object),
                        formula=args.formula)

    # Multi-hook PCA subplots (one subplot per hook), robust to variable dims over time
    def multi_hook_subplots(hooks_dict: Dict[str, list], tag: str, whiten: bool, annotate: bool):
        keys = [k for k in sorted(hooks_dict.keys()) if hooks_dict.get(k)]
        if not keys:
            return
        n = len(keys)
        cols = min(3, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.6), dpi=150)
        axes = np.atleast_1d(axes).ravel()

        for idx, k in enumerate(keys):
            ax = axes[idx]
            seq = hooks_dict[k]
            vecs = [np.asarray(v).reshape(-1) for v in seq]
            if len(vecs) == 0:
                ax.set_title(k); ax.axis('off'); continue
            maxd = max((v.size for v in vecs), default=0)
            if maxd == 0:
                ax.set_title(k); ax.axis('off'); continue
            Xh = np.zeros((len(vecs), maxd), dtype=np.float32)
            for t_i, v in enumerate(vecs):
                d = min(v.size, maxd)
                Xh[t_i, :d] = v[:d]

            try:
                if np.allclose(Xh.var(axis=0).sum(), 0.0) or Xh.shape[0] < 2:
                    XY = np.c_[np.arange(Xh.shape[0]), np.zeros((Xh.shape[0],))]
                else:
                    XY = pca_project(Xh, n=2, whiten=whiten)
            except Exception:
                XY = np.c_[np.arange(Xh.shape[0]), np.zeros((Xh.shape[0],))]

            ax.plot(XY[:, 0], XY[:, 1], '-o', alpha=0.8, markersize=3)
            if annotate:
                for t_i, (x, y) in enumerate(XY):
                    ax.text(x, y, str(t_i), fontsize=7, ha='left', va='bottom')
            ax.set_title(k, fontsize=9)

        # Hide extra axes
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"{tag.upper()} hooks PCA (single episode)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(os.path.join(args.out_dir, f"{tag}_hooks_pca_grid.png"))
        plt.close(fig)

    multi_hook_subplots(ltl_all, tag='ltl', whiten=args.whiten, annotate=args.annotate)
    multi_hook_subplots(env_all, tag='env', whiten=args.whiten, annotate=args.annotate)

    # Optional: save a grid path overlay PNG
    if args.save_path_png:
        try:
            img = base_env.renderer.render_path(base_env, [base_env.actions[a] for a in actions])
            if img is not None:
                plt.imsave(args.save_path_png, img)
        except Exception:
            pass
    print(f"[done] Saved plots + arrays in {args.out_dir}")
    # Print a brief summary similar to simulate_letter
    try:
        print(f"steps_taken: {t}")
        # Aggregate only non-None letters under the agent (pre-step sequence)
        seq_letters = [x for x in letters if x is not None]
        if seq_letters:
            print(f"letters_pre_step: {seq_letters}")
        if hit_letters:
            print(f"letters_hit: {hit_letters}")
        # Final status if available
        if isinstance(info, dict):
            if 'success' in info:
                print('final_status: success')
            elif 'violation' in info:
                print('final_status: violation')
    except Exception:
        pass


if __name__ == '__main__':
    main()
