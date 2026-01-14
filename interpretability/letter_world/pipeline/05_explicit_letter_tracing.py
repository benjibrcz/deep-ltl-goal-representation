#!/usr/bin/env python3
"""
Explicit letter tracing for EnvNet and LTLNet.

Builds causal letter prototypes by toggling only the letter input while holding
all other factors fixed. Produces per-layer PCA subplots and saves prototypes.

Outputs under --out_dir:
- env_letters_pca/<layer>.png (one grid per layer)
- ltl_letters_pca/<layer>.png (one grid per layer)
- env_prototypes.npz (per-letter vectors per env layer)
- ltl_prototypes.npz (per-(role,letter) vectors per LTL layer)
"""

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from preprocessing.vocab import VOCAB
from utils.model_store.model_store import ModelStore
from config import model_configs
from ltl.logic.assignment import Assignment
from preprocessing.preprocessing import preprocess_sequence
from preprocessing.batched_sequences import BatchedReachAvoidSequences


@dataclass
class Args:
    env: str = 'LetterEnv-v0'
    exp: str = 'test'
    seed: int = 1
    letters: str = 'auto12'  # 'a,b,c' or 'a-l' or 'auto12'
    center_only: bool = True  # toggle only the agent tile
    out_dir: str = 'interpretability/letter_world/results/explicit_letter_tracing'
    whiten: bool = True


def letters_from_arg(arg: str) -> List[str]:
    if arg == 'auto12':
        return [chr(ord('a') + i) for i in range(12)]
    if ',' in arg:
        return [s.strip().lower() for s in arg.split(',') if s.strip()]
    if '-' in arg:
        lo, hi = arg.split('-')
        lo = lo.strip().lower(); hi = hi.strip().lower()
        return [chr(c) for c in range(ord(lo), ord(hi) + 1)]
    return [arg.strip().lower()]


def ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def pca_project(X: np.ndarray, n=2, whiten=False) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    if X.shape[0] < 2:
        return np.c_[np.arange(X.shape[0]), np.zeros(X.shape[0])]
    Xs = StandardScaler(with_mean=True, with_std=whiten).fit_transform(X)
    pcs = PCA(n_components=n, random_state=0).fit_transform(Xs)
    return pcs


def register_leaf_hooks(module: nn.Module, capture: dict, ns: str) -> List:
    hooks = []
    for name, mod in module.named_modules():
        if name == '':
            continue
        if any(True for _ in mod.children()):
            continue
        def make_hook(key):
            def _hook(m, inp, out):
                val = out
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    val = out[0]
                try:
                    arr = val.detach().cpu().numpy()
                except Exception:
                    arr = np.asarray(val)
                capture[key] = arr
            return _hook
        hooks.append(mod.register_forward_hook(make_hook(f"{ns}.{name}")))
    return hooks


def grid_letter_prototypes(model, env, letters: List[str], args: Args) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Return per-layer prototypes for each letter under env_net.
    Returns (layer_to_letter_vectors, layer_names)
    """
    # Observation shape and agent/letter mapping
    obs_shape = env.observation_space['features'].shape
    H, W, C = obs_shape
    # base env to get letter types order
    base = env
    while hasattr(base, 'env'):
        base = base.env
    try:
        letter_types = list(getattr(base, 'letter_types'))
    except Exception:
        letter_types = letters
    agent_ch = len(letter_types)  # last channel
    # center
    ci, cj = H // 2, W // 2

    # Build batch [L, H, W, C]
    feats = []
    for ch_letter in letters:
        x = np.zeros((H, W, C), dtype=np.float32)
        # letter plane at center
        if ch_letter in letter_types:
            lid = letter_types.index(ch_letter)
        else:
            # fallback by alpha
            lid = ord(ch_letter) - ord('a')
        x[ci, cj, lid] = 1.0
        x[ci, cj, agent_ch] = 1.0
        feats.append(x)
    X = np.stack(feats, axis=0)  # [L,H,W,C]
    X_t = torch.tensor(X, dtype=torch.float32)

    # Register hooks
    cap = {}
    hooks = []
    if model.env_net is not None:
        # include the root env_net output too
        def root_env_hook(m, inp, out):
            try:
                cap['env_net.out'] = out.detach().cpu().numpy()
            except Exception:
                cap['env_net.out'] = np.asarray(out)
        hooks.append(model.env_net.register_forward_hook(lambda m, i, o: root_env_hook(m, i, o)))
        hooks += register_leaf_hooks(model.env_net, cap, ns='env')

    # Forward only env_net
    if model.env_net is None:
        raise RuntimeError('Model has no env_net to trace.')
    with torch.inference_mode():
        _ = model.env_net(X_t)

    # Collect per-layer vectors: [letters, D]
    layer_to_letter = {}
    for k, arr in cap.items():
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        layer_to_letter[k] = arr  # [L, D]
    # sort layers for plot order
    layer_names = sorted(layer_to_letter.keys())
    return layer_to_letter, layer_names


def save_synthetic_grids(X: np.ndarray, letter_types: List[str], out_dir: str) -> None:
    """Save simple visualizations of synthetic grids used for env tracing.
    Highlights the agent (green) and the active letter at each cell (text),
    using only the provided feature planes.
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, C = X.shape[1:]
    agent_ch = len(letter_types)
    for i, x in enumerate(X):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, W + 1, 1))
        ax.set_yticks(np.arange(0, H + 1, 1))
        ax.grid(True, which='both', color='k', linewidth=0.5, alpha=0.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for r in range(H):
            for c in range(W):
                # detect agent
                has_agent = (x[r, c, agent_ch] > 0.5)
                # detect letter channel set to 1
                lid = None
                for lc in range(len(letter_types)):
                    if x[r, c, lc] > 0.5:
                        lid = lc
                        break
                # draw cell bg
                rect = plt.Rectangle((c, r), 1, 1,
                                     facecolor=(0.88, 0.88, 0.88) if lid is not None else (1.0, 1.0, 1.0),
                                     edgecolor='none')
                ax.add_patch(rect)
                # letter text
                if lid is not None:
                    ax.text(c + 0.5, r + 0.6, letter_types[lid].upper(), ha='center', va='center', fontsize=12, color='k')
                # agent marker (slightly transparent green)
                if has_agent:
                    circ = plt.Circle((c + 0.5, r + 0.5), 0.25, color=(0.0, 0.6, 0.0, 0.6))
                    ax.add_patch(circ)

        ax.set_title('Synthetic grid')
        fig.tight_layout()
        # name by index; caller may rename
        plt.savefig(os.path.join(out_dir, f"grid_{i:02d}.png"))
        plt.close(fig)


def ltl_letter_prototypes(model, env, letters: List[str], args: Args) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Return per-layer prototypes for (role,letter) under ltl_net.
    Returns (layer_to_roleletter_vectors, layer_names)
    """
    # propositions set
    try:
        props = set(env.get_propositions())
    except Exception:
        # use letters list
        props = set(letters)

    # Build FrozenAssignment tokens via Assignment.single_proposition
    ra_seqs = []  # list of (label, seq_list)
    for ch in letters:
        # reach: ( {ch}, ∅ )
        a = Assignment.single_proposition(ch, propositions=props).to_frozen()
        reach_seq = [ (frozenset([a]), frozenset()) ]
        # avoid: ( ∅, {ch} )
        avoid_seq = [ (frozenset(), frozenset([a])) ]
        ra_seqs.append((f'Reach:{ch.upper()}', reach_seq))
        ra_seqs.append((f'Avoid:{ch.upper()}', avoid_seq))

    # Prepare batched sequences using preprocessing
    seq_ints = [ preprocess_sequence(seq) for _, seq in ra_seqs ]
    batched = BatchedReachAvoidSequences(seq_ints, device=None)  # CPU

    # Register hooks across ltl_net
    cap = {}
    hooks = []
    if hasattr(model, 'ltl_net') and model.ltl_net is not None:
        def root_ltl_hook(m, inp, out):
            try:
                cap['ltl_net.out'] = out.detach().cpu().numpy()
            except Exception:
                cap['ltl_net.out'] = np.asarray(out)
        hooks.append(model.ltl_net.register_forward_hook(lambda m, i, o: root_ltl_hook(m, i, o)))
        hooks += register_leaf_hooks(model.ltl_net, cap, ns='ltl')
    else:
        raise RuntimeError('Model has no ltl_net to trace.')

    # Forward ltl_net
    with torch.inference_mode():
        _ = model.ltl_net(batched)

    # Assemble per-layer vectors [roles*letters, D]
    layer_to_roleletter = {}
    labels = [lab for lab, _ in ra_seqs]
    for k, arr in cap.items():
        arr = np.asarray(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        # Ensure length matches labels
        if arr.shape[0] != len(labels):
            # Some layers may broadcast differently; skip
            continue
        layer_to_roleletter[k] = (arr, labels)
    layer_names = sorted(layer_to_roleletter.keys())
    return layer_to_roleletter, layer_names


def plot_per_layer_grid(vectors: Dict[str, np.ndarray], labels: List[str], title_prefix: str, out_dir: str, whiten: bool):
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(vectors.keys())
    for k in keys:
        X = np.asarray(vectors[k])
        X = ensure_2d(X)
        XY = pca_project(X, n=2, whiten=whiten)
        plt.figure(figsize=(6, 5), dpi=150)
        uniq = list(dict.fromkeys(labels))
        lab_arr = np.array(labels, dtype=object)
        for lab in uniq:
            mask = (lab_arr == lab)
            plt.scatter(XY[mask, 0], XY[mask, 1], s=20, alpha=0.7, label=str(lab))
        # centroids
        for lab in uniq:
            mask = (lab_arr == lab)
            cx, cy = XY[mask, :].mean(axis=0)
            plt.scatter([cx], [cy], s=80, marker='X', edgecolors='k')
            plt.text(cx, cy, f" {lab}", fontsize=8)
        plt.title(f"{title_prefix}: {k}")
        plt.legend(fontsize=7, frameon=False, ncol=1)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{k.replace('.', '_')}.png"))
        plt.close()


def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Env+model
    env = make_env(args.env, FixedSampler.partial('F a'), render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    letters = letters_from_arg(args.letters)

    # --- EnvNet tracing ---
    env_vecs, env_layers = grid_letter_prototypes(model, env, letters, args)
    plot_per_layer_grid(env_vecs, [L.upper() for L in letters], 'Env letters', os.path.join(args.out_dir, 'env_letters_pca'), args.whiten)
    # Save prototypes
    np.savez_compressed(os.path.join(args.out_dir, 'env_prototypes.npz'),
                        letters=np.array([L.upper() for L in letters], dtype=object),
                        **{k: v for k, v in env_vecs.items()})
    # Save the synthetic grids we used for env tracing for reference
    # Recreate the batch used in grid_letter_prototypes
    obs_shape = env.observation_space['features'].shape
    H, W, C = obs_shape
    base = env
    while hasattr(base, 'env'):
        base = base.env
    try:
        letter_types = list(getattr(base, 'letter_types'))
    except Exception:
        letter_types = letters
    agent_ch = len(letter_types)
    ci, cj = H // 2, W // 2
    feats = []
    for ch_letter in letters:
        x = np.zeros((H, W, C), dtype=np.float32)
        lid = letter_types.index(ch_letter) if ch_letter in letter_types else (ord(ch_letter) - ord('a'))
        x[ci, cj, lid] = 1.0
        x[ci, cj, agent_ch] = 1.0
        feats.append(x)
    X = np.stack(feats, axis=0)
    save_synthetic_grids(X, letter_types, os.path.join(args.out_dir, 'env_grids'))

    # --- LTLNet tracing ---
    ltl_map, ltl_layers = ltl_letter_prototypes(model, env, letters, args)
    # vectors by layer, labels list shared per layer
    for_layer = {}
    labels = None
    for k in ltl_layers:
        X, labs = ltl_map[k]
        for_layer[k] = X
        labels = labs  # same order for all captured layers
    plot_per_layer_grid(for_layer, labels, 'LTL letters (Reach/Avoid)', os.path.join(args.out_dir, 'ltl_letters_pca'), args.whiten)
    # Save prototypes
    np.savez_compressed(os.path.join(args.out_dir, 'ltl_prototypes.npz'),
                        labels=np.array(labels, dtype=object),
                        **{k: v for k, (v, _) in ltl_map.items()})

    print(f"[done] Tracing written to {args.out_dir}")


if __name__ == '__main__':
    main()
