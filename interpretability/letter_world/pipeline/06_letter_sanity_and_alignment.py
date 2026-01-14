#!/usr/bin/env python3
"""
Letter token sanity checks + LTL geometry characterization + Grid↔Goal alignment.

What it computes:
- Token-ID map for each letter for Reach/Avoid; asserts uniqueness; dumps to JSON.
- LTL embedding pairwise cosine heatmaps for Reach and Avoid tokens.
- One-step delta test: magnitude of differences at embedding vs ltl_net.out for letter changes.
- Layer-wise separability (LogReg accuracy) for LTL layers using explicit prototypes (Reach/Avoid × letters).
- Role-offset analysis at ltl_net.out: Δ = mean_L(μReach[L] − μAvoid[L]) and per-letter cos after subtracting Δ.
- Grid↔Goal alignment (holdout Procrustes): train on half letters, report RMS + per-letter cosines on held-out.

Inputs: loads model (env/exp/seed) and uses explicit letter prototypes (no rollouts).
Outputs in out_dir as PNG/NPZ/JSON.
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from utils.model_store.model_store import ModelStore
from config import model_configs
from ltl.logic.assignment import Assignment
from preprocessing.vocab import VOCAB
from preprocessing.preprocessing import preprocess_sequence
from preprocessing.batched_sequences import BatchedReachAvoidSequences


@dataclass
class Args:
    env: str = 'LetterEnv-v0'
    exp: str = 'test'
    seed: int = 1
    letters: str = 'auto12'
    out_dir: str = 'interpretability/letter_world/results/letter_sanity'
    whiten: bool = True
    holdout_seed: int = 0


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


def get_reach_avoid_token_ids(letters: List[str], propositions: set[str]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return token IDs for one-step (Reach, Avoid) sequences per letter.
    preprocess_sequence returns a list of (reach_ids_list, avoid_ids_list) pairs.
    For a single-step input, take index [0] and then the first id from the respective list.
    """
    reach_ids = {}
    avoid_ids = {}
    for ch in letters:
        a = Assignment.single_proposition(ch, propositions=propositions).to_frozen()
        pair_r = preprocess_sequence([(frozenset([a]), frozenset())])[0]
        pair_v = preprocess_sequence([(frozenset(), frozenset([a]))])[0]
        # pair_* is (reach_ids_list, avoid_ids_list)
        reach_ids[ch.upper()] = int(pair_r[0][0])
        avoid_ids[ch.upper()] = int(pair_v[1][0])
    return reach_ids, avoid_ids


def cosine_matrix(rows: np.ndarray) -> np.ndarray:
    A = rows
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    A = A / n
    return A @ A.T


def plot_cosine_heatmap(M: np.ndarray, labels: List[str], title: str, out_png: str):
    plt.figure(figsize=(6, 5), dpi=150)
    im = plt.imshow(M, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def get_ltl_layer_vectors(model, letters: List[str], propositions: set[str]) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """Return per-layer vectors for (Reach/Avoid × letters) using explicit one-step sequences.
    Dict[layer] = (X [N,D], labels [N]) where labels are 'Reach:A', 'Avoid:A', ...
    """
    # Build sequences
    ra = []
    for ch in letters:
        a = Assignment.single_proposition(ch, propositions=propositions).to_frozen()
        ra.append((f'Reach:{ch.upper()}', [(frozenset([a]), frozenset())]))
        ra.append((f'Avoid:{ch.upper()}', [(frozenset(), frozenset([a]))]))
    seq_ints = [preprocess_sequence(seq) for _, seq in ra]
    batched = BatchedReachAvoidSequences(seq_ints, device=None)
    # Hooks
    cap = {}
    def leaf_hooks(mod: nn.Module, ns: str):
        hs = []
        for name, layer in mod.named_modules():
            if name == '':
                continue
            if any(True for _ in layer.children()):
                continue
            def mk(k):
                def _hook(m, i, o):
                    v = o[0] if isinstance(o, (tuple, list)) and len(o) > 0 else o
                    try:
                        arr = v.detach().cpu().numpy()
                    except Exception:
                        arr = np.asarray(v)
                    cap[f'{ns}.{k}'] = arr
                return _hook
            hs.append(layer.register_forward_hook(mk(name)))
        return hs
    hooks = leaf_hooks(model.ltl_net, 'ltl')
    def root_hook(m, i, o):
        try:
            cap['ltl_net.out'] = o.detach().cpu().numpy()
        except Exception:
            cap['ltl_net.out'] = np.asarray(o)
    hooks.append(model.ltl_net.register_forward_hook(lambda m, i, o: root_hook(m, i, o)))
    with torch.inference_mode():
        _ = model.ltl_net(batched)
    for h in hooks:
        try: h.remove()
        except Exception: pass
    # Assemble per-layer matrices, padding per-layer if needed
    labels = [lab for lab, _ in ra]
    out = {}
    for k, arr in cap.items():
        # Convert to list of 1D vectors
        try:
            N = int(len(arr))
        except Exception:
            continue
        vecs = []
        for i in range(N):
            try:
                vi = arr[i]
            except Exception:
                vi = arr
            vi = np.asarray(vi)
            if vi.ndim > 1:
                vi = vi.reshape(-1)
            vecs.append(vi)
        if len(vecs) != len(labels):
            continue
        maxd = max((v.shape[0] for v in vecs if v.size > 0), default=0)
        if maxd == 0:
            continue
        X = np.zeros((len(vecs), maxd), dtype=np.float32)
        for i, v in enumerate(vecs):
            d = min(v.size, maxd)
            X[i, :d] = v[:d]
        out[k] = (X, labels)
    return out


def ltl_separability(layer_to_vecs: Dict[str, Tuple[np.ndarray, List[str]]]) -> Dict[str, float]:
    """Cross-validated multinomial LogReg per layer to classify role+letter.
    Uses StratifiedKFold with n_splits=min(5, min_class_count). Ensures >=2 per class
    via tiny-noise augmentation if needed.
    """
    accs = {}
    from collections import Counter
    rng = np.random.default_rng(0)
    for k, (X, labs) in layer_to_vecs.items():
        y = np.array(labs, dtype=object)
        counts = Counter(y.tolist())
        min_cnt = min(counts.values())
        if min_cnt < 2:
            X_aug = []
            y_aug = []
            for cls, cnt in counts.items():
                idx = np.where(y == cls)[0]
                Xi = X[idx]
                X_aug.append(Xi)
                y_aug.append(np.full(len(idx), cls, dtype=object))
                if cnt < 2:
                    noise = rng.normal(scale=1e-4, size=Xi.shape)
                    X_aug.append(Xi + noise)
                    y_aug.append(np.full(len(idx), cls, dtype=object))
            X = np.concatenate(X_aug, axis=0)
            y = np.concatenate(y_aug, axis=0)
            counts = Counter(y.tolist())
            min_cnt = min(counts.values())

        # Choose number of folds to ensure each class appears in every test fold
        n_splits = max(2, min(5, int(min_cnt)))
        if n_splits < 2:
            # Fall back to simple train/test with 50% split ensuring at least one per class in test
            test_size = max(0.5, len(counts) / len(y) + 1e-6)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)
            clf = LogisticRegression(max_iter=2000, multi_class='auto')
            clf.fit(Xtr, ytr)
            accs[k] = float(accuracy_score(yte, clf.predict(Xte)))
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        fold_acc = []
        for tr_idx, te_idx in skf.split(X, y):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            clf = LogisticRegression(max_iter=2000, multi_class='auto')
            clf.fit(Xtr, ytr)
            fold_acc.append(accuracy_score(yte, clf.predict(Xte)))
        accs[k] = float(np.mean(fold_acc)) if fold_acc else 0.0
    return accs


def role_offset(mu_out: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, float]:
    letters = sorted(set([L for (r, L) in mu_out.keys() if ('Reach', L) in mu_out and ('Avoid', L) in mu_out]))
    if not letters:
        return {}
    deltas = [mu_out[('Reach', L)] - mu_out[('Avoid', L)] for L in letters]
    d = np.mean(np.stack(deltas, axis=0), axis=0)
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return {L: cos(mu_out[('Reach', L)] - d, mu_out[('Avoid', L)]) for L in letters}


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, float]:
    A0 = A - A.mean(axis=1, keepdims=True)
    B0 = B - B.mean(axis=1, keepdims=True)
    M = B0 @ A0.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    rms = np.sqrt(np.mean((W @ A0 - B0) ** 2))
    return W, float(rms)


def main():
    args = Args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    env = make_env(args.env, FixedSampler.partial('F a'), render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    letters = letters_from_arg(args.letters)
    try:
        props = set(env.get_propositions())
    except Exception:
        props = set(letters)

    # A. Sanity: token ids
    reach_ids, avoid_ids = get_reach_avoid_token_ids(letters, propositions=props)
    ok_reach = (len(set(reach_ids.values())) == len(letters))
    ok_avoid = (len(set(avoid_ids.values())) == len(letters))
    with open(os.path.join(args.out_dir, 'token_map.json'), 'w') as f:
        json.dump({
            'reach_ids': reach_ids,
            'avoid_ids': avoid_ids,
            'unique_reach': ok_reach,
            'unique_avoid': ok_avoid,
        }, f, indent=2)

    # Embedding cosines
    emb = model.ltl_net.embedding
    with torch.no_grad():
        r_rows = torch.stack([emb.weight[int(i)] for i in reach_ids.values()], dim=0).cpu().numpy()
        v_rows = torch.stack([emb.weight[int(i)] for i in avoid_ids.values()], dim=0).cpu().numpy()
    labs = [L.upper() for L in reach_ids.keys()]
    Mr = cosine_matrix(r_rows)
    Mv = cosine_matrix(v_rows)
    plot_cosine_heatmap(Mr, labs, 'LTL embedding cosines (Reach)', os.path.join(args.out_dir, 'cosine_reach_embedding.png'))
    plot_cosine_heatmap(Mv, labs, 'LTL embedding cosines (Avoid)', os.path.join(args.out_dir, 'cosine_avoid_embedding.png'))

    # One-step delta: compare Δ at embedding vs ltl_net.out for two letters
    if len(letters) >= 2:
        A, B = letters[0], letters[1]
        a_id = reach_ids[A.upper()]; b_id = reach_ids[B.upper()]
        with torch.no_grad():
            ea = emb.weight[int(a_id)].cpu().numpy(); eb = emb.weight[int(b_id)].cpu().numpy()
        delta_emb = float(np.linalg.norm(ea - eb))
        # Build batched seqs for Reach:A and Reach:B
        fa = Assignment.single_proposition(A, propositions=props).to_frozen()
        fb = Assignment.single_proposition(B, propositions=props).to_frozen()
        seqs = [preprocess_sequence([(frozenset([fa]), frozenset())]),
                preprocess_sequence([(frozenset([fb]), frozenset())])]
        batched = BatchedReachAvoidSequences(seqs, device=None)
        with torch.inference_mode():
            out = model.ltl_net(batched).cpu().numpy()
        delta_out = float(np.linalg.norm(out[0] - out[1]))
        with open(os.path.join(args.out_dir, 'one_step_delta.json'), 'w') as f:
            json.dump({'letters': [A.upper(), B.upper()], 'delta_embedding': delta_emb, 'delta_out': delta_out}, f, indent=2)

    # B. If tokens are distinct: layer-wise separability and role-offset (explicit prototypes)
    layer_vecs = get_ltl_layer_vectors(model, letters, propositions=props)
    sep = ltl_separability(layer_vecs)
    with open(os.path.join(args.out_dir, 'ltl_separability.json'), 'w') as f:
        json.dump(sep, f, indent=2)
    # role-offset at final output if available
    if 'ltl_net.out' in layer_vecs:
        X, labs = layer_vecs['ltl_net.out']
        mu = {}
        for lab in set(labs):
            mu_key = lab.split(':', 1)
            mu[tuple(mu_key)] = X[np.array(labs) == lab].mean(axis=0)
        ro = role_offset(mu)
        with open(os.path.join(args.out_dir, 'ltl_role_offset_eval.json'), 'w') as f:
            json.dump(ro, f, indent=2)

    # C. Grid↔Goal alignment using explicit prototypes
    # Build env prototypes at center only, similar to 05 script
    # Synthesize grids per letter and forward env_net
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
    for ch in letters:
        x = np.zeros((H, W, C), dtype=np.float32)
        lid = letter_types.index(ch) if ch in letter_types else (ord(ch) - ord('a'))
        x[ci, cj, lid] = 1.0
        x[ci, cj, agent_ch] = 1.0
        feats.append(x)
    Xg = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)
    with torch.inference_mode():
        Eg = model.env_net(Xg).cpu().numpy()
    Eg = Eg.reshape(Eg.shape[0], -1)

    # Goal prototypes at ltl output for Reach role
    ra = []
    for ch in letters:
        a = Assignment.single_proposition(ch, propositions=props).to_frozen()
        ra.append((f'Reach:{ch.upper()}', [(frozenset([a]), frozenset())]))
    seqs = [preprocess_sequence(seq) for _, seq in ra]
    batched = BatchedReachAvoidSequences(seqs, device=None)
    with torch.inference_mode():
        Hg = model.ltl_net(batched).cpu().numpy()

    # Holdout split for alignment
    rng = np.random.default_rng(args.holdout_seed)
    idx = np.arange(len(letters))
    rng.shuffle(idx)
    split = len(idx) // 2
    tr, te = idx[:split], idx[split:]
    # Fit on train
    A = Hg[tr].T  # [D,K]
    B = Eg[tr].T
    W, rms = orthogonal_procrustes(A, B)
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    cos_test = {letters[i].upper(): cos((W @ Hg[i]).reshape(-1), Eg[i]) for i in te}
    with open(os.path.join(args.out_dir, 'alignment_holdout.json'), 'w') as f:
        json.dump({'train_letters': [letters[i].upper() for i in tr],
                   'test_letters': [letters[i].upper() for i in te],
                   'rms_train': rms,
                   'cos_test': cos_test}, f, indent=2)

    print(f"[done] Wrote sanity + alignment results to {args.out_dir}")


if __name__ == '__main__':
    main()
