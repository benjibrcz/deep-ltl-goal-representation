#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import itertools

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from model.model import build_model
from config import model_configs
from preprocessing.preprocessing import preprocess_obss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='LetterEnv-v0')
    ap.add_argument('--exp', type=str, default='test')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--branch_every', type=int, default=10)
    ap.add_argument('--K', type=int, default=2)
    ap.add_argument('--max_sequences', type=int, default=16, help='Max sequences per base (caps 4^K)')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/datasets/branched_kstep.npz')
    args = ap.parse_args()

    sampler = FixedSampler.partial('F a')
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    data = {k: [] for k in ['feature_t', 'a_seq', 'feature_next_K', 'obs_next_raw_K', 'letter_next_id', 'source_id']}

    out = env.reset(seed=args.seed)
    obs = out[0] if isinstance(out, (tuple, list)) else out
    done = False
    t = 0; ep = 0; sid = 0

    # hook capture for actor_mid
    capture = {}
    hooks = []
    try:
        if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
            seq = model.actor.model
            if len(seq) >= 2:
                pen = seq[-2]
                def actor_mid_hook(m, i, o):
                    arr = o.detach().cpu().numpy()
                    if arr.ndim == 4: arr = arr.reshape(arr.shape[0], -1)
                    capture['actor_mid'] = arr
                hooks.append(pen.register_forward_hook(lambda m,i,o: actor_mid_hook(m,i,o)))
    except Exception:
        pass

    def get_base_env(e):
        base = e
        while hasattr(base, 'env'):
            base = base.env
        return base

    while t < args.steps:
        # forward to get feature at t
        capture.clear()
        dl = preprocess_obss([dict(obs, goal=tuple())], set(env.get_propositions()))
        with torch.inference_mode():
            _ = model(dl)
        feat_vec = capture.get('actor_mid', None)
        feat_vec = feat_vec[0] if isinstance(feat_vec, np.ndarray) else None

        # Base world snapshot
        base_env = get_base_env(env)
        agent_xy = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
        map_snapshot = dict(getattr(base_env, 'map', {})) if hasattr(base_env, 'map') else {}

        if t % args.branch_every == 0 and feat_vec is not None:
            current_sid = sid; sid += 1
            # enumerate sequences
            all_seqs = list(itertools.product(range(4), repeat=args.K))
            if len(all_seqs) > args.max_sequences:
                rng = np.random.RandomState(current_sid)
                idx = rng.choice(len(all_seqs), size=args.max_sequences, replace=False)
                seqs = [all_seqs[i] for i in idx]
            else:
                seqs = all_seqs
            for seq in seqs:
                # restore base
                if hasattr(base_env, 'map'): base_env.map = dict(map_snapshot)
                if hasattr(base_env, 'agent'): base_env.agent = tuple(agent_xy)
                # roll sequence
                ok = True
                last_obs = obs
                for a in seq:
                    step_out = env.step(int(a))
                    if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                        last_obs, reward, term, trunc, _ = step_out
                        if bool(term or trunc):
                            ok = False; break
                    else:
                        last_obs, reward, done2, _ = step_out
                # capture next feature and next obs
                capture.clear()
                dl2 = preprocess_obss([dict(last_obs, goal=tuple())], set(env.get_propositions()))
                with torch.inference_mode():
                    _ = model(dl2)
                feat_next = capture.get('actor_mid', None)
                feat_next = feat_next[0] if isinstance(feat_next, np.ndarray) else None
                # next letter id under agent
                letter_id = -1
                if hasattr(base_env, 'map') and hasattr(base_env, 'agent'):
                    pos = tuple(getattr(base_env, 'agent', agent_xy))
                    if pos in base_env.map:
                        letter = base_env.map[pos]
                        try:
                            letter_id = int(getattr(base_env, 'letter_types').index(letter))
                        except Exception:
                            letter_id = -1
                data['feature_t'].append(feat_vec)
                data['a_seq'].append(np.array(seq, dtype=np.int64))
                data['feature_next_K'].append(feat_next)
                data['obs_next_raw_K'].append(np.array(last_obs['features']))
                data['letter_next_id'].append(int(letter_id))
                data['source_id'].append(current_sid)
                # restore base ready for next seq
                if hasattr(base_env, 'map'): base_env.map = dict(map_snapshot)
                if hasattr(base_env, 'agent'): base_env.agent = tuple(agent_xy)

        # progress env with greedy action just to move around
        with torch.inference_mode():
            capture.clear(); _ = model(dl)
        # simple step: move right (0) to diversify; or stay if needed
        step_out = env.step(0)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, reward, term, trunc, _ = step_out
            if bool(term or trunc):
                env.reset(seed=args.seed + ep + 1); ep += 1
        else:
            obs, reward, done, _ = step_out
            if done:
                env.reset(seed=args.seed + ep + 1); ep += 1
        t += 1

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **{k: np.array(v, dtype=object) for k, v in data.items()})
    print('Saved branched K-step dataset to', outp)


if __name__ == '__main__':
    main()


