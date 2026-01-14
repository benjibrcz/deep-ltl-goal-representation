#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import random
import torch

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from preprocessing.preprocessing import preprocess_obss
from model.model import build_model
from config import model_configs


def resolve_layer_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    # Support common actor stacks: .model (Sequential) or .mlp
    if not hasattr(model, 'actor') or model.actor is None:
        raise ValueError('Model has no actor module')
    seq = None
    if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
        seq = model.actor.model
    elif hasattr(model.actor, 'mlp') and hasattr(model.actor.mlp, '__getitem__'):
        seq = model.actor.mlp
    if seq is None or len(seq) < 2:
        raise ValueError('Unsupported actor architecture (no indexable sequence with >=2 layers)')
    # Penultimate module is prelogits output
    if layer_name in ('actor_prelogits', 'actor_mid'):
        return seq[-2]
    raise ValueError(f"Unsupported layer: {layer_name}")


def pick_source_vector(source_npz: Optional[np.lib.npyio.NpzFile], feature: str,
                       match: str, pos: Optional[np.ndarray], action: Optional[int]) -> Optional[np.ndarray]:
    if source_npz is None:
        return None
    if feature not in source_npz.files:
        return None
    Z = source_npz[feature]
    mask = np.array([isinstance(z, np.ndarray) for z in Z])
    idx = np.where(mask)[0]
    if match == 'pos' and pos is not None and 'pos' in source_npz.files:
        pos_all = source_npz['pos']
        ok = []
        for i in idx:
            pi = pos_all[i]
            if isinstance(pi, np.ndarray) and pi.shape == pos.shape and (pi == pos).all():
                ok.append(i)
        if len(ok) > 0:
            return Z[np.random.choice(ok)]
    if match == 'action' and action is not None and 'action' in source_npz.files:
        a_all = source_npz['action']
        ok = [i for i in idx if isinstance(a_all[i], (int, np.integer)) and int(a_all[i]) == int(action)]
        if len(ok) > 0:
            return Z[np.random.choice(ok)]
    # Fallback: random valid vector
    return Z[np.random.choice(idx)] if len(idx) > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LetterEnv-v0')
    p.add_argument('--exp', type=str, default='test')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--layer', type=str, default='actor_prelogits')
    p.add_argument('--mode', type=str, default='steer', choices=['swap', 'steer', 'subspace', 'ablate'])
    p.add_argument('--alpha', type=str, default='0.0,0.25,0.5,1.0')
    p.add_argument('--match', type=str, default='none', choices=['none', 'pos', 'action'])
    p.add_argument('--target_action', type=int, default=None, help='If set, compute targeted flip rate to this action')
    p.add_argument('--base_action', type=int, default=None, help='If set, restrict targeted evaluation to A->B flips')
    p.add_argument('--source', type=str, default=None, help='Path to rollouts npz for z_src')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--formula', type=str, default='F a')
    p.add_argument('--out', type=str, default='interpretability/letter_world/results/patch_live.jsonl')
    args = p.parse_args()

    device = torch.device(args.device)
    # Prepare env + model via ModelStore (align with logging script API)
    sampler = FixedSampler.partial(args.formula)
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location=device)
    model = build_model(env, status, cfg).to(device).eval()
    # Freeze RNGs for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # Propositions for preprocessing
    props = set(env.get_propositions())

    layer = resolve_layer_by_name(model, args.layer)
    # Support comma or space separated alphas
    sep = ',' if ',' in args.alpha else ' '
    alphas = [float(x) for x in args.alpha.split(sep) if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = np.load(args.source, allow_pickle=True) if args.source else None

    # Choose direction: targeted vector for actor_prelogits if provided, else PC1 fallback
    d_vec_np = None
    if args.mode == 'steer' and args.layer == 'actor_prelogits' and args.target_action is not None and args.base_action is not None:
        # explicit retarget vector in logit space: e_target - e_base
        # infer action dimension from any sample in source or default to 4
        action_dim = None
        if src is not None and args.layer in src.files:
            sample = next((z for z in src[args.layer] if isinstance(z, np.ndarray)), None)
            if isinstance(sample, np.ndarray):
                action_dim = sample.shape[-1]
        if action_dim is None:
            action_dim = 4
        v = np.zeros(action_dim, dtype=float)
        v[int(args.target_action)] += 1.0
        v[int(args.base_action)] -= 1.0
        d_vec_np = v
    elif args.mode == 'steer' and src is not None and args.layer in src.files:
        # PC1 fallback from dataset
        Z = src[args.layer]
        Zs = np.stack([z for z in Z if isinstance(z, np.ndarray)], axis=0)
        mu = Zs.mean(0, keepdims=True)
        Zc = Zs - mu
        v = np.random.randn(Zc.shape[1])
        v /= (np.linalg.norm(v) + 1e-8)
        for _ in range(10):
            v = (Zc.T @ (Zc @ v))
            v /= (np.linalg.norm(v) + 1e-8)
        d_vec_np = v / (np.linalg.norm(v) + 1e-8)

    rng = np.random.RandomState(args.seed)
    records = []

    def run_step(obs) -> Tuple[int, np.ndarray]:
        # Ensure baseline forward has no patch applied
        state['alpha'] = 0.0
        obs_forward = dict(obs)
        # Use empty tuple as goal to avoid sampler object issues; consistent with logging
        obs_forward['goal'] = tuple()
        dl = preprocess_obss([obs_forward], props, device=device)
        with torch.inference_mode():
            dist, _ = model(dl)
            if hasattr(dist, 'logits'):
                logits = dist.logits.detach().cpu().numpy()[0]
            elif hasattr(dist, 'probs'):
                logits = np.log(dist.probs.detach().cpu().numpy()[0] + 1e-12)
            else:
                logits = None
            a = int(torch.argmax(dist.logits, dim=-1).flatten()[0].item()) if hasattr(dist, 'logits') else int(np.argmax(logits))
        return a, logits

    # Hook that will steer/swap the selected layer output
    state = {'alpha': 0.0, 'z_src': None, 'd': None}
    metrics_state = {'z_l2': None}

    def hook_fn(_m, _i, out):
        z = out
        alpha = state['alpha']
        if alpha == 0.0:
            return out
        if args.mode == 'swap' and state['z_src'] is not None:
            z_new = (1.0 - alpha) * z + alpha * state['z_src'].to(z)
            try:
                metrics_state['z_l2'] = float(torch.norm(z_new - z).item())
            except Exception:
                pass
            return z_new
        if args.mode == 'steer' and state['d'] is not None:
            z_new = z + alpha * state['d'].to(z)
            try:
                metrics_state['z_l2'] = float(torch.norm(z_new - z).item())
            except Exception:
                pass
            return z_new
        if args.mode == 'subspace' and state['z_src'] is not None:
            # simple subspace: top-1 dir if available
            if state['d'] is not None:
                u = state['d'] / (state['d'].norm() + 1e-8)
                proj = (z - state['z_src'].to(z)) @ u
                z_new = z + alpha * (proj @ u.T)
                try:
                    metrics_state['z_l2'] = float(torch.norm(z_new - z).item())
                except Exception:
                    pass
                return z_new
        if args.mode == 'ablate' and state['d'] is not None:
            u = state['d'] / (state['d'].norm() + 1e-8)
            proj = (z @ u) @ u.T
            z_new = z - proj
            try:
                metrics_state['z_l2'] = float(torch.norm(z_new - z).item())
            except Exception:
                pass
            return z_new
        return out

    h = layer.register_forward_hook(hook_fn)

    # Per-alpha live counters for targeted and collateral
    from collections import defaultdict
    stats = defaultdict(lambda: {'n': 0, 'flip': 0, 'base': 0, 'targ_on_base': 0, 'nonbase': 0, 'to_target_nonbase': 0})

    # rollout without patch to gather context, but we measure immediate flip vs patched forward at same obs
    out_f = out_path.open('w')
    try:
        out = env.reset(seed=args.seed)
        obs = out[0] if isinstance(out, (tuple, list)) else out
        done = False
        t = 0
        while not done and t < args.steps:
            pos = obs['pos'] if 'pos' in obs else None
            base_action, base_logits = run_step(obs)

            # pick source and direction per step
            z_src_np = pick_source_vector(src, args.layer, args.match, pos, base_action)
            d_vec = torch.from_numpy(d_vec_np) if d_vec_np is not None else None
            z_src_t = torch.from_numpy(z_src_np) if isinstance(z_src_np, np.ndarray) else None

            # per-alpha measurements (no env stepping with patch; we step with base action to keep dynamics stable)
            for alpha in alphas:
                state['alpha'] = float(alpha)
                state['z_src'] = z_src_t
                state['d'] = d_vec
                # Patched forward on same obs
                obs_forward = dict(obs)
                obs_forward['goal'] = tuple()
                dl = preprocess_obss([obs_forward], props, device=device)
                with torch.inference_mode():
                    dist_p, _ = model(dl)
                    if hasattr(dist_p, 'logits'):
                        logits_p = dist_p.logits.detach().cpu().numpy()[0]
                        a_p = int(np.argmax(logits_p))
                    elif hasattr(dist_p, 'probs'):
                        logits_p = np.log(dist_p.probs.detach().cpu().numpy()[0] + 1e-12)
                        a_p = int(np.argmax(logits_p))
                    else:
                        logits_p = None
                        a_p = base_action
                # logit margin delta
                margin_delta = None
                if base_logits is not None and logits_p is not None:
                    # margin = top - second best
                    def margin(x):
                        idx = np.argsort(x)[-2:]
                        return float(x[idx[-1]] - x[idx[-2]])
                    margin_delta = margin(logits_p) - margin(base_logits)
                rec = dict(t=int(t), alpha=float(alpha), base_action=int(base_action), patched_action=int(a_p))
                # Targeted flip flags
                if args.target_action is not None:
                    rec['targeted'] = int(a_p == int(args.target_action) and (args.base_action is None or base_action == int(args.base_action)))
                if base_logits is not None and logits_p is not None:
                    # KL(base||patched)
                    pb = torch.softmax(torch.from_numpy(base_logits), dim=-1).numpy()
                    pp = torch.softmax(torch.from_numpy(logits_p), dim=-1).numpy()
                    kl = float((pb * (np.log(pb + 1e-12) - np.log(pp + 1e-12))).sum())
                    rec.update(kl=kl)
                if metrics_state.get('z_l2') is not None:
                    rec.update(z_l2=float(metrics_state['z_l2']))
                if margin_delta is not None:
                    rec.update(margin_delta=float(margin_delta))
                # Update per-alpha counters
                st = stats[float(alpha)]
                st['n'] += 1
                st['flip'] += int(base_action != a_p)
                if args.target_action is not None and args.base_action is not None:
                    if base_action == int(args.base_action):
                        st['base'] += 1
                        st['targ_on_base'] += int(a_p == int(args.target_action))
                    else:
                        st['nonbase'] += 1
                        st['to_target_nonbase'] += int(a_p == int(args.target_action))
                out_f.write(json.dumps(rec) + "\n")

            # Reset patch to no-op before stepping env
            state['alpha'] = 0.0
            # step env with base action (unpatched)
            step_out = env.step(base_action)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, reward, term, trunc, _info = step_out
                done = bool(term or trunc)
            else:
                obs, reward, done, _info = step_out
            t += 1
    finally:
        out_f.close()
        h.remove()
        env.close()

    # Write per-alpha summary CSV next to out JSONL
    try:
        summ_path = out_path.with_name(out_path.stem + '_summary.csv')
        with summ_path.open('w') as sf:
            sf.write('alpha,n,flip,flip_rate,base,targ_on_base,targeted_rate,nonbase,to_target_nonbase,collateral_rate\n')
            for a in sorted(stats.keys()):
                st = stats[a]
                flip_rate = st['flip'] / max(st['n'], 1)
                targ_rate = st['targ_on_base'] / max(st['base'], 1)
                coll_rate = st['to_target_nonbase'] / max(st['nonbase'], 1)
                sf.write(f"{a},{st['n']},{st['flip']},{flip_rate},{st['base']},{st['targ_on_base']},{targ_rate},{st['nonbase']},{st['to_target_nonbase']},{coll_rate}\n")
        print(f"Saved live patch logs to {out_path} and summary to {summ_path}")
    except Exception:
        print(f"Saved live patch logs to {out_path}")


if __name__ == '__main__':
    main()


