import argparse
import os
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[2]
import sys
sys.path.append(str(SRC / 'src'))

from envs.env_utils import make_env, get_env_attr
from ltl.samplers.fixed_sampler import FixedSampler
from model.model import build_model
from model.agent import Agent
from config import model_configs
from sequence.search.exhaustive_search import ExhaustiveSearch
from utils.model_store.model_store import ModelStore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LetterEnv-v0')
    p.add_argument('--exp', type=str, default='test')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--formula', type=str, default='F a')
    p.add_argument('--formulas', type=str, default=None, help='Comma-separated list of formulas, e.g. "F a,F b,F c"')
    p.add_argument('--out', type=str, default=str(SRC / 'interpretability' / 'letter_world' / 'datasets' / 'letterworld_rollouts.npz'))
    p.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def register_hooks(model, capture: Dict[str, Any]):
    hooks = []

    # actor input (embedding passed to actor)
    if hasattr(model, 'actor') and model.actor is not None:
        def actor_pre_hook(m, inp):
            capture['actor_in'] = inp[0].detach().cpu().numpy()
        hooks.append(model.actor.register_forward_pre_hook(lambda m, inp: actor_pre_hook(m, inp)))

        # actor penultimate layer (pre-logits) if available
        try:
            if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
                seq = model.actor.model
                if len(seq) >= 2:
                    penultimate = seq[-2]
                    def actor_mid_hook(m, inp, out):
                        capture['actor_mid'] = out.detach().cpu().numpy()
                    hooks.append(penultimate.register_forward_hook(lambda m, i, o: actor_mid_hook(m, i, o)))
                # true pre-logits: input to final Linear layer
                final_linear = seq[-1]
                def prelogits_hook(m, inp):
                    x = inp[0]
                    capture['actor_prelogits'] = x.detach().cpu().numpy()
                hooks.append(final_linear.register_forward_pre_hook(lambda m, inp: prelogits_hook(m, inp)))
        except Exception:
            pass

    # critic input/output via model.forward; easier: we capture value from top-level forward call
    # ltl_net output (sequence embedding e_sigma)
    if hasattr(model, 'ltl_net') and model.ltl_net is not None:
        def ltl_hook(m, inp, out):
            capture['e_sigma'] = out.detach().cpu().numpy()
        hooks.append(model.ltl_net.register_forward_hook(lambda m, i, o: ltl_hook(m, i, o)))
        # also capture GRU hidden explicitly if available
        try:
            if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
                def gru_hook(m, i, o):
                    try:
                        out_seq, h = o
                        capture['h_seq'] = h[-1].detach().cpu().numpy()
                    except Exception:
                        pass
                hooks.append(model.ltl_net.rnn.register_forward_hook(lambda m, i, o: gru_hook(m, i, o)))
        except Exception:
            pass

    # env_net output (observation embedding)
    if hasattr(model, 'env_net') and model.env_net is not None:
        def env_hook(m, inp, out):
            capture['obs_emb'] = out.detach().cpu().numpy()
        hooks.append(model.env_net.register_forward_hook(lambda m, i, o: env_hook(m, i, o)))

        # try to hook last conv pre-pool to capture spatial features
        try:
            import torch.nn as nn
            last_conv = None
            for name, module in model.env_net.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                def env_conv_hook(m, _, out):
                    arr = out.detach().cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    capture['obs_conv'] = arr
                hooks.append(last_conv.register_forward_hook(lambda m, i, o: env_conv_hook(m, i, o)))
        except Exception:
            pass

    # critic pre-head (critic_mid): input to final Linear layer
    try:
        crit_modules = list(model.critic.modules())
        # find last Linear in critic
        last_linear = None
        import torch.nn as nn
        for mod in crit_modules:
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is not None:
            def critic_pre_hook(m, inp):
                x = inp[0]
                capture['critic_mid'] = x.detach().cpu().numpy()
            hooks.append(last_linear.register_forward_pre_hook(lambda m, inp: critic_pre_hook(m, inp)))
    except Exception:
        pass

    return hooks


def assignment_set_to_vocab_ids(assignment_fset, vocab) -> list[int]:
    # Convert a frozenset of FrozenAssignment to int IDs if present in VOCAB; otherwise return []
    try:
        return [int(vocab[a]) for a in assignment_fset if a in vocab]
    except Exception:
        return []


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Prepare formulas
    if args.formulas:
        formula_list = [f.strip() for f in args.formulas.split(',') if f.strip()]
    else:
        formula_list = [args.formula]
    sampler = FixedSampler.partial(formula_list[0])
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)

    # Capture the trained vocabulary loaded by ModelStore
    from preprocessing.vocab import VOCAB as GLOBAL_VOCAB
    vocab = dict(GLOBAL_VOCAB)

    # Buffers
    logs: Dict[str, list] = {
        'obs_raw': [],
        'pos': [],
        'action': [],
        'logits': [],
        'value': [],
        'q': [],
        'A_plus_ids': [],
        'A_minus_ids': [],
        'e_sigma': [],
        'obs_emb': [],
        'obs_conv': [],
        'actor_in': [],
        'actor_mid': [],
        'actor_prelogits': [],
        'critic_mid': [],
        'h_seq': [],
        'next_pos': [],
        'episode': [],
        't': [],
    }

    capture: Dict[str, Any] = {}
    hooks = register_hooks(model, capture)

    episodes = args.episodes
    ep = 0
    for f_idx, current_formula in enumerate(formula_list):
        if f_idx > 0:
            try:
                env.close()
            except Exception:
                pass
            env = make_env(args.env, FixedSampler.partial(current_formula), render_mode=None)
        for _ in range(episodes):
            # Reset episode
            out = env.reset(seed=args.seed + ep)
            obs = out[0] if isinstance(out, (tuple, list)) else out
            agent.reset()
            done = False
            step_idx = 0

            # Derive current agent position (LetterEnv exposes agent tuple)
            def get_pos():
                try:
                    agent_pos = get_env_attr(env, 'agent')
                    return np.asarray(agent_pos, dtype=int)
                except Exception:
                    return None

            while not done:
                logs['episode'].append(ep)
                logs['t'].append(step_idx)
                logs.setdefault('formula', []).append(current_formula)
                step_idx += 1
                # Raw obs grid
                logs['obs_raw'].append(np.array(obs['features']))

                # Current pos and q (LDBA state)
                logs['pos'].append(get_pos())
                logs['q'].append(int(obs.get('ldba_state', -1)))

                # Select sequence σ* for the current obs via the agent's search and log A+ / A-
                try:
                    selected_seq = search(obs['ldba'], obs['ldba_state'], obs)
                except Exception:
                    selected_seq = None
                if selected_seq is not None and len(selected_seq) > 0:
                    # immediate planned step (sequence is encoded forward; preprocess will reverse internally)
                    reach, avoid = selected_seq[0]
                    logs['A_plus_ids'].append(assignment_set_to_vocab_ids(reach, vocab))
                    logs['A_minus_ids'].append(assignment_set_to_vocab_ids(avoid, vocab))
                else:
                    logs['A_plus_ids'].append([])
                    logs['A_minus_ids'].append([])
                    # Ensure we have a valid (possibly empty) sequence for preprocessing
                    selected_seq = tuple()

                # Forward pass to populate hooks and get logits/value
                capture.clear()
                with torch.inference_mode():
                    # Minimal preprocessing used by model.forward: we expect dictlist; construct quickly
                    from preprocessing.preprocessing import preprocess_obss
                    obs_for_forward = dict(obs)
                    # Always set a valid goal sequence (may be empty tuple)
                    obs_for_forward['goal'] = selected_seq
                    dl = preprocess_obss([obs_for_forward], props)
                    dist, value = model(dl)
                # Capture logits/value from the same dist
                if hasattr(dist, 'logits'):
                    logs['logits'].append(dist.logits.detach().cpu().numpy()[0])
                elif hasattr(dist, 'probs'):
                    probs = dist.probs.detach().cpu().numpy()[0]
                    logs['logits'].append(np.log(probs + 1e-12))
                else:
                    logs['logits'].append(None)
                logs['value'].append(float(value.detach().cpu().numpy()[0]))

                # Capture embeddings
                def safe0(x):
                    return x[0] if (isinstance(x, np.ndarray) and x.ndim >= 1) else None
                logs['e_sigma'].append(safe0(capture.get('e_sigma')))
                logs['obs_emb'].append(safe0(capture.get('obs_emb')))
                logs['obs_conv'].append(safe0(capture.get('obs_conv')))
                logs['actor_in'].append(safe0(capture.get('actor_in')))
                logs['actor_mid'].append(safe0(capture.get('actor_mid')))
                logs['actor_prelogits'].append(safe0(capture.get('actor_prelogits')))
                logs['critic_mid'].append(safe0(capture.get('critic_mid')))
                # Prefer GRU hidden; fallback to e_sigma
                h_val = safe0(capture.get('h_seq'))
                if h_val is None:
                    h_val = safe0(capture.get('e_sigma'))
                logs['h_seq'].append(h_val)

                # Choose action from the same dist for consistency
                try:
                    if args.deterministic and hasattr(dist, 'mode'):
                        action_t = dist.mode()
                    elif args.deterministic and hasattr(dist, 'logits'):
                        action_t = torch.argmax(dist.logits, dim=-1)
                    else:
                        action_t = dist.sample()
                    action = int(action_t.flatten()[0].item())
                except Exception:
                    # fallback: argmax over last logged logits
                    last_logits = logs['logits'][-1]
                    action = int(np.argmax(last_logits)) if last_logits is not None else 0
                logs['action'].append(action)

                # Step env
                step_out = env.step(action)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                    obs, reward, term, trunc, info = step_out
                    done = bool(term or trunc)
                else:
                    obs, reward, done, info = step_out

                # Next pos
                logs['next_pos'].append(get_pos())
            ep += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert lists to arrays where possible
    np_logs = {}
    for k, v in logs.items():
        try:
            np_logs[k] = np.array(v, dtype=object if any(isinstance(x, (list, tuple)) for x in v) else None)
        except Exception:
            np_logs[k] = np.array(v, dtype=object)

    np.savez_compressed(out_path, **np_logs)
    print(f"Saved {out_path}")

    # Close env
    try:
        env.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()


