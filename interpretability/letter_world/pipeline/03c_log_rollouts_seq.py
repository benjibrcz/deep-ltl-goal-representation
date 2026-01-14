#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from model.model import build_model
from config import model_configs
from preprocessing.preprocessing import preprocess_obss


def detect_tA_tB(letter_id_seq: List[int]) -> Tuple[int, int, int]:
    """
    Returns (tA, tB, first_letter_id). tA is first index with any letter (>=0).
    tB is first index > tA with a different letter id. (-1,-1,None) if not found.
    """
    tA = -1; first_letter = None
    for t, lid in enumerate(letter_id_seq):
        if int(lid) != -1:
            tA, first_letter = t, int(lid)
            break
    if tA < 0:
        return -1, -1, None
    tB = -1
    for t in range(tA + 1, len(letter_id_seq)):
        lid = int(letter_id_seq[t])
        if lid != -1 and lid != first_letter:
            tB = t
            break
    return tA, tB, first_letter


def episode_satisfies_minimums(letter_id_seq: List[int], min_pre: int, min_post: int) -> bool:
    tA, tB, _ = detect_tA_tB(letter_id_seq)
    if tA < 0 or tB < 0:
        return False
    return (tA >= int(min_pre)) and ((tB - tA) >= int(min_post))


def register_hooks(model, capture: dict, log_all_env: bool = False, log_all_ltl: bool = False):
    hooks = []
    # actor penultimate (actor_mid)
    try:
        if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
            seq = model.actor.model
            # prelogits input
            final_linear = seq[-1]
            def prelogits_hook(m, inp):
                x = inp[0]
                capture['actor_prelogits'] = x.detach().cpu().numpy()
            hooks.append(final_linear.register_forward_pre_hook(lambda m, inp: prelogits_hook(m, inp)))
            # penultimate output
            if len(seq) >= 2:
                penultimate = seq[-2]
                def actor_mid_hook(m, inp, out):
                    arr = out.detach().cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    capture['actor_mid'] = arr
                hooks.append(penultimate.register_forward_hook(lambda m, i, o: actor_mid_hook(m, i, o)))
    except Exception:
        pass
    # critic penultimate (critic_mid) and prelogits input (critic_prelogits)
    try:
        crit = getattr(model, 'critic', None)
        if crit is not None:
            # Resolve last Linear under critic
            last_linear = None
            penult_module = None
            # Try as Sequential with numeric indices
            if isinstance(crit, nn.Sequential) and len(crit) >= 1:
                for m in crit:
                    pass
                # find last Linear
                linear_idxs = [i for i, m in enumerate(crit) if isinstance(m, nn.Linear)]
                if linear_idxs:
                    li = linear_idxs[-1]
                    last_linear = crit[li]
                    if li - 1 >= 0:
                        penult_module = crit[li - 1]
            # Fallback: scan named_modules
            if last_linear is None:
                for name, mod in crit.named_modules():
                    if isinstance(mod, nn.Linear):
                        last_linear = mod
                # crude penult fallback: whole critic
                penult_module = penult_module or crit

            if last_linear is not None:
                def critic_prelogits_hook(m, inp):
                    x = inp[0]
                    try:
                        arr = x.detach().cpu().numpy()
                    except Exception:
                        arr = np.asarray(x)
                    capture['critic_prelogits'] = arr
                hooks.append(last_linear.register_forward_pre_hook(lambda m, inp: critic_prelogits_hook(m, inp)))
            if penult_module is not None:
                def critic_mid_hook(m, inp, out):
                    try:
                        arr = out.detach().cpu().numpy()
                    except Exception:
                        arr = np.asarray(out)
                    if isinstance(arr, np.ndarray) and arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    capture['critic_mid'] = arr
                hooks.append(penult_module.register_forward_hook(lambda m, i, o: critic_mid_hook(m, i, o)))
    except Exception:
        pass
    # env_net output (alias to hook_env_mlp3)
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
    except Exception:
        pass
    # ltl_net RNN/GRU hidden (aliases: hook_ltl_rnn_h, hook_ltl_gru_h) and optional sequence output
    try:
        if hasattr(model, 'ltl_net'):
            rnn_mod = getattr(model.ltl_net, 'rnn', None)
            if rnn_mod is None:
                rnn_mod = getattr(model.ltl_net, 'gru', None)
            if rnn_mod is not None:
                def ltl_rnn_hook(m, inp, out):
                    # GRU returns (output, h), Transformer may differ; handle tuples
                    try:
                        if isinstance(out, (tuple, list)) and len(out) >= 2:
                            gru_out, h_n = out[0], out[1]
                        else:
                            gru_out, h_n = out, getattr(m, 'last_h', None)
                        # save sequence output if available
                        if gru_out is not None:
                            try:
                                go = gru_out.detach().cpu().numpy()
                            except Exception:
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
    except Exception:
        pass
    # Optionally: log all leaf submodules in env_net
    try:
        if log_all_env and hasattr(model, 'env_net') and model.env_net is not None:
            for name, mod in model.env_net.named_modules():
                if name == '':
                    continue
                if any(True for _ in mod.children()):
                    continue  # skip non-leaf to reduce duplication
                def make_hook(key):
                    def _hook(m, inp, out):
                        val = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                        try:
                            arr = val.detach().cpu().numpy()
                        except Exception:
                            arr = np.asarray(val)
                        capture[key] = arr
                    return _hook
                key = f"hook_env.{name}"
                hooks.append(mod.register_forward_hook(make_hook(key)))
    except Exception:
        pass
    # Optionally: log all leaf submodules in ltl_net (goal embedding network)
    try:
        if log_all_ltl and hasattr(model, 'ltl_net') and model.ltl_net is not None:
            for name, mod in model.ltl_net.named_modules():
                if name == '':
                    continue
                if any(True for _ in mod.children()):
                    continue
                def make_hook(key):
                    def _hook(m, inp, out):
                        val = out
                        if isinstance(m, nn.GRU):
                            # GRU returns (output, h_n)
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
                key = f"hook_ltl.{name}"
                hooks.append(mod.register_forward_hook(make_hook(key)))
    except Exception:
        pass
    return hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='LetterEnv-v0')
    ap.add_argument('--exp', type=str, default='test')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps_per_ep', type=int, default=100)
    ap.add_argument('--feature_key', type=str, default='actor_mid', help='actor_mid | actor_prelogits | critic_mid | critic_prelogits | hook_env_mlp3 | hook_ltl_rnn_h | hook_ltl_gru_h')
    ap.add_argument('--formula', type=str, default='F a & F b', help='LTL formula for the sampler, e.g., "F a then F b"')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/datasets/sr_rollouts.npz')
    ap.add_argument('--min-pre', type=int, default=0, help='Min steps from start to first letter A (tA).')
    ap.add_argument('--min-post', type=int, default=0, help='Min steps from A to B (tB - tA).')
    ap.add_argument('--max-resamples-per-episode', type=int, default=100, help='Max attempts to regenerate an episode that satisfies the constraints.')
    ap.add_argument('--progress', action='store_true', help='Show a progress bar over accepted episodes')
    ap.add_argument('--early_stop_on_B', action='store_true', help='Stop episode when A and B minima are satisfied')
    ap.add_argument('--log_all_env', action='store_true', help='Capture all env_net leaf layer activations')
    ap.add_argument('--log_all_ltl', action='store_true', help='Capture all ltl_net leaf layer activations')
    ap.add_argument('--save_obs', action='store_true', help='Also save raw observations per step as obs_raw')
    args = ap.parse_args()

    sampler = FixedSampler.partial(args.formula)
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    capture = {}
    hooks = register_hooks(model, capture, log_all_env=args.log_all_env, log_all_ltl=args.log_all_ltl)
    # precompute propositions set once
    try:
        props = set(env.get_propositions())
    except Exception:
        props = set()

    feats = []
    actions = []
    ep_ids = []
    positions = []
    base_ids = []
    letters_now = []
    obs_list = []

    accepted = 0
    rejected = 0
    ep = 0
    # optional progress bar
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=args.episodes, desc='03c rollouts (accepted)')
        except Exception:
            pbar = None
    # global hook buffers (stepwise across accepted episodes)
    global_hook_buffers = {}

    while ep < args.episodes:
        # resample loop
        attempts = 0
        letter_id_buffer = []
        feature_buffer = []
        action_buffer = []
        pos_buffer = []
        baseid_buffer = []
        ep_hook_buf = {}
        out = env.reset(seed=args.seed + ep + attempts)
        obs = out[0] if isinstance(out, (tuple, list)) else out
        done = False
        steps = 0
        tA_local = -1
        first_letter_local = None
        satisfied = False
        while not done and steps < args.steps_per_ep:
            # record current position from base env
            base_env = env
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            agent_xy = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
            pos_buffer.append(np.array(agent_xy, dtype=np.int64))
            # current letter id (if standing on a letter); else -1
            cur_letter = -1
            if hasattr(base_env, 'map') and agent_xy in getattr(base_env, 'map'):
                letter = base_env.map[agent_xy]
                # map to index within env letter_types if available
                try:
                    idx = int(getattr(base_env, 'letter_types').index(letter))
                except Exception:
                    idx = -1
                cur_letter = idx
            letter_id_buffer.append(int(cur_letter))
            baseid_buffer.append(len(base_ids) + len(baseid_buffer))

            # forward once
            capture.clear()
            # minimal empty reach/avoid pair to force LTL path to execute
            obs_forward = dict(obs)
            obs_forward['goal'] = ((tuple(), tuple()),)
            dl = preprocess_obss([obs_forward], props)
            with torch.inference_mode():
                dist, _ = model(dl)
            # feature
            fk = args.feature_key
            z = None
            if fk in capture:
                z = capture[fk][0]
            elif fk == 'actor_prelogits' and 'actor_prelogits' in capture:
                z = capture['actor_prelogits'][0]
            if z is None:
                raise RuntimeError(f'Feature {fk} not captured')
            feature_buffer.append(z.astype(np.float32))

            # stash all captured hook keys for this step
            for k, arr in capture.items():
                if k not in ep_hook_buf:
                    ep_hook_buf[k] = []
                try:
                    v = arr[0] if isinstance(arr, np.ndarray) and arr.ndim >= 2 else np.asarray(arr)
                except Exception:
                    v = np.asarray(arr)
                ep_hook_buf[k].append(v)

            # action from same dist
            try:
                if hasattr(dist, 'mode'):
                    a_t = dist.mode()
                elif hasattr(dist, 'logits'):
                    a_t = torch.argmax(dist.logits, dim=-1)
                else:
                    a_t = dist.sample()
                a = int(a_t.flatten()[0].item())
            except Exception:
                a = int(np.argmax(dist.logits.detach().cpu().numpy()[0])) if hasattr(dist, 'logits') else 0
            action_buffer.append(a)
            if args.save_obs:
                try:
                    obs_arr = np.array(obs['features'])
                    obs_list.append(obs_arr)
                except Exception:
                    pass

            # step env
            step_out = env.step(a)
            if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                obs, reward, term, trunc, _ = step_out
                done = bool(term or trunc)
            else:
                obs, reward, done, _ = step_out
            steps += 1

            # update local phase detection incrementally
            if tA_local < 0 and letter_id_buffer[-1] != -1:
                tA_local = len(letter_id_buffer) - 1
                first_letter_local = letter_id_buffer[-1]
            # early stop if satisfied minima and flag set
            if args.early_stop_on_B and tA_local >= 0 and (len(letter_id_buffer) - 1 - tA_local) >= args.min_post:
                # check if we have seen a different letter than first after tA
                seen_diff = any((lid != -1 and lid != first_letter_local) for lid in letter_id_buffer[tA_local+1:])
                if seen_diff and tA_local >= args.min_pre:
                    satisfied = True
                    break
            # early reject if impossible to satisfy remaining post length
            remaining = args.steps_per_ep - steps
            if args.min_post > 0 and tA_local < 0 and remaining < (args.min_post + 1):
                # even if we hit A next, not enough steps to reach post length
                break
            if args.min_post > 0 and tA_local >= 0:
                needed = args.min_post - max(0, len(letter_id_buffer) - 1 - tA_local)
                if remaining < needed:
                    break

        # constraints check
        if satisfied or episode_satisfies_minimums(letter_id_buffer, args.min_pre, args.min_post):
            accepted += 1
            # persist buffers
            feats.extend(feature_buffer)
            actions.extend(action_buffer)
            positions.extend(pos_buffer)
            letters_now.extend(letter_id_buffer)
            ep_ids.extend([ep] * len(feature_buffer))
            base_ids.extend(baseid_buffer)
            # persist hook buffers into global buffers
            for k, seq in ep_hook_buf.items():
                if k not in global_hook_buffers:
                    global_hook_buffers[k] = []
                global_hook_buffers[k].extend(seq)
            ep += 1
            if pbar is not None:
                pbar.update(1)
        else:
            rejected += 1
            # resample: try again without advancing ep index
            continue

    # save
    X = np.stack(feats, axis=0)
    A = np.asarray(actions, dtype=np.int64)
    E = np.asarray(ep_ids, dtype=np.int64)
    P = np.stack(positions, axis=0).astype(np.int64)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    save_payload = dict(feature_t=X, action=A, episode=E, agent_pos=P, base_id=np.asarray(base_ids, dtype=np.int64), letter_id=np.asarray(letters_now, dtype=np.int64))
    if args.save_obs and len(obs_list) == len(feats):
        try:
            save_payload['obs_raw'] = np.stack(obs_list, axis=0)
        except Exception:
            pass
    # attach captured hooks
    for k, seq in global_hook_buffers.items():
        if len(seq) == 0:
            continue
        arr = np.asarray(seq)
        if arr.ndim == 1:
            arr = arr[:, None]
        save_payload[k] = arr
    np.savez_compressed(outp, **save_payload)
    print('Saved sequential rollouts to', outp)
    total_attempted = accepted + rejected
    print(f"[min-phase] accepted={accepted}  rejected={rejected}  accept_rate={accepted / max(1,total_attempted):.3f}  (min_pre={args.min_pre}, min_post={args.min_post}, max_resamples_per_episode={args.max_resamples_per_episode})")

    # cleanup
    for h in hooks:
        try: h.remove()
        except Exception: pass
    env.close()
    if pbar is not None:
        try: pbar.close()
        except Exception: pass


if __name__ == '__main__':
    main()


