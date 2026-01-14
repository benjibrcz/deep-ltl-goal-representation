#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch

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
    ap.add_argument('--formula', type=str, default='F a')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--branch_every', type=int, default=10)
    ap.add_argument('--features', type=str, default='obs_local_flat,obs_conv,obs_emb,actor_prelogits,actor_mid,critic_mid,critic_prelogits,hook_ltl_rnn_h', help='Comma list: obs_local_flat,obs_conv,obs_emb,actor_prelogits,actor_mid,critic_mid,critic_prelogits,hook_ltl_rnn_h')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/datasets/branched_one_step.npz')
    ap.add_argument('--clean_out', type=str, default=None, help='Optional path to also write a CLEAN compact NPZ')
    ap.add_argument('--clean_use_ego', action='store_true', help='If set, CLEAN feature_t will use ego_now flattened')
    args = ap.parse_args()

    sampler = FixedSampler.partial(args.formula)
    env = make_env(args.env, sampler, render_mode=None)
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    data = {k: [] for k in ['obs_raw', 'feature', 'feature_base', 'feature_next', 'action', 'next_obs_raw', 'branched_action', 'episode', 'agent_pos', 'agent_pos_next', 'ego_now', 'ego_next', 'source_id']}

    out = env.reset(seed=args.seed)
    obs = out[0] if isinstance(out, (tuple, list)) else out
    done = False
    t = 0; ep = 0
    # Setup hooks for capturing features
    capture = {}
    hooks = []
    # env_net outputs
    if hasattr(model, 'env_net') and model.env_net is not None:
        def env_hook(m, inp, out):
            try:
                arr = out.detach().cpu().numpy()
                if arr.ndim == 4:
                    arr = arr.reshape(arr.shape[0], -1)
                capture['obs_emb'] = arr
            except Exception:
                pass
        hooks.append(model.env_net.register_forward_hook(lambda m, i, o: env_hook(m, i, o)))
        # last conv
        try:
            import torch.nn as nn
            last_conv = None
            for name, module in model.env_net.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                def conv_hook(m, inp, out):
                    arr = out.detach().cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr.reshape(arr.shape[0], -1)
                    capture['obs_conv'] = arr
                hooks.append(last_conv.register_forward_hook(lambda m, i, o: conv_hook(m, i, o)))
        except Exception:
            pass
    # actor prelogits
    try:
        if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
            seq = model.actor.model
            final_linear = seq[-1]
            def prelogits_hook(m, inp):
                x = inp[0]
                capture['actor_prelogits'] = x.detach().cpu().numpy()
            hooks.append(final_linear.register_forward_pre_hook(lambda m, inp: prelogits_hook(m, inp)))
            # actor mid (penultimate module output)
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

    # critic prelogits + penultimate (critic_mid)
    try:
        import torch.nn as nn
        crit = getattr(model, 'critic', None)
        if crit is not None:
            last_linear = None
            penult_module = None
            if isinstance(crit, nn.Sequential) and len(crit) >= 1:
                linear_idxs = [i for i, m in enumerate(crit) if isinstance(m, nn.Linear)]
                if linear_idxs:
                    li = linear_idxs[-1]
                    last_linear = crit[li]
                    if li - 1 >= 0:
                        penult_module = crit[li - 1]
            if last_linear is None:
                for _, mod in crit.named_modules():
                    if isinstance(mod, nn.Linear):
                        last_linear = mod
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

    # ltl_net hidden (hook_ltl_rnn_h)
    try:
        ltl = getattr(model, 'ltl_net', None)
        if ltl is not None:
            rnn_mod = getattr(ltl, 'rnn', None) or getattr(ltl, 'gru', None)
            if rnn_mod is not None:
                def ltl_rnn_hook(m, inp, out):
                    try:
                        if isinstance(out, (tuple, list)) and len(out) >= 2:
                            h_n = out[1]
                        else:
                            h_n = getattr(m, 'last_h', None)
                        if h_n is not None:
                            try:
                                h_np = h_n[-1].detach().cpu().numpy() if hasattr(h_n, 'detach') else np.asarray(h_n)
                            except Exception:
                                h_np = np.asarray(h_n)
                            capture['hook_ltl_rnn_h'] = h_np
                    except Exception:
                        pass
                hooks.append(rnn_mod.register_forward_hook(lambda m, i, o: ltl_rnn_hook(m, i, o)))
    except Exception:
        pass

    wanted = [s.strip() for s in args.features.split(',') if s.strip()]

    sid = 0  # source state identifier

    # helper to access underlying LetterEnv for state restore
    def get_base_env(e):
        base = e
        while hasattr(base, 'env'):
            base = base.env
        return base
    while ep < 100 and t < args.steps:
        # log current
        data['obs_raw'].append(np.array(obs['features']))
        # get features via forward + hooks
        dl = preprocess_obss([dict(obs, goal=((tuple(), tuple()),))], set(env.get_propositions()))
        capture.clear()
        with torch.inference_mode():
            dist, _ = model(dl)
        # Build feature vector at time t
        feats = []
        if 'obs_local_flat' in wanted:
            local = np.array(obs['features']).reshape(-1)
            feats.append(local)
        if 'obs_conv' in wanted and 'obs_conv' in capture:
            feats.append(capture['obs_conv'][0])
        if 'obs_emb' in wanted and 'obs_emb' in capture:
            feats.append(capture['obs_emb'][0])
        if 'actor_mid' in wanted and 'actor_mid' in capture:
            feats.append(capture['actor_mid'][0])
        if 'actor_prelogits' in wanted and 'actor_prelogits' in capture:
            feats.append(capture['actor_prelogits'][0])
        if 'critic_mid' in wanted and 'critic_mid' in capture:
            feats.append(capture['critic_mid'][0])
        if 'critic_prelogits' in wanted and 'critic_prelogits' in capture:
            feats.append(capture['critic_prelogits'][0])
        if 'hook_ltl_rnn_h' in wanted and 'hook_ltl_rnn_h' in capture:
            arr = capture['hook_ltl_rnn_h'][0] if getattr(capture['hook_ltl_rnn_h'], 'ndim', 0) >= 2 else np.asarray(capture['hook_ltl_rnn_h'])
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            feats.append(arr)
        feat_vec = np.concatenate(feats, axis=0) if feats else None

        # helper: egocentric crop around agent (r=3 for 7x7)
        def to_egocentric(grid_hw_c: np.ndarray, agent_xy: tuple[int, int], r: int = 3):
            H, W, C = grid_hw_c.shape
            i, j = int(agent_xy[0]), int(agent_xy[1])
            rows = [((i + dr) % H) for dr in range(-r, r + 1)]
            cols = [((j + dc) % W) for dc in range(-r, r + 1)]
            return grid_hw_c[np.ix_(rows, cols, range(C))]

        # record egocentric now (agent pos per branch logged below)
        base_env = get_base_env(env)
        agent_xy = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
        # snapshot base world state for branching
        map_snapshot = dict(getattr(base_env, 'map', {})) if hasattr(base_env, 'map') else {}
        data['ego_now'].append(to_egocentric(np.array(obs['features']), agent_xy))

        # branch every K steps
        if t % args.branch_every == 0:
            # new base state; assign a source_id for all its branches
            current_sid = sid
            sid += 1
            base_vec = feat_vec
            for a in range(4):
                # clone env if supported; otherwise step and immediately reset back via saved state
                # We approximate branching by taking a step, logging, and then undoing via env.reset(seed+offset)
                # log base agent position per branch for alignment
                data['agent_pos'].append(np.array(agent_xy))
                # restore base world state before each branch
                if hasattr(base_env, 'map'):
                    base_env.map = dict(map_snapshot)
                if hasattr(base_env, 'agent'):
                    base_env.agent = tuple(agent_xy)
                step_out = env.step(a)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                    obs2, reward, term, trunc, _ = step_out
                    done2 = bool(term or trunc)
                else:
                    obs2, reward, done2, _ = step_out
                data['branched_action'].append(a)
                data['next_obs_raw'].append(np.array(obs2['features']))
                data['episode'].append(ep)
                data['feature'].append(feat_vec)
                data['feature_base'].append(base_vec)
                data['source_id'].append(current_sid)
                # capture next feature at same site (e.g., actor_mid)
                capture.clear()
                dl2 = preprocess_obss([dict(obs2, goal=((tuple(), tuple()),))], set(env.get_propositions()))
                with torch.inference_mode():
                    _dist2, _ = model(dl2)
                if 'actor_mid' in capture:
                    data['feature_next'].append(capture['actor_mid'][0])
                elif 'actor_prelogits' in capture:
                    data['feature_next'].append(capture['actor_prelogits'][0])
                elif 'critic_mid' in capture:
                    data['feature_next'].append(capture['critic_mid'][0])
                elif 'critic_prelogits' in capture:
                    data['feature_next'].append(capture['critic_prelogits'][0])
                else:
                    data['feature_next'].append(None)
                # record next agent pos and egocentric next
                agent_xy2 = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
                data['agent_pos_next'].append(np.array(agent_xy2))
                data['ego_next'].append(to_egocentric(np.array(obs2['features']), agent_xy2))
        # follow policy to progress (choose once from current dist)
        a0 = int(np.argmax(dist.logits.detach().cpu().numpy()[0])) if hasattr(dist, 'logits') else 0
        data['action'].append(a0)
        # restore base state, then step with chosen action to progress env
        if hasattr(base_env, 'map'):
            base_env.map = dict(map_snapshot)
        if hasattr(base_env, 'agent'):
            base_env.agent = tuple(agent_xy)
        step_out = env.step(a0)
        if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
            obs, reward, term, trunc, _ = step_out
            done = bool(term or trunc)
        else:
            obs, reward, done, _ = step_out
        if done:
            try:
                env.reset(seed=args.seed + ep + 1)
            except Exception:
                env.reset()
            ep += 1
            done = False
        t += 1

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **{k: np.array(v, dtype=object) for k, v in data.items()})
    print(f"Saved branched one-step dataset to {outp}")

    # Optional: also write a cleaned, compact file compatible with probe loaders
    if args.clean_out:
        D = np.load(outp, allow_pickle=True)
        def as_2d_float(x):
            if x is None: return None
            x = np.array(x, dtype=object)
            if x.dtype == object:
                x = np.vstack([np.asarray(r).ravel() for r in x]).astype(np.float32)
            else:
                x = np.asarray(x)
                if x.ndim == 1: x = x[:, None]
                if x.ndim > 2: x = x.reshape(x.shape[0], -1)
                x = x.astype(np.float32)
            return x
        # Prefer ego views if present for features; and use feature_base for hashing bases
        X = D['feature'] if 'feature' in D.files else None
        X_base = D['feature_base'] if 'feature_base' in D.files else X
        EGO_NOW = D['ego_now'] if 'ego_now' in D.files else None
        EGO_NEXT = D['ego_next'] if 'ego_next' in D.files else None
        NEXT_RAW = D['next_obs_raw'] if 'next_obs_raw' in D.files else None
        A = D['branched_action'] if 'branched_action' in D.files else (D['action'] if 'action' in D.files else None)
        POS_NOW = D['agent_pos'] if 'agent_pos' in D.files else None
        POS_NEXT = D['agent_pos_next'] if 'agent_pos_next' in D.files else None
        # Coerce
        X = as_2d_float(X) if X is not None else None
        Xb = as_2d_float(X_base) if X_base is not None else None
        A = np.asarray(A).astype(np.int64) if A is not None else None
        Y_next = EGO_NEXT if EGO_NEXT is not None else NEXT_RAW
        if Y_next is not None:
            Y_arr = np.asarray(Y_next)
            if isinstance(Y_arr, np.ndarray) and Y_arr.dtype == object:
                # stack object-array of arrays into dense float tensor
                Y_next = np.stack(list(Y_next), axis=0).astype(np.float32)
            else:
                Y_next = Y_arr.astype(np.float32)
        # Optionally use ego_now as features (image to flat)
        if args.clean_use_ego and EGO_NOW is not None:
            EN = np.asarray(EGO_NOW)
            X = EN.reshape(EN.shape[0], -1).astype(np.float32)
        # Align lengths
        Ns = []
        if X is not None: Ns.append(len(X))
        if Xb is not None: Ns.append(len(Xb))
        if A is not None: Ns.append(len(A))
        if Y_next is not None: Ns.append(len(Y_next))
        if Ns:
            N = min(Ns)
            X = X[:N]
            Xb = Xb[:N] if Xb is not None else X
            A = A[:N]
            Y_next = Y_next[:N]
            if POS_NOW is not None:
                POS_NOW = np.asarray(POS_NOW)[:N]
            if POS_NEXT is not None:
                POS_NEXT = np.asarray(POS_NEXT)[:N]
            # Build/repair base_id by hashing X rows
            r = np.ascontiguousarray(np.round(Xb, 6))
            h = r.view(np.dtype((np.void, r.shape[1]*r.dtype.itemsize)))
            _, base_id = np.unique(h, return_inverse=True)
            clean_path = Path(args.clean_out)
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            # optional next feature for contrastive probe
            FEAT_NEXT = D['feature_next'] if 'feature_next' in D.files else None
            if FEAT_NEXT is not None:
                FN = np.array(FEAT_NEXT, dtype=object)
                try:
                    FN = np.vstack([np.asarray(r).ravel() for r in FN])
                except Exception:
                    FN = np.asarray(FN)
                    if FN.ndim > 2:
                        FN = FN.reshape(FN.shape[0], -1)
                FN = FN[:N].astype(np.float32)
            else:
                FN = np.array([], dtype=np.float32)

            np.savez_compressed(clean_path,
                                feature_t=X,
                                action=A,
                                obs_next_raw=Y_next,
                                base_id=base_id,
                                agent_pos=POS_NOW if POS_NOW is not None else np.array([]),
                                agent_pos_next=POS_NEXT if POS_NEXT is not None else np.array([]),
                                feature_next=FN)
            print("Wrote CLEAN dataset to", clean_path)


if __name__ == '__main__':
    main()


