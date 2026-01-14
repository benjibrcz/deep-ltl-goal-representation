#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch.nn as nn
import numpy as np
import torch

from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from model.model import build_model
from config import model_configs
from preprocessing.preprocessing import preprocess_obss
from preprocessing.vocab import VOCAB
from ltl.logic.assignment import FrozenAssignment


# ---------- Phase helpers (as in your script) ----------

def detect_tA_tB(letter_id_seq: List[int]) -> Tuple[int, int, int]:
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


# ---------- Hook capture (copied & lightly refactored) ----------

def register_hooks(model, capture: dict, log_all_env: bool = False, log_all_ltl: bool = False):
    hooks = []
    # actor prelogits + penultimate (actor_mid)
    try:
        if hasattr(model.actor, 'model') and hasattr(model.actor.model, '__getitem__'):
            seq = model.actor.model
            final_linear = seq[-1]
            def prelogits_hook(m, inp):
                x = inp[0]
                capture['actor_prelogits'] = x.detach().cpu().numpy()
            hooks.append(final_linear.register_forward_pre_hook(lambda m, inp: prelogits_hook(m, inp)))

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

    # env_net (hook_env_mlp3)
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
            # Optionally record all leaf submodules of env_net
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

    # ltl_net RNN hidden (hook_ltl_rnn_h) + optional all LTL leaves
    try:
        if hasattr(model, 'ltl_net') and model.ltl_net is not None:
            # Prefer 'rnn' attribute (used in this repo); fall back to 'gru' if present
            rnn_mod = getattr(model.ltl_net, 'rnn', None)
            if rnn_mod is None:
                rnn_mod = getattr(model.ltl_net, 'gru', None)

            if rnn_mod is not None:
                def ltl_rnn_hook(m, inp, out):
                    try:
                        # nn.GRU returns (output, h)
                        if isinstance(out, (tuple, list)) and len(out) >= 2:
                            gru_out, h_n = out[0], out[1]
                        else:
                            gru_out, h_n = out, getattr(m, 'last_h', None)
                        # Save sequence output and final hidden
                        try:
                            capture['hook_ltl_gru_out'] = gru_out.detach().cpu().numpy()
                        except Exception:
                            capture['hook_ltl_gru_out'] = np.asarray(gru_out)
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

            # Optionally record all leaf submodules of ltl_net (regardless of rnn/gru attr)
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


# ---------- Goal sweep utilities ----------

def letters_from_arg(arg: str) -> List[str]:
    if arg == "auto12":
        return [chr(ord('a') + i) for i in range(12)]  # a..l
    if "," in arg:
        return [s.strip().lower() for s in arg.split(",") if s.strip()]
    if "-" in arg:
        lo, hi = arg.split("-")
        lo = lo.strip().lower(); hi = hi.strip().lower()
        return [chr(c) for c in range(ord(lo), ord(hi) + 1)]
    return [arg.strip().lower()]

def make_formula(mode: str, a: str, b: str = None, avoid: str = None) -> str:
    """
    mode in {'reach', 'avoid', 'reach_avoid', 'reach2', 'reach2_avoid'}
    F x  = eventually visit x
    G !y = always avoid y
    'F a & F b' is unordered conjunction; if your sampler supports ordered, adapt here.
    """
    parts = []
    if mode in ("reach", "reach_avoid"):
        parts.append(f"F {a}")
    if mode in ("reach2", "reach2_avoid"):
        assert b is not None and b != a, "reach2 requires distinct a,b"
        parts.append(f"F {a}")
        parts.append(f"F {b}")
    if mode in ("avoid", "reach_avoid", "reach2_avoid"):
        assert avoid is not None and avoid not in (a, b), "avoid letter must differ"
        parts.append(f"G ! {avoid}")
    return " & ".join(parts)

def letter_to_idx(letter: str, alphabet: List[str]) -> int:
    try:
        return alphabet.index(letter.lower())
    except ValueError:
        return -1


def build_goal_seq(mode: str, reach_tok: Optional[str], reach2_tok: Optional[str], avoid_tok: Optional[str]):
    """
    Build a tuple of (reach_set, avoid_set) pairs using token strings
    matching preprocess_obss expectations (iterables of proposition tokens).
    """
    seq = []
    if mode in ("reach", "reach_avoid"):
        r = tuple([reach_tok]) if (reach_tok is not None) else tuple()
        av = tuple([avoid_tok]) if (mode == "reach_avoid" and (avoid_tok is not None)) else tuple()
        seq.append((r, av))
    elif mode in ("avoid",):
        r = tuple()
        av = tuple([avoid_tok]) if (avoid_tok is not None) else tuple()
        seq.append((r, av))
    elif mode in ("reach2", "reach2_avoid"):
        r1 = tuple([reach_tok]) if (reach_tok is not None) else tuple()
        av1 = tuple([avoid_tok]) if (mode == "reach2_avoid" and (avoid_tok is not None)) else tuple()
        seq.append((r1, av1))
        r2 = tuple([reach2_tok]) if (reach2_tok is not None) else tuple()
        av2 = tuple([avoid_tok]) if (mode == "reach2_avoid" and (avoid_tok is not None)) else tuple()
        seq.append((r2, av2))
    return tuple(seq)


def resolve_letter_token(letter: Optional[str], vocab_keys=None, props=None):
    """Map a letter like 'a' to a FrozenAssignment token from the VOCAB, if available.
    We only return FrozenAssignment objects (never raw strings) so downstream indexing into VOCAB works.
    """
    if letter is None:
        return None
    target = str(letter).strip().lower()
    # Search only FrozenAssignment keys in the vocab
    if vocab_keys is not None:
        for k in vocab_keys:
            if isinstance(k, FrozenAssignment):
                s = str(k).strip()
                if s.lower() == target or s.upper() == target.upper():
                    return k
                if s.lower().endswith(target) or s.lower().startswith(target):
                    return k
    return None


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', type=str, default='LetterEnv-v0')
    ap.add_argument('--exp', type=str, default='test')
    ap.add_argument('--seed', type=int, default=1)

    # Sweep controls
    ap.add_argument('--letters', type=str, default='auto12',
                    help="Comma list (e.g. 'a,b,c'), range 'a-l', or 'auto12' for a..l")
    ap.add_argument('--modes', type=str, default='reach,avoid,reach_avoid,reach2,reach2_avoid',
                    help="Comma list of modes to include")
    ap.add_argument('--episodes_per_combo', type=int, default=40)
    ap.add_argument('--steps_per_ep', type=int, default=60)

    # Constraints / early stop like your script
    ap.add_argument('--min-pre', type=int, default=3)
    ap.add_argument('--min-post', type=int, default=3)
    ap.add_argument('--early_stop_on_B', action='store_true')

    # Features / hooks
    ap.add_argument('--feature_key', type=str, default='actor_mid',
                    help='Which captured key to store as feature_t (actor_mid | actor_prelogits | hook_env_mlp3 | hook_ltl_rnn_h)')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/datasets/letter_sweep_hooks.npz')
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--log_all_env', action='store_true', help='Capture all env_net leaf layer activations')
    ap.add_argument('--log_all_ltl', action='store_true', help='Capture all ltl_net leaf layer activations')

    args = ap.parse_args()

    letters = letters_from_arg(args.letters)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # Build all goal combos we’ll generate
    combos = []
    for m in modes:
        if m == "reach":
            for a in letters:
                combos.append(dict(mode=m, a=a, b=None, avoid=None))
        elif m == "avoid":
            for z in letters:
                combos.append(dict(mode=m, a=None, b=None, avoid=z))
        elif m == "reach_avoid":
            for a in letters:
                for z in letters:
                    if z != a:
                        combos.append(dict(mode=m, a=a, b=None, avoid=z))
        elif m == "reach2":
            for i, a in enumerate(letters):
                for j, b in enumerate(letters):
                    if j > i:
                        combos.append(dict(mode=m, a=a, b=b, avoid=None))
        elif m == "reach2_avoid":
            for i, a in enumerate(letters):
                for j, b in enumerate(letters):
                    if j > i:
                        for z in letters:
                            if z not in (a, b):
                                combos.append(dict(mode=m, a=a, b=b, avoid=z))
        else:
            raise ValueError(f"Unknown mode: {m}")

    # Environment / model
    env = make_env(args.env, FixedSampler.partial("F a"), render_mode=None)  # dummy formula to init
    cfg = model_configs[args.env]
    store = ModelStore(args.env, args.exp, args.seed)
    # Populate global VOCAB
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    model = build_model(env, status, cfg).eval()

    # Props once
    try:
        props = set(env.get_propositions())
    except Exception:
        props = set()
    vocab_keys = set(VOCAB.keys())
    
    capture: Dict[str, np.ndarray] = {}
    hooks = register_hooks(model, capture, log_all_env=args.log_all_env, log_all_ltl=args.log_all_ltl)

    # Storage
    feats = []
    actions = []
    ep_ids = []
    positions = []
    base_ids = []
    letters_now = []

    # Hook time-series per step (only keys that actually appear)
    hook_buffers = {}

    # Episode-level labels per step
    goal_mode = []
    goal_reach = []
    goal_reach2 = []
    goal_avoid = []
    goal_formula = []

    letters_vocab = letters  # a..l
    letters_to_idx = {ch: i for i, ch in enumerate(letters_vocab)}

    # Optional progress
    pbar = None
    if args.progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(combos) * args.episodes_per_combo, desc='letter sweep (accepted)')
        except Exception:
            pbar = None

    global_step_base = 0
    rng_seed = args.seed

    def ensure_buffer_key(k: str, arr: np.ndarray):
        """Create per-step list for a captured hook key."""
        if k not in hook_buffers:
            hook_buffers[k] = []

    for combo in combos:
        mode, a, b, z = combo['mode'], combo['a'], combo['b'], combo['avoid']
        formula = make_formula(mode, a, b, z)
        # Rebuild env sampler to use the new formula
        env.sampler = FixedSampler.partial(formula)

        # Determine episode-level numeric labels (map a,b,z to indices in letters_vocab; -1 if N/A)
        # Resolve letter tokens against vocab/props
        reach_tok  = resolve_letter_token(a, vocab_keys, props) if a is not None else None
        reach2_tok = resolve_letter_token(b, vocab_keys, props) if b is not None else None
        avoid_tok  = resolve_letter_token(z, vocab_keys, props) if z is not None else None
        # Fallback: pick distinct valid tokens from vocab/props to ensure LTL path executes
        # Robust fallback: choose distinct FrozenAssignment tokens from VOCAB
        fa_pool = [k for k in vocab_keys if isinstance(k, FrozenAssignment)]
        # keep deterministic order by string repr
        fa_pool.sort(key=lambda x: str(x))
        if reach_tok is None and fa_pool:
            reach_tok = fa_pool[0]
        if reach2_tok is None:
            for k in fa_pool:
                if reach_tok is None or k != reach_tok:
                    reach2_tok = k
                    break
        if avoid_tok is None:
            for k in fa_pool:
                if k not in (reach_tok, reach2_tok):
                    avoid_tok = k
                    break
        # Also keep numeric ids for per-step labels in the NPZ
        reach_id  = letter_to_idx(a, letters_vocab) if a is not None else -1
        reach2_id = letter_to_idx(b, letters_vocab) if b is not None else -1
        avoid_id  = letter_to_idx(z, letters_vocab) if z is not None else -1

        accepted = 0
        while accepted < args.episodes_per_combo:
            # rollout buffers
            letter_id_seq = []
            feature_seq = []
            action_seq = []
            pos_seq = []
            baseid_seq = []
            # per-episode hook capture buffers (stepwise)
            ep_hook_buf: Dict[str, list] = {}

            out = env.reset(seed=rng_seed); rng_seed += 1
            obs = out[0] if isinstance(out, (tuple, list)) else out
            done = False
            steps = 0
            tA_local = -1
            first_letter_local = None
            satisfied = False

            while not done and steps < args.steps_per_ep:
                # raw base env
                base_env = env
                while hasattr(base_env, 'env'):
                    base_env = base_env.env

                agent_xy = tuple(getattr(base_env, 'agent', (0, 0))) if hasattr(base_env, 'agent') else (0, 0)
                pos_seq.append(np.array(agent_xy, dtype=np.int64))

                # current letter id at agent pos wrt letters_vocab (a..l), else -1
                cur_letter = -1
                if hasattr(base_env, 'map') and agent_xy in getattr(base_env, 'map'):
                    letter = str(base_env.map[agent_xy]).lower()
                    cur_letter = letters_to_idx.get(letter, -1)
                letter_id_seq.append(int(cur_letter))
                baseid_seq.append(global_step_base + len(baseid_seq))

                # forward pass + capture
                capture.clear()
                # Goal forwarding: construct a proper (reach, avoid) pair sequence from indices
                obs_forward = dict(obs)
                obs_forward['goal'] = build_goal_seq(mode, reach_tok, reach2_tok, avoid_tok)
                try:
                    dl = preprocess_obss([obs_forward], props)
                except Exception:
                    # Fallback: minimal one-step empty (reach, avoid) pair to force LTL path
                    obs_forward['goal'] = ((tuple(), tuple()),)
                    dl = preprocess_obss([obs_forward], props)
                with torch.inference_mode():
                    dist, _ = model(dl)

                # choose feature_t from captured keys
                fk = args.feature_key
                if fk not in capture:
                    # fallbacks: often actor_prelogits or actor_mid will be present
                    if fk == 'actor_prelogits' and 'actor_prelogits' in capture:
                        pass
                    elif fk == 'actor_mid' and 'actor_mid' in capture:
                        pass
                    elif fk == 'hook_env_mlp3' and 'hook_env_mlp3' in capture:
                        pass
                    elif fk == 'hook_ltl_rnn_h' and 'hook_ltl_rnn_h' in capture:
                        pass
                    else:
                        raise RuntimeError(f"Feature key '{fk}' not captured at step {steps}. Captured: {list(capture.keys())}")

                # push all captured hook keys into per-episode buffer
                for k, arr in capture.items():
                    arr_np = arr[0] if getattr(arr, 'ndim', 0) >= 2 else np.asarray(arr)
                    if k not in ep_hook_buf:
                        ep_hook_buf[k] = []
                    ep_hook_buf[k].append(arr_np)

                # feature_t
                zfeat = capture[fk][0] if capture[fk].ndim >= 2 else np.asarray(capture[fk])
                if zfeat.ndim > 1:
                    zfeat = zfeat.reshape(-1)
                feature_seq.append(zfeat.astype(np.float32))

                # greedy/mode action from dist
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
                action_seq.append(act)

                # env.step
                step_out = env.step(act)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 5:
                    obs, reward, term, trunc, _ = step_out
                    done = bool(term or trunc)
                else:
                    obs, reward, done, _ = step_out
                steps += 1

                # same early-stop logic for two distinct letters when requested
                if tA_local < 0 and letter_id_seq[-1] != -1:
                    tA_local = len(letter_id_seq) - 1
                    first_letter_local = letter_id_seq[-1]

                if args.early_stop_on_B and tA_local >= 0 and (len(letter_id_seq) - 1 - tA_local) >= args.min_post:
                    seen_diff = any((lid != -1 and lid != first_letter_local) for lid in letter_id_seq[tA_local+1:])
                    if seen_diff and tA_local >= args.min_pre:
                        satisfied = True
                        break

                # impossible to satisfy remaining post length (same heuristic)
                remaining = args.steps_per_ep - steps
                if args.min_post > 0 and tA_local < 0 and remaining < (args.min_post + 1):
                    break
                if args.min_post > 0 and tA_local >= 0:
                    needed = args.min_post - max(0, len(letter_id_seq) - 1 - tA_local)
                    if remaining < needed:
                        break

            # Decide accept/reject using same rule when it's relevant (reach2-type)
            ok = True
            if ("reach2" in mode):
                ok = satisfied or episode_satisfies_minimums(letter_id_seq, args.min_pre, args.min_post)

            if ok:
                # persist stepwise
                feats.extend(feature_seq)
                actions.extend(action_seq)
                positions.extend(pos_seq)
                letters_now.extend(letter_id_seq)
                ep_len = len(feature_seq)
                ep_ids.extend([len(ep_ids) // 1_000_000 + 1] * ep_len)  # simple ep id increment pattern
                base_ids.extend(baseid_seq)

                # episode-level labels broadcast to steps
                goal_mode.extend([mode] * ep_len)
                goal_reach.extend([reach_id] * ep_len)
                goal_reach2.extend([reach2_id] * ep_len)
                goal_avoid.extend([avoid_id] * ep_len)
                goal_formula.extend([formula] * ep_len)

                # commit per-episode hooks into global buffers only upon acceptance
                for k, seq in ep_hook_buf.items():
                    if k not in hook_buffers:
                        hook_buffers[k] = []
                    hook_buffers[k].extend(seq)

                accepted += 1
                global_step_base += ep_len
                if pbar is not None:
                    pbar.update(1)

    # Stack everything
    X = np.stack(feats, axis=0) if len(feats) else np.zeros((0, 1), dtype=np.float32)
    A = np.asarray(actions, dtype=np.int64)
    E = np.asarray(ep_ids, dtype=np.int64)
    P = np.stack(positions, axis=0).astype(np.int64) if len(positions) else np.zeros((0, 2), dtype=np.int64)
    Lnow = np.asarray(letters_now, dtype=np.int64)
    Gmode = np.asarray(goal_mode, dtype=object)
    Greach = np.asarray(goal_reach, dtype=np.int64)
    Greach2 = np.asarray(goal_reach2, dtype=np.int64)
    Gavoid = np.asarray(goal_avoid, dtype=np.int64)
    Gformula = np.asarray(goal_formula, dtype=object)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(
        feature_t=X, action=A, episode=E, agent_pos=P, base_id=np.asarray(base_ids, dtype=np.int64),
        letter_id=Lnow,
        goal_mode=Gmode, goal_reach=Greach, goal_reach2=Greach2, goal_avoid=Gavoid, goal_formula=Gformula,
        letters_vocab=np.asarray(letters_vocab, dtype=object),
        modes=np.asarray(modes, dtype=object),
    )
    # add captured hooks (concatenate stepwise list -> [T, D])
    for k, seq in hook_buffers.items():
        if len(seq) == 0:
            continue
        arr = np.asarray(seq)
        if arr.ndim == 1:
            arr = arr[:, None]
        save_dict[k] = arr

    np.savez_compressed(outp, **save_dict)
    print(f"[saved] {outp}")

    # cleanup
    for h in hooks:
        try: h.remove()
        except Exception: pass
    env.close()
    if pbar is not None:
        try: pbar.close()
        except Exception: pass


if __name__ == "__main__":
    main()
