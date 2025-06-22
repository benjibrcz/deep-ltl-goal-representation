#!/usr/bin/env python3
import pickle
import torch
import numpy as np

from utils.model_store import ModelStore
from model.model        import build_model
from config             import model_configs
from ltl                import FixedSampler
from envs               import make_env
from sequence.search    import ExhaustiveSearch
from model.agent        import Agent
from evaluation.simulate import simulate  # re-use their entry point

# ── CONFIG ─────────────────────────────────────────────────────────
ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 0
FORMULA   = "GF blue & GF green"
LAYER     = "env_net.mlp.0"
NUM_EP    = 5
DEVICE    = "cpu"
# ─────────────────────────────────────────────────────────────────

def main():
    # 1) build model + layer hook
    sampler = FixedSampler.partial(FORMULA)
    store   = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status  = store.load_training_status(map_location=DEVICE)
    cfg     = model_configs[ENV]
    dummy   = make_env(ENV, sampler, sequence=False, render_mode=None)
    model   = build_model(dummy, status, cfg).to(DEVICE).eval()
    dummy.close()

    # find module
    layer = dict(model.named_modules())[LAYER]
    activations = []

    def hook_fn(module, inp, out):
        # out may be (rnn_out, h_n) or a tensor
        if isinstance(out, tuple):
            x = out[1]
        else:
            x = out
        # flatten to 1d
        activations.append(x.detach().cpu().numpy().ravel())

    handle = layer.register_forward_hook(hook_fn)

    # 2) run the same simulate() call you already use
    #    simulate() will write out trajectories.pkl / zone_positions.pkl
    simulate(
        env=ENV,
        gamma=None,
        exp=EXP,
        seed=SEED,
        num_episodes=NUM_EP,
        formula=FORMULA,
        finite=False,         # *no* --finite
        render=None,
        deterministic=True
    )

    handle.remove()

    # 3) load back the zones & trajectories
    with open("zone_positions.pkl", "rb") as f:
        zone_positions = pickle.load(f)
    with open("trajectories.pkl", "rb") as f:
        trajectories = pickle.load(f)

    # now `activations` is a flat list of length sum(T_i),
    # but you know each episode was length T_i in trajectories[i].
    # so you can split it back into per-episode, per-step chunks:
    idx = 0
    per_episode_acts = []
    for traj in trajectories:
        T = len(traj)
        per_episode_acts.append( np.stack(activations[idx:idx+T]) )
        idx += T

    # dump out for your probing script
    with open("hooks_activations.pkl", "wb") as f:
        pickle.dump(per_episode_acts, f)

    print("Saved hooks_activations.pkl — now feed this into your linear‐probe code.")

if __name__ == "__main__":
    main()
