#!/usr/bin/env python3
import random
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from envs.flatworld    import FlatWorld
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 1
MAX_STEPS = 1000

# pick whatever formula you like here:
FORMULA = "GF blue & GF green"

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) figure out which atomic props exist
    props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
    C2I   = {p: i for i, p in enumerate(props)}
    I2C   = {i: p for i, p in enumerate(props)}
    print(f"Propositions: {props}")
    
    sampler_fn = FixedSampler.partial(FORMULA)

    # 2) load the pretrained model & wrap with LTL-searching agent
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store     = ModelStore(ENV, EXP, 0) # always load from seed 0
    store.load_vocab()
    status    = store.load_training_status(map_location="cpu")
    cfg       = model_configs[ENV]
    model     = build_model(build_env, status, cfg).eval()
    search    = ExhaustiveSearch(model, set(props), num_loops=2)
    agent     = Agent(model, search=search, propositions=set(props), verbose=False)
    build_env.close()

    # 3) hook into the LTL RNN to grab its hidden state
    feats = []
    def hook_fn(mod, inp, out):
        h_n = out[1]
        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        feats.append(arr)
    
    handle = None
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handle = model.ltl_net.rnn.register_forward_hook(hook_fn)

    # 4) create a rollout env
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()

    labels = []
    for step in trange(MAX_STEPS, desc="Rolling out"):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    prop = next(iter(true_props))
                    if prop in C2I:
                        labels.append(C2I[prop])
                    else:
                        labels.append(-1)
                else:
                    labels.append(-1)
            else:
                labels.append(-1)
        else:
            labels.append(-1)
            
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        
        if done:
            break
    
    if handle:
        handle.remove()
    rollout_env.close()

    # 5) process and print results
    X = np.array(feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X) > len(y):
        X = X[:len(y)]
    X, y = X[valid_idxs], y[valid_idxs]

    print(f"\nCollected {len(y)} valid next‐prop labels")
    if len(y) > 0:
        print("Label distribution:")
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label in I2C:
                print(f"  {I2C[label]}: {count}")

    if len(np.unique(y)) <= 1:
        print("\nProbe not run: only one class of label was collected.")
        return
    
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"\nNext-prop probe accuracy: {acc:.2%}\n")

    print("Learned probe weights:")
    for i, class_idx in enumerate(clf.classes_):
        prop = I2C[class_idx]
        if i < len(clf.estimators_):
            w = clf.estimators_[i].coef_[0]  # OneVsRestClassifier stores estimators in estimators_
            print(f"  {prop:<10} | w_norm={np.linalg.norm(w):.3f}")
        else:
            print(f"  {prop:<10} | w_norm=N/A (no coefficient)")

if __name__ == '__main__':
    main()
