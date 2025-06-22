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

    print(f"Observation shape: {build_env.observation_space.shape}")
    print(f"Num LTLNet params: {sum(p.numel() for p in model.ltl_net.parameters())}")
    
    # 3) hook into different parts of the model to grab hidden states
    ltl_feats = []
    policy_feats = []
    env_feats = []
    
    def ltl_hook_fn(mod, inp, out):
        h_n = out[1]  # Final hidden state
        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        ltl_feats.append(arr)
    
    def policy_hook_fn(mod, inp, out):
        # For policy network, we might need to hook into different layers
        # Let's try to get the final layer activations
        if hasattr(out, 'detach'):
            arr = out.detach().squeeze().cpu().numpy()
        else:
            arr = out.squeeze().cpu().numpy()
        policy_feats.append(arr)
    
    def env_hook_fn(mod, inp, out):
        # For environment network
        if hasattr(out, 'detach'):
            arr = out.detach().squeeze().cpu().numpy()
        else:
            arr = out.squeeze().cpu().numpy()
        env_feats.append(arr)
    
    # Register hooks on different parts of the model
    handles = []
    
    # LTL network hook
    if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
        handle = model.ltl_net.rnn.register_forward_hook(ltl_hook_fn)
        handles.append(handle)
        print("Registered LTL network hook")
    
    # Policy network hooks - try different layers
    if hasattr(model, 'policy'):
        # Try hooking into the policy network's final layer
        if hasattr(model.policy, 'mlp'):
            handle = model.policy.mlp.register_forward_hook(policy_hook_fn)
            handles.append(handle)
            print("Registered policy MLP hook")
        elif hasattr(model.policy, 'actor'):
            handle = model.policy.actor.register_forward_hook(policy_hook_fn)
            handles.append(handle)
            print("Registered policy actor hook")
    
    # Environment network hooks
    if hasattr(model, 'env_net'):
        if hasattr(model.env_net, 'mlp'):
            handle = model.env_net.mlp.register_forward_hook(env_hook_fn)
            handles.append(handle)
            print("Registered environment MLP hook")
    
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
        
        if done:
            break
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    rollout_env.close()

    # 5) process and print results for each network
    print(f"\n=== LTL Network Results ===")
    if len(ltl_feats) > 0:
        X = np.array(ltl_feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]

        print(f"Collected {len(y)} valid next‐prop labels")
        if len(y) > 0:
            print("Label distribution:")
            unique_labels, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label in I2C:
                    print(f"  {I2C[label]}: {count}")

        if len(np.unique(y)) > 1:
            clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            clf.fit(X, y)
            acc = clf.score(X, y)
            print(f"LTL Next-prop probe accuracy: {acc:.2%}")
            
            print("Learned probe weights:")
            for i, class_idx in enumerate(clf.classes_):
                prop = I2C[class_idx]
                if i < len(clf.estimators_):
                    w = clf.estimators_[i].coef_[0]
                    print(f"  {prop:<10} | w_norm={np.linalg.norm(w):.3f}")
        else:
            print("Only one class found for LTL network")
    else:
        print("No LTL features collected")

    print(f"\n=== Policy Network Results ===")
    if len(policy_feats) > 0:
        X = np.array(policy_feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]

        print(f"Collected {len(y)} valid next‐prop labels")
        if len(y) > 0:
            print("Label distribution:")
            unique_labels, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label in I2C:
                    print(f"  {I2C[label]}: {count}")

        if len(np.unique(y)) > 1:
            clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            clf.fit(X, y)
            acc = clf.score(X, y)
            print(f"Policy Next-prop probe accuracy: {acc:.2%}")
            
            print("Learned probe weights:")
            for i, class_idx in enumerate(clf.classes_):
                prop = I2C[class_idx]
                if i < len(clf.estimators_):
                    w = clf.estimators_[i].coef_[0]
                    print(f"  {prop:<10} | w_norm={np.linalg.norm(w):.3f}")
        else:
            print("Only one class found for policy network")
    else:
        print("No policy features collected")

    print(f"\n=== Environment Network Results ===")
    if len(env_feats) > 0:
        X = np.array(env_feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]

        print(f"Collected {len(y)} valid next‐prop labels")
        if len(y) > 0:
            print("Label distribution:")
            unique_labels, counts = np.unique(y, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label in I2C:
                    print(f"  {I2C[label]}: {count}")

        if len(np.unique(y)) > 1:
            clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            clf.fit(X, y)
            acc = clf.score(X, y)
            print(f"Environment Next-prop probe accuracy: {acc:.2%}")
            
            print("Learned probe weights:")
            for i, class_idx in enumerate(clf.classes_):
                prop = I2C[class_idx]
                if i < len(clf.estimators_):
                    w = clf.estimators_[i].coef_[0]
                    print(f"  {prop:<10} | w_norm={np.linalg.norm(w):.3f}")
        else:
            print("Only one class found for environment network")
    else:
        print("No environment features collected")

    # 6) Summary
    print(f"\n=== Summary ===")
    print("This experiment probes different parts of the model to see where goal information is encoded:")
    print("- LTL Network: Processes LTL formulas and maintains goal state")
    print("- Policy Network: Makes action decisions based on current state and goals")
    print("- Environment Network: Processes environment observations")
    print("\nThe network with the highest probe accuracy likely contains the most goal-relevant information.")

if __name__ == '__main__':
    main() 