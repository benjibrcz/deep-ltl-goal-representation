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

def get_layer_features(model, layer_name, rollout_env, agent, max_steps=MAX_STEPS):
    """Extract features from a specific layer during rollout."""
    feats = []
    
    def hook_fn(mod, inp, out):
        if layer_name == 'ltl_rnn':
            # For LTL RNN, output is (packed, h_n)
            h_n = out[1]
            arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        else:
            # For other layers, output is the activation tensor
            arr = out.detach().squeeze().cpu().numpy()
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        feats.append(arr)
    
    # Register hook based on layer name
    handle = None
    if layer_name == 'ltl_rnn':
        if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
            handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
    elif layer_name.startswith('policy_mlp_'):
        layer_idx = int(layer_name.split('_')[-1])
        if hasattr(model.policy, 'mlp') and len(model.policy.mlp) > layer_idx:
            handle = model.policy.mlp[layer_idx].register_forward_hook(hook_fn)
    elif layer_name.startswith('env_net_mlp_'):
        layer_idx = int(layer_name.split('_')[-1])
        if hasattr(model.env_net, 'mlp') and len(model.env_net.mlp) > layer_idx:
            handle = model.env_net.mlp[layer_idx].register_forward_hook(hook_fn)
    elif layer_name == 'env_net':
        if hasattr(model, 'env_net'):
            handle = model.env_net.register_forward_hook(hook_fn)
    
    if handle is None:
        print(f"Warning: Could not find layer {layer_name}")
        return [], []
    
    # Run rollout
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    labels = []
    for step in trange(max_steps, desc=f"Rolling out ({layer_name})"):
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
    
    handle.remove()
    return feats, labels

def evaluate_layer(model, layer_name, rollout_env, agent, C2I, I2C):
    """Evaluate probe accuracy for a specific layer."""
    print(f"\n=== Testing layer: {layer_name} ===")
    
    feats, labels = get_layer_features(model, layer_name, rollout_env, agent)
    
    if not feats:
        print(f"  No features collected for {layer_name}")
        return None
    
    # Process features and labels
    X = np.array(feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X) > len(y):
        X = X[:len(y)]
    X, y = X[valid_idxs], y[valid_idxs]
    
    print(f"  Collected {len(y)} valid next-prop labels")
    if len(y) == 0:
        print(f"  No valid labels for {layer_name}")
        return None
    
    if len(np.unique(y)) <= 1:
        print(f"  Probe not run for {layer_name}: only one class of label was collected.")
        return None
    
    # Train probe
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X, y)
    acc = clf.score(X, y)
    
    print(f"  Next-prop probe accuracy: {acc:.2%}")
    
    # Print weight norms
    weight_norms = []
    for i, class_idx in enumerate(clf.classes_):
        prop = I2C[class_idx]
        if i < len(clf.estimators_):
            w = clf.estimators_[i].coef_[0]
            w_norm = np.linalg.norm(w)
            weight_norms.append(w_norm)
            print(f"    {prop:<10} | w_norm={w_norm:.3f}")
    
    return {
        'layer': layer_name,
        'accuracy': acc,
        'num_samples': len(y),
        'weight_norms': weight_norms,
        'mean_weight_norm': np.mean(weight_norms) if weight_norms else 0.0
    }

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

    # 3) Define layers to test
    layers_to_test = [
        'ltl_rnn',
        'policy_mlp_0',
        'policy_mlp_1', 
        'policy_mlp_2',
        'env_net',
        'env_net_mlp_0',
        'env_net_mlp_1',
        'env_net_mlp_2',
        'env_net_mlp_3'
    ]
    
    # 4) Create rollout environment
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # 5) Test each layer
    results = []
    for layer_name in layers_to_test:
        result = evaluate_layer(model, layer_name, rollout_env, agent, C2I, I2C)
        if result:
            results.append(result)
    
    rollout_env.close()
    
    # 6) Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL LAYERS")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Accuracy':<10} {'Samples':<8} {'Mean W Norm':<12}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['layer']:<15} {result['accuracy']:<10.2%} {result['num_samples']:<8} {result['mean_weight_norm']:<12.3f}")
    
    # Find best layer
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest layer for steering: {best_result['layer']} (accuracy: {best_result['accuracy']:.2%})")

if __name__ == '__main__':
    main() 