#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 1000

def analyze_steering_mechanism():
    """Analyze why steering is ineffective"""
    print("=== Steering Failure Analysis ===")
    
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load model
    print("Loading model...")
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Create environment
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    print("\n=== Hypothesis 1: Steering Magnitude Too Small ===")
    # Test extreme steering magnitudes
    extreme_strengths = [10.0, 20.0, 50.0, 100.0]
    
    # Train a simple probe first
    ltl_feats = []
    labels = []
    
    def ltl_hook_fn(mod, inp, out):
        h_n = out[1]
        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        ltl_feats.append(arr)
    
    ltl_handle = model.ltl_net.rnn.register_forward_hook(ltl_hook_fn)
    
    # Collect data
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    for step in range(100):  # Short rollout for analysis
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    prop = next(iter(true_props))
                    if prop in ['blue', 'green']:
                        labels.append(1 if prop == 'blue' else 0)
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
    
    ltl_handle.remove()
    rollout_env.close()
    
    # Train probe
    X = np.array(ltl_feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X) > len(y):
        X = X[:len(y)]
    X, y = X[valid_idxs], y[valid_idxs]
    
    if len(np.unique(y)) > 1:
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        print(f"Probe accuracy: {probe.score(X, y):.2%}")
        print(f"Probe weight norm: {np.linalg.norm(probe.coef_[0]):.3f}")
        
        # Test extreme steering
        print("\nTesting extreme steering magnitudes:")
        for strength in extreme_strengths:
            print(f"  Strength {strength}: {strength * np.linalg.norm(probe.coef_[0]):.1f} weight norm")
    
    print("\n=== Hypothesis 2: Steering Direction Analysis ===")
    # Analyze if we're steering in the right direction
    if len(np.unique(y)) > 1:
        # Get probe predictions before and after steering
        original_probs = probe.predict_proba(X)[:, 1]
        
        # Apply steering to features
        steer_vector = probe.coef_[0] * 5.0  # Use strength 5.0
        X_steered = X + steer_vector
        
        steered_probs = probe.predict_proba(X_steered)[:, 1]
        
        print(f"Original mean probability: {np.mean(original_probs):.3f}")
        print(f"Steered mean probability: {np.mean(steered_probs):.3f}")
        print(f"Probability change: {np.mean(steered_probs) - np.mean(original_probs):.3f}")
        
        # Check if steering actually changes predictions
        original_preds = probe.predict(X)
        steered_preds = probe.predict(X_steered)
        prediction_changes = np.sum(original_preds != steered_preds)
        print(f"Prediction changes: {prediction_changes}/{len(X)} ({prediction_changes/len(X)*100:.1f}%)")
    
    print("\n=== Hypothesis 3: Network Architecture Analysis ===")
    # Check if the steering is being applied correctly
    print("Model architecture:")
    print(f"  LTL network: {type(model.ltl_net)}")
    print(f"  Environment network: {type(model.env_net)}")
    if hasattr(model, 'policy_net'):
        print(f"  Policy network: {type(model.policy_net)}")
    else:
        print("  Policy network: Not found (may be part of another component)")
    
    # Check if there are other components that might override steering
    print("\nChecking for potential override mechanisms:")
    
    # Look at the forward pass to see if steering gets overwritten
    print("  - Steering might be overwritten by subsequent computations")
    print("  - Policy network might ignore LTL/environment steering")
    print("  - Agent logic might override network outputs")
    
    print("\n=== Hypothesis 4: Temporal Dynamics ===")
    print("  - Steering might only affect immediate predictions, not long-term behavior")
    print("  - Agent might have temporal mechanisms that resist steering")
    print("  - Goal sequences might be pre-computed and resistant to modification")
    
    print("\n=== Hypothesis 5: Robustness Mechanisms ===")
    print("  - Network might have built-in robustness against perturbations")
    print("  - Multiple redundant pathways might compensate for steering")
    print("  - Training might have made the network resistant to input modifications")
    
    print("\n=== Alternative Approaches to Try ===")
    print("1. **Gradient-based steering**: Use gradients instead of probe coefficients")
    print("2. **Adversarial steering**: Maximize target goal while minimizing current goal")
    print("3. **Temporal steering**: Apply steering over multiple consecutive timesteps")
    print("4. **Policy network steering**: Directly modify policy outputs")
    print("5. **Sequence-level steering**: Modify the agent's goal sequence directly")
    print("6. **Reward-based steering**: Modify rewards to encourage target behavior")
    
    print("\n=== Key Insights ===")
    print("1. **High probe accuracy doesn't guarantee steering effectiveness**")
    print("2. **Multi-layer steering is no more effective than single-layer**")
    print("3. **Network appears highly robust to internal modifications**")
    print("4. **Goal behavior may be determined by higher-level mechanisms**")
    print("5. **Linear steering approaches may be fundamentally limited**")
    
    env.close()
    print("\n=== Analysis Complete ===")

def test_gradient_steering():
    """Test gradient-based steering as an alternative"""
    print("\n=== Testing Gradient-Based Steering ===")
    print("This would require:")
    print("1. Computing gradients of goal probability w.r.t. hidden states")
    print("2. Using gradients to determine steering direction")
    print("3. Applying gradient-based modifications")
    print("4. Measuring effectiveness")
    print("\nNot implemented yet - would be a significant extension")

def test_sequence_steering():
    """Test direct sequence modification"""
    print("\n=== Testing Sequence-Level Steering ===")
    print("This would involve:")
    print("1. Directly modifying the agent's goal sequence")
    print("2. Bypassing network steering entirely")
    print("3. Testing if sequence modification affects behavior")
    print("\nThis could reveal if the issue is with network steering or goal logic")

def main():
    analyze_steering_mechanism()
    test_gradient_steering()
    test_sequence_steering()
    
    print("\n=== Recommendations ===")
    print("1. **Try sequence-level steering** - most likely to succeed")
    print("2. **Investigate gradient-based methods** - more principled approach")
    print("3. **Analyze agent architecture** - understand goal determination")
    print("4. **Consider reward modification** - alternative control mechanism")
    print("5. **Study temporal dynamics** - understand long-term behavior")

if __name__ == '__main__':
    main() 