#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

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
MAX_STEPS = 500

class NonlinearProbe(nn.Module):
    """Simple neural network probe for non-linear steering"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self):
        """Extract weights from the first layer for steering"""
        return self.network[0].weight.data.mean(dim=0).cpu().numpy()

def get_layer_and_hook(model, layer_name, hook_fn):
    if layer_name == 'ltl_rnn':
        if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
            handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
            return handle
    elif layer_name == 'policy_mlp_0':
        if hasattr(model, 'actor') and hasattr(model.actor, 'enc'):
            first_layer = model.actor.enc[0]
            handle = first_layer.register_forward_hook(hook_fn)
            return handle
    elif layer_name == 'env_net':
        if hasattr(model, 'env_net'):
            handle = model.env_net.register_forward_hook(hook_fn)
            return handle
    elif layer_name.startswith('env_net_mlp_'):
        layer_idx = int(layer_name.split('_')[-1])
        if hasattr(model.env_net, 'mlp') and len(model.env_net.mlp) > layer_idx:
            handle = model.env_net.mlp[layer_idx].register_forward_hook(hook_fn)
            return handle
    return None

def analyze_layer_effectiveness(model, env, sampler_fn):
    """Analyze which layers are most effective for steering"""
    print("\n=== Layer Effectiveness Analysis ===")
    
    layers_to_test = [
        'ltl_rnn',
        'env_net',
        'env_net_mlp_0',
        'env_net_mlp_1',
        'env_net_mlp_2',
        'env_net_mlp_3'
    ]
    
    results = {}
    
    for layer_name in layers_to_test:
        print(f"\n--- Testing layer: {layer_name} ---")
        
        # Hook into the layer
        feats = []
        def hook_fn(mod, inp, out):
            if layer_name == 'ltl_rnn':
                h_n = out[1]
                arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy().flatten()
            else:
                arr = out.detach().cpu().numpy().flatten()
            feats.append(arr)
        
        handle = get_layer_and_hook(model, layer_name, hook_fn)
        if handle is None:
            print(f"  Could not hook into {layer_name}")
            continue
        
        # Create agent and collect data
        props = set(env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        
        ret = env.reset(seed=SEED)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        labels = []
        for step in trange(300, desc=f"Collecting data for {layer_name}"):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Get current goal
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
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            if done:
                break
        
        if handle:
            handle.remove()
        
        # Process data
        X = np.array(feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]
        
        if len(y) == 0:
            print(f"  No valid labels for {layer_name}")
            continue
        
        if len(np.unique(y)) <= 1:
            print(f"  Only one class for {layer_name}, skipping")
            continue
        
        # Train both linear and non-linear probes
        # Linear probe
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        linear_acc = clf.score(X, y)
        linear_weight_norm = np.linalg.norm(clf.coef_[0])
        
        # Non-linear probe
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        probe = NonlinearProbe(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = Adam(probe.parameters(), lr=0.001)
        
        # Training loop
        probe.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = probe(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate accuracy
        probe.eval()
        with torch.no_grad():
            predictions = probe(X_tensor).squeeze()
            predicted_labels = (predictions > 0.5).float()
            nonlinear_acc = (predicted_labels == y_tensor).float().mean().item()
        
        nonlinear_weight_norm = np.linalg.norm(probe.get_weights())
        
        print(f"  Linear probe accuracy: {linear_acc:.3f}, weight norm: {linear_weight_norm:.3f}")
        print(f"  Non-linear probe accuracy: {nonlinear_acc:.3f}, weight norm: {nonlinear_weight_norm:.3f}")
        print(f"  Feature norm: {np.linalg.norm(X, axis=1).mean():.3f}")
        
        results[layer_name] = {
            'linear_acc': linear_acc,
            'nonlinear_acc': nonlinear_acc,
            'linear_weight_norm': linear_weight_norm,
            'nonlinear_weight_norm': nonlinear_weight_norm,
            'feature_norm': np.linalg.norm(X, axis=1).mean(),
            'num_samples': len(y)
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print("LAYER EFFECTIVENESS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Layer':<15} {'Linear Acc':<12} {'Nonlinear Acc':<15} {'Lin W Norm':<12} {'Nonlin W Norm':<15} {'Feature Norm':<12}")
    print("-" * 80)
    
    for layer_name, result in sorted(results.items(), key=lambda x: x[1]['nonlinear_acc'], reverse=True):
        print(f"{layer_name:<15} {result['linear_acc']:<12.3f} {result['nonlinear_acc']:<15.3f} "
              f"{result['linear_weight_norm']:<12.3f} {result['nonlinear_weight_norm']:<15.3f} "
              f"{result['feature_norm']:<12.3f}")
    
    return results

def test_steering_strengths(model, env, sampler_fn, layer_name, probe_weights):
    """Test different steering strengths to find effective range"""
    print(f"\n=== Steering Strength Analysis for {layer_name} ===")
    
    strengths_to_test = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    results = {}
    
    for strength in strengths_to_test:
        print(f"\n--- Testing strength: {strength} ---")
        
        # Hook into the layer
        feats = []
        def hook_fn(mod, inp, out):
            if layer_name == 'ltl_rnn':
                h_n = out[1]
                arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy().flatten()
            else:
                arr = out.detach().cpu().numpy().flatten()
            feats.append(arr)
        
        handle = get_layer_and_hook(model, layer_name, hook_fn)
        if handle is None:
            print(f"  Could not hook into {layer_name}")
            continue
        
        # Create agent and collect data
        props = set(env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        
        ret = env.reset(seed=SEED)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        labels = []
        steering_interventions = 0
        
        for step in trange(200, desc=f"Testing strength {strength}"):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Get current goal
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
                            steering_interventions += 1
                        else:
                            labels.append(-1)
                    else:
                        labels.append(-1)
                else:
                    labels.append(-1)
            else:
                labels.append(-1)
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            if done:
                break
        
        if handle:
            handle.remove()
        
        # Process data
        X = np.array(feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]
        
        if len(y) > 0:
            # Calculate steering effect
            steering_direction = np.array(probe_weights)
            steering_magnitude = strength * np.linalg.norm(steering_direction)
            
            # Simulate steering effect on features
            X_steered = X + steering_direction * strength
            
            # Train probe on original data
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X, y)
            
            # Check how steering affects predictions
            original_probs = clf.predict_proba(X)[:, 1]
            steered_probs = clf.predict_proba(X_steered)[:, 1]
            
            prob_change = np.mean(steered_probs) - np.mean(original_probs)
            prediction_changes = np.sum(clf.predict(X) != clf.predict(X_steered))
            
            print(f"  Steering magnitude: {steering_magnitude:.3f}")
            print(f"  Probability change: {prob_change:.3f}")
            print(f"  Prediction changes: {prediction_changes}/{len(X)} ({prediction_changes/len(X)*100:.1f}%)")
            print(f"  Steering interventions: {steering_interventions}")
            
            results[strength] = {
                'steering_magnitude': steering_magnitude,
                'prob_change': prob_change,
                'prediction_changes': prediction_changes,
                'prediction_change_rate': prediction_changes/len(X) if len(X) > 0 else 0,
                'steering_interventions': steering_interventions
            }
        else:
            print(f"  No valid data for strength {strength}")
    
    return results

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Steering Debug Analysis ===")
    
    # Load model and environment
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Analyze layer effectiveness
    layer_results = analyze_layer_effectiveness(model, env, sampler_fn)
    
    # Find best layer
    if layer_results:
        best_layer = max(layer_results.keys(), key=lambda x: layer_results[x]['nonlinear_acc'])
        print(f"\nBest layer for steering: {best_layer}")
        
        # Train probe on best layer
        print(f"\nTraining probe on {best_layer}...")
        
        # Hook into the layer
        feats = []
        def hook_fn(mod, inp, out):
            if best_layer == 'ltl_rnn':
                h_n = out[1]
                arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy().flatten()
            else:
                arr = out.detach().cpu().numpy().flatten()
            feats.append(arr)
        
        handle = get_layer_and_hook(model, best_layer, hook_fn)
        
        # Create agent and collect data
        props = set(env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        
        ret = env.reset(seed=SEED)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        agent.reset()
        
        labels = []
        for step in trange(300, desc="Collecting probe data"):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Get current goal
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
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
            if done:
                break
        
        if handle:
            handle.remove()
        
        # Process data
        X = np.array(feats)
        y = np.array(labels)
        valid_idxs = (y != -1)
        if len(X) > len(y):
            X = X[:len(y)]
        X, y = X[valid_idxs], y[valid_idxs]
        
        if len(y) > 0:
            # Train non-linear probe
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            probe = NonlinearProbe(input_dim=X.shape[1])
            criterion = nn.BCELoss()
            optimizer = Adam(probe.parameters(), lr=0.001)
            
            # Training loop
            probe.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = probe(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
            
            # Get probe weights
            probe_weights = probe.get_weights()
            
            # Test steering strengths
            strength_results = test_steering_strengths(model, env, sampler_fn, best_layer, probe_weights)
            
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Layer effectiveness
            plt.subplot(2, 3, 1)
            layers = list(layer_results.keys())
            linear_accs = [layer_results[l]['linear_acc'] for l in layers]
            nonlinear_accs = [layer_results[l]['nonlinear_acc'] for l in layers]
            
            x = np.arange(len(layers))
            width = 0.35
            plt.bar(x - width/2, linear_accs, width, label='Linear Probe')
            plt.bar(x + width/2, nonlinear_accs, width, label='Non-linear Probe')
            plt.xlabel('Layer')
            plt.ylabel('Accuracy')
            plt.title('Layer Effectiveness')
            plt.xticks(x, layers, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Weight norms
            plt.subplot(2, 3, 2)
            linear_norms = [layer_results[l]['linear_weight_norm'] for l in layers]
            nonlinear_norms = [layer_results[l]['nonlinear_weight_norm'] for l in layers]
            
            plt.bar(x - width/2, linear_norms, width, label='Linear Probe')
            plt.bar(x + width/2, nonlinear_norms, width, label='Non-linear Probe')
            plt.xlabel('Layer')
            plt.ylabel('Weight Norm')
            plt.title('Probe Weight Norms')
            plt.xticks(x, layers, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Feature norms
            plt.subplot(2, 3, 3)
            feature_norms = [layer_results[l]['feature_norm'] for l in layers]
            plt.bar(x, feature_norms)
            plt.xlabel('Layer')
            plt.ylabel('Feature Norm')
            plt.title('Average Feature Norms')
            plt.xticks(x, layers, rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Plot 4: Steering strength effects
            if strength_results:
                plt.subplot(2, 3, 4)
                strengths = list(strength_results.keys())
                prob_changes = [strength_results[s]['prob_change'] for s in strengths]
                plt.plot(strengths, prob_changes, 'bo-')
                plt.xlabel('Steering Strength')
                plt.ylabel('Probability Change')
                plt.title('Steering Effect on Predictions')
                plt.xscale('log')
                plt.grid(True, alpha=0.3)
                
                # Plot 5: Prediction change rate
                plt.subplot(2, 3, 5)
                change_rates = [strength_results[s]['prediction_change_rate'] for s in strengths]
                plt.plot(strengths, change_rates, 'ro-')
                plt.xlabel('Steering Strength')
                plt.ylabel('Prediction Change Rate')
                plt.title('Steering Effect on Predictions')
                plt.xscale('log')
                plt.grid(True, alpha=0.3)
                
                # Plot 6: Steering magnitude vs strength
                plt.subplot(2, 3, 6)
                steering_magnitudes = [strength_results[s]['steering_magnitude'] for s in strengths]
                plt.plot(strengths, steering_magnitudes, 'go-')
                plt.xlabel('Steering Strength')
                plt.ylabel('Steering Magnitude')
                plt.title('Steering Magnitude vs Strength')
                plt.xscale('log')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('steering_debug_analysis.png', dpi=150, bbox_inches='tight')
            print("\nSaved steering debug analysis to steering_debug_analysis.png")
            plt.show()
    
    env.close()

if __name__ == '__main__':
    main() 