#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
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
from visualize.zones import draw_trajectories

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 700
STEERING_STRENGTHS = [0.0, 0.5, 1.0, 2.0, 5.0]  # 0.0 = no steering
STEER_LAYER = 'ltl_rnn'  # Options: 'ltl_rnn', 'policy_mlp_0'

class SubgoalSteerer:
    def __init__(self, model, probe_weights, steering_strength=1.0, layer='ltl_rnn'):
        self.model = model
        self.probe_weights = probe_weights
        self.steering_strength = steering_strength
        self.layer = layer
        self.original_hidden = None
        self.steering_direction = torch.tensor(probe_weights, dtype=torch.float32)

    def hook_fn(self, module, input, output):
        # For ltl_rnn, output is (packed, h_n)
        # For policy_mlp_0, output is the activation tensor
        if self.steering_strength == 0.0:
            return output  # No-op: do not modify or re-wrap
        print(f"[DEBUG] Steering layer: {self.layer}, output type: {type(output)}, output shape: {getattr(output, 'shape', None)}")
        if self.layer == 'ltl_rnn':
            h_n = output[1]
            self.original_hidden = h_n.clone()
            steering_adjustment = self.steering_direction.unsqueeze(0).unsqueeze(0) * self.steering_strength
            h_n_modified = h_n + steering_adjustment
            return (output[0], h_n_modified, output[2]) if len(output) > 2 else (output[0], h_n_modified)
        else:
            # Policy MLP: output is activation tensor
            self.original_hidden = output.clone()
            steering_adjustment = self.steering_direction * self.steering_strength
            return output + steering_adjustment

    def get_steering_stats(self):
        """Get statistics about the steering intervention"""
        if self.original_hidden is not None:
            adjustment = self.steering_direction * self.steering_strength
            adjustment_norm = torch.norm(adjustment).item()
            original_norm = torch.norm(self.original_hidden).item()
            relative_change = adjustment_norm / original_norm if original_norm > 0 else 0
            return {
                'adjustment_norm': adjustment_norm,
                'original_norm': original_norm,
                'relative_change': relative_change
            }
        return None

def get_layer_and_hook(model, layer_name, hook_fn):
    if layer_name == 'ltl_rnn':
        if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
            handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
            return handle
    elif layer_name == 'policy_mlp_0':
        # For ContinuousActor
        if hasattr(model, 'actor') and hasattr(model.actor, 'enc'):
            first_layer = model.actor.enc[0]
            handle = first_layer.register_forward_hook(hook_fn)
            return handle
    elif layer_name == 'env_net':
        # For env_net layer
        if hasattr(model, 'env_net'):
            handle = model.env_net.register_forward_hook(hook_fn)
            return handle
    elif layer_name.startswith('env_net_mlp_'):
        # For env_net MLP layers
        layer_idx = int(layer_name.split('_')[-1])
        if hasattr(model.env_net, 'mlp') and len(model.env_net.mlp) > layer_idx:
            handle = model.env_net.mlp[layer_idx].register_forward_hook(hook_fn)
            return handle
    
    # If we get here, the layer wasn't found
    return None

def train_probe_for_steering(model, env, sampler_fn):
    """Train a probe to get weights for steering"""
    print("Training probe for steering...")
    
    # Hook into selected network layer
    feats = []
    def hook_fn(mod, inp, out):
        print(f"[DEBUG] Probe training hook, layer: {STEER_LAYER}, output type: {type(out)}, output shape: {getattr(out, 'shape', None)}")
        if STEER_LAYER == 'ltl_rnn':
            h_n = out[1]
            arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy().flatten()
        else:
            arr = out.detach().cpu().numpy().flatten()
        feats.append(arr)
    
    handle = get_layer_and_hook(model, STEER_LAYER, hook_fn)
    
    # Create agent and collect data
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    ret = env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    labels = []
    for step in trange(500, desc="Collecting probe data"):
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
    
    print(f"Collected {len(y)} valid samples for probe training")
    
    # Train probe
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"Probe accuracy: {acc:.3f}")
    
    return clf.coef_[0], clf.intercept_[0]

def run_steered_rollout(model, env, sampler_fn, probe_weights, steering_strength, world_idx=0):
    print(f"Running steered rollout with strength {steering_strength}...")
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    steerer = SubgoalSteerer(model, probe_weights, steering_strength, layer=STEER_LAYER)
    handle = None
    if steering_strength > 0.0:
        handle = get_layer_and_hook(model, STEER_LAYER, steerer.hook_fn)
    agent_positions = []
    episode_events = {'zone_entries': {}, 'goal_steps': {}}
    for step in range(MAX_STEPS):
        # --- DEBUG PRINTS ---
        # Print current world file if available
        if hasattr(env, 'world_file'):
            print(f"Step {step}: World file: {env.world_file}")
        # Agent position
        agent_pos = getattr(env, 'agent', None)
        if agent_pos is not None and hasattr(agent_pos, 'pos'):
            pos = np.array(agent_pos.pos)
        elif hasattr(env, 'agent_pos'):
            pos = np.array(env.agent_pos)
        else:
            pos = None
        print(f"Step {step}: Agent position: {pos}")
        # Zone centers and radii
        zone_centers = getattr(env, 'zone_positions', None)
        zone_radii = getattr(env, 'zone_radii', None)
        if zone_centers is not None and zone_radii is not None:
            for i, (zc, zr) in enumerate(zip(zone_centers, zone_radii)):
                if pos is not None and not isinstance(pos, float) and isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    dist = np.linalg.norm(np.array(pos[:2]) - np.array(zc[:2]))
                else:
                    dist = None
                print(f"Step {step}: Zone {i} center: {zc}, radius: {zr}, dist to agent: {dist}")
        # Propositions
        if isinstance(obs, dict):
            print(f"Step {step}: obs['propositions']: {obs.get('propositions', None)}")
        else:
            print(f"Step {step}: obs: {obs}")
        if isinstance(info, dict):
            print(f"Step {step}: info['propositions']: {info.get('propositions', None)}")
            print(f"Step {step}: info (raw): {info}")
        else:
            print(f"Step {step}: info: {info}")
        # Current subgoal
        seq = getattr(agent, "sequence", None)
        current_goal = None
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    current_goal = next(iter(true_props))
        print(f"Step {step}: Current subgoal: {current_goal}")
        # --- END DEBUG PRINTS ---
        # Robustly handle pos for agent_positions
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            agent_positions.append([float(pos[0]), float(pos[1])])
        elif isinstance(pos, np.ndarray) and pos.ndim >= 1 and pos.shape[0] >= 2:
            agent_positions.append([float(pos[0]), float(pos[1])])
        else:
            agent_positions.append([0.0, 0.0])
        action = agent.get_action(obs, info, deterministic=True)
        if hasattr(action, 'cpu') and not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        if hasattr(action, 'flatten'):
            action = action.flatten()
        action = np.asarray(action)
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        if done:
            break
        # Only check zone entries if zone_centers and zone_radii are valid
        if pos is not None and not isinstance(pos, float) and isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
            if zone_centers is not None and zone_radii is not None and hasattr(zone_centers, '__iter__') and hasattr(zone_radii, '__iter__'):
                for i, (zc, zr) in enumerate(zip(zone_centers, zone_radii)):
                    if isinstance(zc, (list, tuple, np.ndarray)) and len(zc) >= 2:
                        pos2 = np.array(pos[:2])
                        zc2 = np.array(zc[:2])
                        dist = np.linalg.norm(pos2 - zc2)
                        if dist <= zr:
                            if f"zone_{i}" not in episode_events['zone_entries']:
                                episode_events['zone_entries'][f"zone_{i}"] = 0
                            episode_events['zone_entries'][f"zone_{i}"] += 1
        if current_goal:
            if current_goal not in episode_events['goal_steps']:
                episode_events['goal_steps'][current_goal] = step
    if handle:
        handle.remove()
    return agent_positions, episode_events

def analyze_steering_effect(results_dict):
    """Analyze the effect of steering on behavior"""
    print("\n=== Steering Effect Analysis ===")
    
    for strength, results in results_dict.items():
        print(f"\nSteering strength: {strength}")
        
        # Goal prediction accuracy
        valid_mask = results['predicted_goals'] != -1
        valid_mask &= results['true_goals'] != -1
        
        if np.sum(valid_mask) > 0:
            pred_acc = np.mean(results['predicted_goals'][valid_mask] == results['true_goals'][valid_mask])
            print(f"  Goal prediction accuracy: {pred_acc:.3f}")
        
        # Goal distribution
        valid_preds = results['predicted_goals'][results['predicted_goals'] != -1]
        if len(valid_preds) > 0:
            blue_ratio = np.mean(valid_preds)
            print(f"  Predicted blue ratio: {blue_ratio:.3f}")
        
        # Steering intervention stats
        if results['steering_stats']:
            avg_relative_change = np.mean([s['relative_change'] for s in results['steering_stats']])
            print(f"  Average relative change: {avg_relative_change:.3f}")
        
        # Total reward
        total_reward = np.sum(results['rewards'])
        print(f"  Total reward: {total_reward:.2f}")
        
        # Trajectory length
        print(f"  Trajectory length: {results['steps']}")

def visualize_steering_results(results_dict, save_path=None):
    """Create visualizations of steering results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Goal prediction over time for different strengths
    ax1 = axes[0, 0]
    for strength in sorted(results_dict.keys()):
        results = results_dict[strength]
        valid_mask = results['predicted_goals'] != -1
        if np.sum(valid_mask) > 0:
            ax1.plot(results['predicted_goals'][valid_mask], 
                    label=f'Strength {strength}', alpha=0.7)
    ax1.set_title('Predicted Goals Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Predicted Goal (1=Blue, 0=Green)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Blue ratio vs steering strength
    ax2 = axes[0, 1]
    strengths = []
    blue_ratios = []
    for strength in sorted(results_dict.keys()):
        results = results_dict[strength]
        valid_preds = results['predicted_goals'][results['predicted_goals'] != -1]
        if len(valid_preds) > 0:
            strengths.append(strength)
            blue_ratios.append(np.mean(valid_preds))
    
    ax2.plot(strengths, blue_ratios, 'bo-', linewidth=2, markersize=8)
    ax2.set_title('Predicted Blue Ratio vs Steering Strength')
    ax2.set_xlabel('Steering Strength')
    ax2.set_ylabel('Predicted Blue Ratio')
    ax2.grid(True, alpha=0.3)
    
    # 3. Trajectory comparison
    ax3 = axes[1, 0]
    colors = ['black', 'red', 'orange', 'green', 'blue']
    for i, strength in enumerate(sorted(results_dict.keys())):
        results = results_dict[strength]
        positions = results['agent_positions']
        ax3.plot(positions[:, 0], positions[:, 1], 
                color=colors[i], label=f'Strength {strength}', alpha=0.7, linewidth=2)
    ax3.set_title('Agent Trajectories')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Total reward vs steering strength
    ax4 = axes[1, 1]
    strengths = []
    total_rewards = []
    for strength in sorted(results_dict.keys()):
        results = results_dict[strength]
        strengths.append(strength)
        total_rewards.append(np.sum(results['rewards']))
    
    ax4.plot(strengths, total_rewards, 'go-', linewidth=2, markersize=8)
    ax4.set_title('Total Reward vs Steering Strength')
    ax4.set_xlabel('Steering Strength')
    ax4.set_ylabel('Total Reward')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved steering analysis to {save_path}")
    plt.show()

def test_multiple_layers(model, env, sampler_fn):
    """Test probe accuracy on multiple layers to find the best one for steering"""
    print("\n=== Testing Multiple Layers for Probe Accuracy ===")
    
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
    
    results = []
    
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
        
        handle = None
        try:
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
        except Exception as e:
            print(f"  Could not hook into {layer_name}: {e}")
            continue
        
        if handle is None:
            print(f"  Could not find layer {layer_name}")
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
        
        # Train probe
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X, y)
        acc = clf.score(X, y)
        
        print(f"  Probe accuracy: {acc:.3f} ({len(y)} samples)")
        
        results.append({
            'layer': layer_name,
            'accuracy': acc,
            'num_samples': len(y),
            'weights': clf.coef_[0],
            'intercept': clf.intercept_[0]
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("LAYER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Accuracy':<10} {'Samples':<8}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['layer']:<15} {result['accuracy']:<10.3f} {result['num_samples']:<8}")
    
    # Find best layer
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest layer for steering: {best_result['layer']} (accuracy: {best_result['accuracy']:.3f})")
        return best_result
    else:
        print("\nNo valid layers found!")
        return None

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("=== Subgoal Steering Trajectory Visualization ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
    print()
    sampler_fn = FixedSampler.partial(FORMULA)
    
    # --- Train probe on a separate model/env instance ---
    probe_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    probe_build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    probe_store = ModelStore(ENV, EXP, SEED)
    probe_store.load_vocab()
    probe_status = probe_store.load_training_status(map_location="cpu")
    probe_cfg = model_configs[ENV]
    probe_model = build_model(probe_build_env, probe_status, probe_cfg).eval()
    
    # Test multiple layers first
    best_layer_result = test_multiple_layers(probe_model, probe_env, sampler_fn)
    
    if best_layer_result:
        # Update the global STEER_LAYER with the best one
        global STEER_LAYER
        STEER_LAYER = best_layer_result['layer']
        print(f"\nUsing best layer for steering: {STEER_LAYER}")
        
        # Train probe on the best layer
        probe_weights, probe_intercept = train_probe_for_steering(probe_model, probe_env, sampler_fn)
    else:
        print("Falling back to default layer and probe training...")
        probe_weights, probe_intercept = train_probe_for_steering(probe_model, probe_env, sampler_fn)
    
    probe_env.close()
    probe_build_env.close()
    
    # --- Run experiments with fresh model/env instances ---
    world_files = [
        'eval_datasets/PointLtl2-v0/worlds/world_info_30.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_31.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_32.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_33.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_44.pkl',
    ]
    # --- Run unsteered rollouts ---
    print("\n=== Unsteered rollouts (steering_strength=0.0) ===")
    trajectories_unsteered = []
    zone_poss_unsteered = []
    for world_file in world_files:
        # Reload model and env for each rollout
        build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        store = ModelStore(ENV, EXP, SEED)
        store.load_vocab()
        status = store.load_training_status(map_location="cpu")
        cfg = model_configs[ENV]
        model = build_model(build_env, status, cfg).eval()
        build_env.close()
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        if hasattr(env, 'load_world_info'):
            env.load_world_info(world_file)
            print(f"\nWorld file: {world_file}")
        agent_positions, episode_events = run_steered_rollout(model, env, sampler_fn, probe_weights, steering_strength=0.0)
        trajectories_unsteered.append(agent_positions)
        zone_poss_unsteered.append(getattr(env, 'zone_positions', []))
        env.close()
        # Print summary for unsteered
        print(f"Summary (unsteered) for World {world_file}:")
        for subgoal, step in episode_events['goal_steps'].items():
            print(f"  Subgoal '{subgoal}' satisfied at step {step}")
    # --- Minimal fix for zone keys before plotting ---
    def fix_zone_keys(zone_dict):
        if not isinstance(zone_dict, dict):
            return zone_dict
        fixed = {}
        for k, v in zone_dict.items():
            if isinstance(k, (list, tuple)):
                key_str = '_'.join(str(x) for x in k)
            else:
                key_str = str(k)
            fixed[key_str] = v
        return fixed
    zone_poss_unsteered = [fix_zone_keys(z) for z in zone_poss_unsteered]
    num_cols = 3
    num_rows = 2
    plt.figure(figsize=(16, 8))
    draw_trajectories(zone_poss_unsteered, trajectories_unsteered, num_cols, num_rows)
    plt.title('Unsteered Trajectories (Selected Worlds)')
    plt.savefig('trajectories_selected_worlds_unsteered.png')
    plt.close()
    # --- Run steered rollouts ---
    print("\n=== Steered rollouts (steering_strength=1.0) ===")
    trajectories_steered = []
    zone_poss_steered = []
    for world_file in world_files:
        # Reload model and env for each rollout
        build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        store = ModelStore(ENV, EXP, SEED)
        store.load_vocab()
        status = store.load_training_status(map_location="cpu")
        cfg = model_configs[ENV]
        model = build_model(build_env, status, cfg).eval()
        build_env.close()
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        if hasattr(env, 'load_world_info'):
            env.load_world_info(world_file)
            print(f"\nWorld file: {world_file}")
        agent_positions, episode_events = run_steered_rollout(model, env, sampler_fn, probe_weights, steering_strength=1.0)
        trajectories_steered.append(agent_positions)
        zone_poss_steered.append(getattr(env, 'zone_positions', []))
        env.close()
        # Print summary for steered
        print(f"Summary (steered) for World {world_file}:")
        for subgoal, step in episode_events['goal_steps'].items():
            print(f"  Subgoal '{subgoal}' satisfied at step {step}")
    zone_poss_steered = [fix_zone_keys(z) for z in zone_poss_steered]
    num_cols = 3
    num_rows = 2
    plt.figure(figsize=(16, 8))
    draw_trajectories(zone_poss_steered, trajectories_steered, num_cols, num_rows)
    plt.title('Steered Trajectories (Selected Worlds)')
    plt.savefig('trajectories_selected_worlds_steered.png')
    plt.close()

if __name__ == '__main__':
    main() 