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
from visualize.zones import draw_trajectories

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 700
STEERING_STRENGTHS = [0.0, 10.0, 50.0, 100.0, 500.0]  # Much stronger steering
STEER_LAYER = 'env_net_mlp_0'  # Best layer from debug analysis

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

class StrongSubgoalSteerer:
    def __init__(self, model, probe_weights, steering_strength=1.0, layer='env_net_mlp_0'):
        self.model = model
        self.probe_weights = probe_weights
        self.steering_strength = steering_strength
        self.layer = layer
        self.original_hidden = None
        self.steering_direction = torch.tensor(probe_weights, dtype=torch.float32)
        self.intervention_count = 0

    def hook_fn(self, module, input, output):
        if self.steering_strength == 0.0:
            return output
        
        # Apply much stronger steering
        if self.layer == 'ltl_rnn':
            h_n = output[1]
            self.original_hidden = h_n.clone()
            # Scale steering to be comparable to feature magnitude
            steering_adjustment = self.steering_direction.unsqueeze(0).unsqueeze(0) * self.steering_strength
            h_n_modified = h_n + steering_adjustment
            self.intervention_count += 1
            return (output[0], h_n_modified, output[2]) if len(output) > 2 else (output[0], h_n_modified)
        else:
            # For MLP layers, apply stronger steering
            self.original_hidden = output.clone()
            steering_adjustment = self.steering_direction * self.steering_strength
            modified_output = output + steering_adjustment
            self.intervention_count += 1
            return modified_output

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
                'relative_change': relative_change,
                'intervention_count': self.intervention_count
            }
        return None

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

def train_strong_probe_for_steering(model, env, sampler_fn):
    """Train a non-linear probe to get weights for strong steering"""
    print("Training strong probe for steering...")
    
    # Hook into selected network layer
    feats = []
    def hook_fn(mod, inp, out):
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
    
    if len(y) == 0:
        print("No valid samples collected!")
        return np.zeros(X.shape[1]), 0.0
    
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
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate accuracy
    probe.eval()
    with torch.no_grad():
        predictions = probe(X_tensor).squeeze()
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_tensor).float().mean().item()
    
    print(f"Strong probe accuracy: {accuracy:.3f}")
    
    # Get weights for steering and scale them appropriately
    weights = probe.get_weights()
    
    # Scale weights to be more effective (based on debug analysis)
    feature_norm = np.linalg.norm(X, axis=1).mean()
    weight_norm = np.linalg.norm(weights)
    print(f"Feature norm: {feature_norm:.3f}, Weight norm: {weight_norm:.3f}")
    
    # Scale weights to be comparable to feature magnitude
    scaled_weights = weights * (feature_norm / weight_norm) * 0.1  # 10% of feature magnitude
    print(f"Scaled weight norm: {np.linalg.norm(scaled_weights):.3f}")
    
    return scaled_weights, 0.0

def run_strong_steered_rollout(model, env, sampler_fn, probe_weights, steering_strength, world_idx=0):
    print(f"Running strong steered rollout with strength {steering_strength}...")
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    steerer = StrongSubgoalSteerer(model, probe_weights, steering_strength, layer=STEER_LAYER)
    handle = None
    if steering_strength > 0.0:
        handle = get_layer_and_hook(model, STEER_LAYER, steerer.hook_fn)
    agent_positions = []
    episode_events = {'zone_entries': {}, 'goal_steps': {}}
    
    for step in range(MAX_STEPS):
        # Get agent position
        agent_pos = getattr(env, 'agent', None)
        if agent_pos is not None and hasattr(agent_pos, 'pos'):
            pos = np.array(agent_pos.pos)
        elif hasattr(env, 'agent_pos'):
            pos = np.array(env.agent_pos)
        else:
            pos = None
        
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
        
        # Track goal satisfaction
        seq = getattr(agent, "sequence", None)
        current_goal = None
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    current_goal = next(iter(true_props))
        
        if current_goal:
            if current_goal not in episode_events['goal_steps']:
                episode_events['goal_steps'][current_goal] = step
        
        if done:
            break
    
    if handle:
        handle.remove()
    
    # Get steering statistics
    steering_stats = steerer.get_steering_stats()
    if steering_stats:
        print(f"  Steering interventions: {steering_stats['intervention_count']}")
        print(f"  Relative change: {steering_stats['relative_change']:.3f}")
    
    return agent_positions, episode_events

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("=== Strong Subgoal Steering Trajectory Visualization ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
    print(f"Steering layer: {STEER_LAYER}")
    print(f"Steering strengths: {STEERING_STRENGTHS}")
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
    
    # Train strong probe
    probe_weights, probe_intercept = train_strong_probe_for_steering(probe_model, probe_env, sampler_fn)
    
    probe_env.close()
    probe_build_env.close()
    
    # --- Run experiments with different steering strengths ---
    world_files = [
        'eval_datasets/PointLtl2-v0/worlds/world_info_30.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_31.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_32.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_33.pkl',
        'eval_datasets/PointLtl2-v0/worlds/world_info_44.pkl',
    ]
    
    all_results = {}
    
    for steering_strength in STEERING_STRENGTHS:
        print(f"\n=== Testing steering strength: {steering_strength} ===")
        
        trajectories = []
        zone_poss = []
        
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
            
            agent_positions, episode_events = run_strong_steered_rollout(
                model, env, sampler_fn, probe_weights, steering_strength
            )
            trajectories.append(agent_positions)
            zone_poss.append(getattr(env, 'zone_positions', []))
            env.close()
            
            # Print summary
            print(f"Summary (strength={steering_strength}) for World {world_file}:")
            for subgoal, step in episode_events['goal_steps'].items():
                print(f"  Subgoal '{subgoal}' satisfied at step {step}")
        
        # Fix zone keys for plotting
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
        
        zone_poss = [fix_zone_keys(z) for z in zone_poss]
        
        # Plot trajectories for this strength
        num_cols = 3
        num_rows = 2
        plt.figure(figsize=(16, 8))
        draw_trajectories(zone_poss, trajectories, num_cols, num_rows)
        plt.title(f'Trajectories with Steering Strength {steering_strength}')
        plt.savefig(f'trajectories_steering_strength_{steering_strength}.png')
        plt.close()
        
        all_results[steering_strength] = {
            'trajectories': trajectories,
            'zone_poss': zone_poss
        }
    
    print(f"\n=== Strong Steering Analysis Complete ===")
    print(f"Generated trajectory plots for strengths: {STEERING_STRENGTHS}")
    print("Check the PNG files to see the effect of different steering strengths!")

if __name__ == '__main__':
    main() 