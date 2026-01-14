#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

class NonlinearProbe(nn.Module):
    """Deeper neural network probe for non-linear steering (2 hidden layers)"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),  # Second hidden layer, same size
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
        # print(f"[DEBUG] Steering layer: {self.layer}, output type: {type(output)}, output shape: {getattr(output, 'shape', None)}")
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

def train_nonlinear_probe_for_steering(model, env, sampler_fn):
    """Train a non-linear probe to get weights for steering"""
    print("Training non-linear probe for steering...")
    
    # Hook into selected network layer
    feats = []
    def hook_fn(mod, inp, out):
        # print(f"[DEBUG] Probe training hook, layer: {STEER_LAYER}, output type: {type(out)}, output shape: {getattr(out, 'shape', None)}")
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
    optimizer = optim.Adam(probe.parameters(), lr=0.001)
    
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
    
    print(f"Non-linear probe accuracy: {accuracy:.3f}")
    
    # Get weights for steering
    weights = probe.get_weights()
    return weights, 0.0  # No intercept for non-linear probe

def test_multiple_layers_nonlinear(model, env, sampler_fn):
    """Test non-linear probe accuracy on multiple layers to find the best one for steering"""
    print("\n=== Testing Multiple Layers with Non-linear Probes ===")
    
    layers_to_test = [
        'ltl_rnn',
        # 'policy_mlp_0',
        # 'policy_mlp_1', 
        # 'policy_mlp_2',
        # 'env_net',
        # 'env_net_mlp_0',
        # 'env_net_mlp_1',
        # 'env_net_mlp_2',
        # 'env_net_mlp_3'
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
        
        # Train non-linear probe
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        probe = NonlinearProbe(input_dim=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(probe.parameters(), lr=0.001)
        
        # Training loop
        probe.train()
        for epoch in range(50):  # Shorter training for layer testing
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
            accuracy = (predicted_labels == y_tensor).float().mean().item()
        
        print(f"  Non-linear probe accuracy: {accuracy:.3f} ({len(y)} samples)")
        
        results.append({
            'layer': layer_name,
            'accuracy': accuracy,
            'num_samples': len(y),
            'weights': probe.get_weights(),
            'intercept': 0.0
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("NON-LINEAR LAYER COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Accuracy':<10} {'Samples':<8}")
    print("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{result['layer']:<15} {result['accuracy']:<10.3f} {result['num_samples']:<8}")
    
    # Find best layer
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest layer for non-linear steering: {best_result['layer']} (accuracy: {best_result['accuracy']:.3f})")
        return best_result
    else:
        print("\nNo valid layers found!")
        return None

def run_steered_rollout(model, env, sampler_fn, probe_weights, steering_strength, world_idx=0):
    print(f"Running steered rollout with strength {steering_strength}...")
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    # Print all possible agent position attributes after reset
    print("[DEBUG] After env.reset():")
    if hasattr(env, 'agent_pos'):
        print(f"  env.agent_pos: {getattr(env, 'agent_pos')}")
    if hasattr(env, 'agent') and hasattr(env.agent, 'pos'):
        print(f"  env.agent.pos: {getattr(env.agent, 'pos')}")
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'agent_pos'):
        print(f"  env.unwrapped.agent_pos: {getattr(env.unwrapped, 'agent_pos')}")
    if hasattr(env, 'xlim'):
        print(f"  env.xlim: {getattr(env, 'xlim')}")
    if hasattr(env, 'ylim'):
        print(f"  env.ylim: {getattr(env, 'ylim')}")
    # After loading world file, print agent position again
    if hasattr(env, 'world_file'):
        print(f"[DEBUG] After loading world file: {env.world_file}")
        if hasattr(env, 'agent_pos'):
            print(f"  env.agent_pos: {getattr(env, 'agent_pos')}")
        if hasattr(env, 'agent') and hasattr(env.agent, 'pos'):
            print(f"  env.agent.pos: {getattr(env.agent, 'pos')}")
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'agent_pos'):
            print(f"  env.unwrapped.agent_pos: {getattr(env.unwrapped, 'agent_pos')}")
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    steerer = SubgoalSteerer(model, probe_weights, steering_strength, layer=STEER_LAYER)
    handle = None
    if steering_strength > 0.0:
        handle = get_layer_and_hook(model, STEER_LAYER, steerer.hook_fn)
    agent_positions = []
    episode_events = {'zone_entries': {}, 'goal_steps': {}, 'physical_goal_steps': {}}
    for step in range(MAX_STEPS):
        # --- DEBUG PRINTS ---
        # Print current world file if available
        if hasattr(env, 'world_file'):
            print(f"Step {step}: World file: {env.world_file}")
        # Try all possible agent position attributes and print them
        pos = None
        pos_source = None
        if hasattr(env, 'agent') and hasattr(env.agent, 'pos'):
            pos = np.array(env.agent.pos)
            pos_source = 'env.agent.pos'
        elif hasattr(env, 'agent_pos'):
            pos = np.array(env.agent_pos)
            pos_source = 'env.agent_pos'
        elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'agent_pos'):
            pos = np.array(env.unwrapped.agent_pos)
            pos_source = 'env.unwrapped.agent_pos'
        # Comment out per-step print/debug statements
        # print(f"Step {step}: Using {pos_source} for agent position: {pos}")
        if pos is not None and (np.any(np.abs(pos) > 5)):
            # print(f"[WARNING] Agent position out of expected bounds: {pos}")
            pass
        # Zone centers and radii
        zone_centers = getattr(env, 'zone_positions', None)
        zone_radii = getattr(env, 'zone_radii', None)
        if zone_centers is not None and zone_radii is not None:
            for i, (zc, zr) in enumerate(zip(zone_centers, zone_radii)):
                if pos is not None and not isinstance(pos, float) and isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
                    dist = np.linalg.norm(np.array(pos[:2]) - np.array(zc[:2]))
                else:
                    dist = None
                # print(f"Step {step}: Zone {i} center: {zc}, radius: {zr}, dist to agent: {dist}")
        # Propositions
        # if isinstance(obs, dict):
        #     print(f"Step {step}: obs['propositions']: {obs.get('propositions', None)}")
        # else:
        #     print(f"Step {step}: obs: {obs}")
        # if isinstance(info, dict):
        #     print(f"Step {step}: info['propositions']: {info.get('propositions', None)}")
        #     print(f"Step {step}: info (raw): {info}")
        # else:
        #     print(f"Step {step}: info: {info}")
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
        # print(f"Step {step}: Current subgoal: {current_goal}")
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
        # --- Physical goal reached check ---
        if current_goal and zone_centers is not None and zone_radii is not None:
            print(f"[DEBUG] Step {step}: Checking physical goal for subgoal '{current_goal}'")
            print(f"[DEBUG] Agent position: {pos}")
            # Print all available zones
            if isinstance(zone_centers, dict):
                print(f"[DEBUG] zone_centers (dict): {zone_centers}")
                print(f"[DEBUG] zone_radii (dict): {zone_radii}")
            else:
                print(f"[DEBUG] zone_centers (list): {zone_centers}")
                print(f"[DEBUG] zone_radii (list): {zone_radii}")
            found_match = False
            for i, (zc, zr) in enumerate(zip(zone_centers.values() if isinstance(zone_centers, dict) else zone_centers, zone_radii.values() if isinstance(zone_radii, dict) else zone_radii)):
                zone_name = None
                if isinstance(zone_centers, dict):
                    # Try to match by color name
                    for k, v in zone_centers.items():
                        if hasattr(current_goal, 'lower') and k.lower().startswith(current_goal.lower()) and np.allclose(v, zc):
                            zone_name = k
                            break
                else:
                    zone_name = str(i)
                if pos is not None and zc is not None and len(zc) >= 2:
                    dist = np.linalg.norm(np.array(pos[:2]) - np.array(zc[:2]))
                    print(f"[DEBUG] Comparing to zone '{zone_name}': center={zc}, radius={zr}, dist={dist}")
                    if hasattr(current_goal, 'lower') and zone_name and zone_name.lower().startswith(current_goal.lower()):
                        found_match = True
                        if dist <= zr:
                            print(f"[DEBUG] PHYSICALLY REACHED subgoal '{current_goal}' at step {step}, dist={dist}, radius={zr}")
                            if current_goal not in episode_events['physical_goal_steps']:
                                episode_events['physical_goal_steps'][current_goal] = step
            if not found_match:
                print(f"[DEBUG] No matching zone found for subgoal '{current_goal}'!")
        # --- Logical goal satisfaction debug ---
        # When a goal is satisfied, print all zones with the current subgoal colour and their distances from the agent
        if zone_centers is not None and current_goal is not None:
            # print(f"[DEBUG] zone_centers type: {type(zone_centers)}, keys: {list(zone_centers.keys()) if hasattr(zone_centers, 'keys') else zone_centers}")
            # print(f"[DEBUG] current_goal: {current_goal}")
            print(f"All zones for subgoal '{current_goal}':")
            for zone_name, center in zone_centers.items():
                if zone_name.startswith(current_goal):
                    if agent_pos is not None:
                        d = np.linalg.norm(np.array(agent_pos[:2]) - np.array(center[:2]))
                        print(f"  Zone '{zone_name}' at {center}, distance from agent: {d:.3f}")
                    else:
                        print(f"  Zone '{zone_name}' at {center}, agent position unknown!")
        # --- Debug: Print agent position before goal satisfaction message ---
        # print(f"[DEBUG] (Goal satisfaction) At step {step}: agent_pos variable = {pos}")
        # ---
        # Initialize to 'N/A' to avoid UnboundLocalError
        zone_center = 'N/A'
        dist = 'N/A'
        zone_radius = 'N/A'
        agent_pos = getattr(env, 'agent_pos', None)
        zone_center = pos if pos is not None else 'N/A'
        dist = dist if dist is not None else 'N/A'
        zone_radius = zone_radii if zone_radii is not None else 'N/A'
        # Print agent and all relevant zone locations when a goal is reached
        print(f"Goal satisfied! Subgoal: {current_goal}")
        agent_xy = agent_pos[:2] if agent_pos is not None else None
        print(f"  Agent position (x, y): {agent_xy}")
        if zone_centers is not None and current_goal is not None:
            for zone_name, center in zone_centers.items():
                if zone_name.startswith(current_goal):
                    if agent_xy is not None:
                        center_xy = np.array(center[:2])
                        d = np.linalg.norm(np.array(agent_xy) - center_xy)
                        print(f"  Zone '{zone_name}' at {center_xy}, distance: {d:.3f}")
                    else:
                        print(f"  Zone '{zone_name}' at {center}, agent position unknown!")
    if handle:
        handle.remove()
    return agent_positions, episode_events

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("=== Non-linear Subgoal Steering Trajectory Visualization ===")
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
    
    # Test multiple layers first with non-linear probes
    best_layer_result = test_multiple_layers_nonlinear(probe_model, probe_env, sampler_fn)
    
    if best_layer_result:
        # Update the global STEER_LAYER with the best one
        global STEER_LAYER
        STEER_LAYER = best_layer_result['layer']
        print(f"\nUsing best layer for non-linear steering: {STEER_LAYER}")
        
        # Train non-linear probe on the best layer
        probe_weights, probe_intercept = train_nonlinear_probe_for_steering(probe_model, probe_env, sampler_fn)
    else:
        print("Falling back to default layer and probe training...")
        probe_weights, probe_intercept = train_nonlinear_probe_for_steering(probe_model, probe_env, sampler_fn)
    
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
    plt.title('Unsteered Trajectories (Non-linear Probe Test)')
    plt.savefig('trajectories_selected_worlds_unsteered_nonlinear2.png')
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
        # Print physical goal steps
        if 'physical_goal_steps' in episode_events:
            for subgoal, step in episode_events['physical_goal_steps'].items():
                print(f"  Subgoal '{subgoal}' PHYSICALLY REACHED at step {step}")
    
    zone_poss_steered = [fix_zone_keys(z) for z in zone_poss_steered]
    num_cols = 3
    num_rows = 2
    plt.figure(figsize=(16, 8))
    draw_trajectories(zone_poss_steered, trajectories_steered, num_cols, num_rows)
    plt.title('Steered Trajectories (Non-linear Probe)')
    plt.savefig('trajectories_selected_worlds_steered_nonlinear2.png')
    plt.close()

if __name__ == '__main__':
    main() 