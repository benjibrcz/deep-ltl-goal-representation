#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

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
N_WORLDS = 50  # Reduced for faster testing
MAX_STEPS = 200  # Reduced for faster testing

# Color mapping
COLOR_MAP = {
    "blue": 0, "green": 1, "yellow": 2, "pink": 3, "magenta": 4, "orange": 5, "red": 6
}
COLOR_RGB = {
    0: "#4C72B0",  # blue
    1: "#55A868",  # green  
    2: "#E1C027",  # yellow
    3: "#BB78A5",  # pink
    4: "#C44E52",  # magenta
    5: "#FF7F0E",  # orange
    6: "#D62728",  # red
}
COLOR_NAMES = ["blue", "green", "yellow", "pink", "magenta", "orange", "red"]

class ZoneAlignmentAnalyzer:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.layer = dict(model.named_modules())[layer_name]
        self.features = []
        self.positions = []
        self.colors = []
        self.world_info = []
        
    def hook_fn(self, module, input, output):
        """Hook function to capture layer activations"""
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        self.features.append(feat)
    
    def collect_data(self, env, sampler_fn):
        """Collect zone data and activations from multiple worlds"""
        print(f"Collecting data from layer: {self.layer_name}")
        
        # Register hook
        handle = self.layer.register_forward_hook(self.hook_fn)
        
        # Create agent for consistent behavior
        props = set(env.get_propositions())
        search = ExhaustiveSearch(self.model, props, num_loops=2)
        agent = Agent(self.model, search=search, propositions=props, verbose=False)
        
        for world_idx in trange(N_WORLDS, desc="Collecting worlds"):
            # Reset environment with new world
            ret = env.reset(seed=SEED + world_idx)
            obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
            agent.reset()
            
            # Extract zone information first
            self.extract_zone_info(env, world_idx)
            
            # Get agent action to trigger forward pass
            try:
                action = agent.get_action(obs, info, deterministic=True)
            except:
                # Fallback if agent fails
                action = env.action_space.sample()
        
        handle.remove()
        
        # Convert to arrays
        self.features = np.array(self.features)
        self.positions = np.array(self.positions)
        self.colors = np.array(self.colors)
        
        print(f"Collected {len(self.features)} samples")
        print(f"Feature shape: {self.features.shape}")
        print(f"Position shape: {self.positions.shape}")
        print(f"Color shape: {self.colors.shape}")
        
        # Verify shapes match
        if len(self.features) != len(self.positions):
            print(f"WARNING: Feature count ({len(self.features)}) doesn't match position count ({len(self.positions)})")
            # Truncate to match
            min_len = min(len(self.features), len(self.positions))
            self.features = self.features[:min_len]
            self.positions = self.positions[:min_len]
            self.colors = self.colors[:min_len]
            print(f"Truncated to {min_len} samples")
    
    def extract_zone_info(self, env, world_idx):
        """Extract zone positions and colors from environment"""
        try:
            # Try to get zone positions from environment
            if hasattr(env, 'zone_positions'):
                zone_pos = env.zone_positions
                if isinstance(zone_pos, dict):
                    positions = []
                    colors = []
                    for key, pos in sorted(zone_pos.items()):
                        positions.extend(pos[:2])  # Take x, y coordinates
                        # Extract color from key
                        color_name = None
                        for color in COLOR_MAP:
                            if color in key.lower():
                                color_name = color
                                break
                        if color_name:
                            colors.append(COLOR_MAP[color_name])
                        else:
                            colors.append(0)  # Default to blue
                    
                    self.positions.append(positions)
                    self.colors.append(colors)
                    self.world_info.append({
                        'world_idx': world_idx,
                        'num_zones': len(colors),
                        'positions': positions,
                        'colors': colors
                    })
                else:
                    # Fallback: use layout information
                    self.extract_from_layout(env, world_idx)
            else:
                # Fallback: use layout information
                self.extract_from_layout(env, world_idx)
                
        except Exception as e:
            print(f"Error extracting zone info for world {world_idx}: {e}")
            # Add dummy data to maintain alignment
            self.positions.append([0, 0, 0, 0])  # 2 zones at origin
            self.colors.append([0, 1])  # blue, green
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': 2,
                'positions': [0, 0, 0, 0],
                'colors': [0, 1]
            })
    
    def extract_from_layout(self, env, world_idx):
        """Extract zone info from environment layout"""
        try:
            layout = env.task.world_info.layout
            positions = []
            colors = []
            
            for key, val in sorted(layout.items()):
                if 'zone' in key.lower():
                    # Extract position
                    if isinstance(val, dict) and 'pos' in val:
                        pos = val['pos']
                        positions.extend(pos[:2])
                    elif isinstance(val, np.ndarray):
                        positions.extend(val[:2].tolist())
                    else:
                        positions.extend([0, 0])
                    
                    # Extract color
                    color_name = None
                    for color in COLOR_MAP:
                        if color in key.lower():
                            color_name = color
                            break
                    if color_name:
                        colors.append(COLOR_MAP[color_name])
                    else:
                        colors.append(0)  # Default to blue
            
            if not positions:  # No zones found
                positions = [0, 0, 0, 0]
                colors = [0, 1]
            
            self.positions.append(positions)
            self.colors.append(colors)
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': len(colors),
                'positions': positions,
                'colors': colors
            })
            
        except Exception as e:
            print(f"Error extracting from layout for world {world_idx}: {e}")
            self.positions.append([0, 0, 0, 0])
            self.colors.append([0, 1])
            self.world_info.append({
                'world_idx': world_idx,
                'num_zones': 2,
                'positions': [0, 0, 0, 0],
                'colors': [0, 1]
            })
    
    def train_probes(self):
        """Train position and color prediction probes"""
        print("Training position and color probes...")
        
        # Train position probe (Ridge regression)
        self.position_probe = Ridge(alpha=1.0)
        self.position_probe.fit(self.features, self.positions)
        
        # Calculate position prediction accuracy
        pos_pred = self.position_probe.predict(self.features)
        pos_mse = mean_squared_error(self.positions, pos_pred)
        print(f"Position MSE: {pos_mse:.4f}")
        
        # Train color probes (Logistic regression for each zone)
        self.color_probes = []
        color_accuracies = []
        
        for zone_idx in range(self.colors.shape[1]):
            zone_colors = self.colors[:, zone_idx]
            unique_colors = np.unique(zone_colors)
            
            if len(unique_colors) > 1:
                # Train classifier
                clf = LogisticRegression(max_iter=1000, random_state=SEED)
                clf.fit(self.features, zone_colors)
                self.color_probes.append(clf)
                
                # Calculate accuracy
                pred_colors = clf.predict(self.features)
                acc = accuracy_score(zone_colors, pred_colors)
                color_accuracies.append(acc)
                print(f"Zone {zone_idx} color accuracy: {acc:.3f}")
            else:
                # Only one color present
                self.color_probes.append(None)
                color_accuracies.append(1.0)
                print(f"Zone {zone_idx} color accuracy: 1.000 (single class)")
        
        return {
            'position_mse': pos_mse,
            'color_accuracies': color_accuracies,
            'avg_color_accuracy': np.mean(color_accuracies)
        }
    
    def predict_zones(self, features):
        """Predict zone positions and colors from features"""
        # Predict positions
        pos_pred = self.position_probe.predict(features)
        
        # Predict colors
        color_pred = []
        for zone_idx, probe in enumerate(self.color_probes):
            if probe is not None:
                zone_colors = probe.predict(features)
                color_pred.append(zone_colors)
            else:
                # Use most common color for this zone
                most_common = np.bincount(self.colors[:, zone_idx]).argmax()
                color_pred.append([most_common] * len(features))
        
        color_pred = np.array(color_pred).T
        
        return pos_pred, color_pred

def analyze_rollout_alignment(probe, model, env, layer_name, world_idx=0, max_steps=200):
    """Analyze how well predicted zones align with agent movement during rollout"""
    print(f"\nAnalyzing zone alignment during rollout in world {world_idx}...")
    
    # Reset environment to a specific world
    ret = env.reset(seed=SEED + world_idx)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    
    # Prepare agent
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    
    # Prepare to collect activations
    layer = dict(model.named_modules())[layer_name]
    rollout_features = []
    agent_positions = []
    agent_actions = []
    
    def hook_fn(module, input, output):
        if hasattr(output, 'detach'):
            feat = output.detach().cpu().squeeze().numpy()
        else:
            feat = output.squeeze().cpu().numpy()
        rollout_features.append(feat)
    
    handle = layer.register_forward_hook(hook_fn)
    
    # Rollout
    for t in range(max_steps):
        agent_positions.append(np.array(env.agent_pos[:2]))
        try:
            action = agent.get_action(obs, info, deterministic=True)
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            if hasattr(action, 'flatten'):
                action = action.flatten()
            action = np.asarray(action)
            agent_actions.append(action.copy())
        except:
            action = env.action_space.sample()
            agent_actions.append(action.copy())
        
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        if done:
            break
    
    handle.remove()
    
    rollout_features = np.array(rollout_features)
    agent_positions = np.array(agent_positions)
    agent_actions = np.array(agent_actions)
    actual_steps = len(rollout_features)
    
    print(f"Actual rollout length: {actual_steps} steps")
    
    # Predict zones at each step
    pos_preds, col_preds = probe.predict_zones(rollout_features)
    num_zones = len(probe.colors[world_idx])
    true_pos = np.array(probe.positions[world_idx]).reshape(num_zones, 2)
    true_col = probe.colors[world_idx]
    
    # Calculate alignment metrics
    alignment_metrics = calculate_alignment_metrics(
        agent_positions, agent_actions, pos_preds, col_preds, 
        true_pos, true_col, num_zones
    )
    
    return alignment_metrics, {
        'rollout_features': rollout_features,
        'agent_positions': agent_positions,
        'agent_actions': agent_actions,
        'pos_preds': pos_preds,
        'col_preds': col_preds,
        'true_pos': true_pos,
        'true_col': true_col,
        'num_zones': num_zones
    }

def calculate_alignment_metrics(agent_positions, agent_actions, pos_preds, col_preds, true_pos, true_col, num_zones):
    """Calculate various alignment metrics"""
    metrics = {}
    
    # 1. Movement direction alignment
    movement_directions = []
    zone_directions = []
    alignment_scores = []
    
    for t in range(1, len(agent_positions)):
        # Calculate agent movement direction
        movement = agent_positions[t] - agent_positions[t-1]
        movement_norm = np.linalg.norm(movement)
        if movement_norm > 0:
            movement_dir = movement / movement_norm
            movement_directions.append(movement_dir)
            
            # Calculate direction to each predicted zone
            step_pred_pos = pos_preds[t].reshape(num_zones, 2)
            step_pred_col = col_preds[t]
            
            best_alignment = -1
            best_zone_dir = None
            
            for zone_idx in range(num_zones):
                # Direction from agent to predicted zone
                zone_dir = step_pred_pos[zone_idx] - agent_positions[t]
                zone_norm = np.linalg.norm(zone_dir)
                
                if zone_norm > 0:
                    zone_dir = zone_dir / zone_norm
                    # Calculate alignment (dot product)
                    alignment = np.dot(movement_dir, zone_dir)
                    
                    if alignment > best_alignment:
                        best_alignment = alignment
                        best_zone_dir = zone_dir
            
            if best_zone_dir is not None:
                zone_directions.append(best_zone_dir)
                alignment_scores.append(best_alignment)
    
    metrics['movement_alignment'] = {
        'mean_alignment': np.mean(alignment_scores) if alignment_scores else 0,
        'std_alignment': np.std(alignment_scores) if alignment_scores else 0,
        'positive_alignment_rate': np.mean([a > 0 for a in alignment_scores]) if alignment_scores else 0,
        'strong_alignment_rate': np.mean([a > 0.5 for a in alignment_scores]) if alignment_scores else 0
    }
    
    # 2. Goal-seeking behavior (alignment with true goal directions)
    goal_alignment_scores = []
    for t in range(len(agent_positions)):
        # Find current goal (simplified - assume blue is always the goal)
        current_goal_color = 0  # blue
        goal_pos = None
        
        # Find the true position of the current goal
        for zone_idx, color in enumerate(true_col):
            if color == current_goal_color:
                goal_pos = true_pos[zone_idx]
                break
        
        if goal_pos is not None:
            # Direction to true goal
            goal_dir = goal_pos - agent_positions[t]
            goal_norm = np.linalg.norm(goal_dir)
            
            if goal_norm > 0:
                goal_dir = goal_dir / goal_norm
                
                # Find predicted zone of the same color
                step_pred_pos = pos_preds[t].reshape(num_zones, 2)
                step_pred_col = col_preds[t]
                
                for zone_idx, pred_color in enumerate(step_pred_col):
                    if pred_color == current_goal_color:
                        pred_zone_dir = step_pred_pos[zone_idx] - agent_positions[t]
                        pred_norm = np.linalg.norm(pred_zone_dir)
                        
                        if pred_norm > 0:
                            pred_zone_dir = pred_zone_dir / pred_norm
                            # Calculate alignment between predicted and true goal direction
                            goal_alignment = np.dot(goal_dir, pred_zone_dir)
                            goal_alignment_scores.append(goal_alignment)
                        break
    
    metrics['goal_alignment'] = {
        'mean_goal_alignment': np.mean(goal_alignment_scores) if goal_alignment_scores else 0,
        'std_goal_alignment': np.std(goal_alignment_scores) if goal_alignment_scores else 0,
        'positive_goal_alignment_rate': np.mean([a > 0 for a in goal_alignment_scores]) if goal_alignment_scores else 0
    }
    
    # 3. Spatial consistency (how much predictions drift)
    spatial_drift = []
    for zone_idx in range(num_zones):
        zone_positions = []
        for t in range(len(pos_preds)):
            step_pred_pos = pos_preds[t].reshape(num_zones, 2)
            zone_positions.append(step_pred_pos[zone_idx])
        
        zone_positions = np.array(zone_positions)
        
        # Calculate total distance traveled by this predicted zone
        total_distance = 0
        for t in range(1, len(zone_positions)):
            distance = np.linalg.norm(zone_positions[t] - zone_positions[t-1])
            total_distance += distance
        
        spatial_drift.append(total_distance)
    
    metrics['spatial_consistency'] = {
        'mean_zone_drift': np.mean(spatial_drift),
        'std_zone_drift': np.std(spatial_drift),
        'total_drift': np.sum(spatial_drift)
    }
    
    # 4. Prediction accuracy (distance to true zones)
    prediction_errors = []
    for t in range(len(pos_preds)):
        step_pred_pos = pos_preds[t].reshape(num_zones, 2)
        
        for zone_idx in range(num_zones):
            pred_pos = step_pred_pos[zone_idx]
            true_pos_zone = true_pos[zone_idx]
            error = np.linalg.norm(pred_pos - true_pos_zone)
            prediction_errors.append(error)
    
    metrics['prediction_accuracy'] = {
        'mean_prediction_error': np.mean(prediction_errors),
        'std_prediction_error': np.std(prediction_errors),
        'min_prediction_error': np.min(prediction_errors),
        'max_prediction_error': np.max(prediction_errors)
    }
    
    return metrics

def create_alignment_visualization(metrics, rollout_data, save_path=None):
    """Create visualization of alignment analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Movement alignment over time
    ax1 = axes[0, 0]
    alignment_scores = []
    for t in range(1, len(rollout_data['agent_positions'])):
        movement = rollout_data['agent_positions'][t] - rollout_data['agent_positions'][t-1]
        movement_norm = np.linalg.norm(movement)
        if movement_norm > 0:
            movement_dir = movement / movement_norm
            step_pred_pos = rollout_data['pos_preds'][t].reshape(rollout_data['num_zones'], 2)
            
            best_alignment = -1
            for zone_idx in range(rollout_data['num_zones']):
                zone_dir = step_pred_pos[zone_idx] - rollout_data['agent_positions'][t]
                zone_norm = np.linalg.norm(zone_dir)
                if zone_norm > 0:
                    zone_dir = zone_dir / zone_norm
                    alignment = np.dot(movement_dir, zone_dir)
                    best_alignment = max(best_alignment, alignment)
            
            alignment_scores.append(best_alignment)
    
    ax1.plot(alignment_scores, 'b-', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
    ax1.set_title('Movement-Zone Alignment Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Alignment Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Goal alignment over time
    ax2 = axes[0, 1]
    goal_alignments = []
    for t in range(len(rollout_data['agent_positions'])):
        # Simplified goal alignment calculation
        goal_pos = rollout_data['true_pos'][0]  # Assume first zone is goal
        goal_dir = goal_pos - rollout_data['agent_positions'][t]
        goal_norm = np.linalg.norm(goal_dir)
        
        if goal_norm > 0:
            goal_dir = goal_dir / goal_norm
            step_pred_pos = rollout_data['pos_preds'][t].reshape(rollout_data['num_zones'], 2)
            
            # Find predicted zone of same color as goal
            pred_zone_dir = step_pred_pos[0] - rollout_data['agent_positions'][t]
            pred_norm = np.linalg.norm(pred_zone_dir)
            
            if pred_norm > 0:
                pred_zone_dir = pred_zone_dir / pred_norm
                goal_alignment = np.dot(goal_dir, pred_zone_dir)
                goal_alignments.append(goal_alignment)
    
    ax2.plot(goal_alignments, 'g-', alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_title('Goal Direction Alignment Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Goal Alignment Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Zone drift over time
    ax3 = axes[1, 0]
    for zone_idx in range(rollout_data['num_zones']):
        zone_positions = []
        for t in range(len(rollout_data['pos_preds'])):
            step_pred_pos = rollout_data['pos_preds'][t].reshape(rollout_data['num_zones'], 2)
            zone_positions.append(step_pred_pos[zone_idx])
        
        zone_positions = np.array(zone_positions)
        ax3.plot(zone_positions[:, 0], zone_positions[:, 1], 'o-', 
                label=f'Zone {zone_idx}', alpha=0.7, markersize=3)
    
    # Plot true zones
    for zone_idx, (pos, col) in enumerate(zip(rollout_data['true_pos'], rollout_data['true_col'])):
        color = COLOR_RGB.get(col, "#000000")
        circle = patches.Circle(pos, 0.3, facecolor=color, alpha=0.3, edgecolor='black')
        ax3.add_patch(circle)
        ax3.text(pos[0], pos[1], f'True {zone_idx}', ha='center', va='center', fontsize=8)
    
    ax3.set_title('Predicted Zone Trajectories')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
Zone Alignment Analysis Summary

Movement Alignment:
• Mean: {metrics['movement_alignment']['mean_alignment']:.3f}
• Positive Rate: {metrics['movement_alignment']['positive_alignment_rate']:.1%}
• Strong Rate: {metrics['movement_alignment']['strong_alignment_rate']:.1%}

Goal Alignment:
• Mean: {metrics['goal_alignment']['mean_goal_alignment']:.3f}
• Positive Rate: {metrics['goal_alignment']['positive_goal_alignment_rate']:.1%}

Spatial Consistency:
• Mean Zone Drift: {metrics['spatial_consistency']['mean_zone_drift']:.3f}
• Total Drift: {metrics['spatial_consistency']['total_drift']:.3f}

Prediction Accuracy:
• Mean Error: {metrics['prediction_accuracy']['mean_prediction_error']:.3f}
• Error Range: {metrics['prediction_accuracy']['min_prediction_error']:.3f} - {metrics['prediction_accuracy']['max_prediction_error']:.3f}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved alignment analysis to {save_path}")
    plt.show()

def main():
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Zone Alignment Analysis ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
    print()
    
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
    
    # Create environment for data collection
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Use the best layer from previous analysis
    layer_name = "actor.enc.2"
    print(f"Using layer: {layer_name}")
    
    # Create probe and collect data
    probe = ZoneAlignmentAnalyzer(model, layer_name)
    probe.collect_data(env, sampler_fn)
    probe.train_probes()
    
    # Analyze alignment during rollout
    metrics, rollout_data = analyze_rollout_alignment(probe, model, env, layer_name)
    
    # Print results
    print("\n=== Alignment Analysis Results ===")
    print(f"Movement Alignment: {metrics['movement_alignment']['mean_alignment']:.3f}")
    print(f"Positive Movement Alignment Rate: {metrics['movement_alignment']['positive_alignment_rate']:.1%}")
    print(f"Strong Movement Alignment Rate: {metrics['movement_alignment']['strong_alignment_rate']:.1%}")
    print(f"Goal Alignment: {metrics['goal_alignment']['mean_goal_alignment']:.3f}")
    print(f"Mean Zone Drift: {metrics['spatial_consistency']['mean_zone_drift']:.3f}")
    print(f"Mean Prediction Error: {metrics['prediction_accuracy']['mean_prediction_error']:.3f}")
    
    # Create visualization
    create_alignment_visualization(metrics, rollout_data, 'zone_alignment_analysis.png')
    
    env.close()
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- zone_alignment_analysis.png")

if __name__ == '__main__':
    main() 