#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches
from datetime import datetime

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 5  # Number of rollouts per test type to visualize
MAX_STEPS  = 40  # Maximum steps per trajectory for visualization
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_test_trajectories_by_split(model, layer_name, sampler, n_rollouts_per_split=5, max_steps=40):
    """
    Collect test trajectories organized by generalization split type.
    Returns trajectories for: same_rollout, same_world, different_world
    """
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]
    
    trajectories_by_split = {
        'same_rollout': [],
        'same_world': [],
        'different_world': []
    }
    
    # World splits for different test types
    train_worlds = [0, 1, 2]  # Use fewer worlds for visualization
    test_worlds = [7, 8, 9]   # Different worlds for testing
    
    print("Collecting test trajectories...")
    
    # 1. Same rollout trajectories (temporal split)
    print("- Same rollout (temporal split)...")
    world_id = train_worlds[0]  # Use first training world
    for traj_idx in range(n_rollouts_per_split):
        trajectory = collect_single_trajectory(
            env, agent, module, world_id, 
            seed=SEED + world_id * 1000 + traj_idx * 100, 
            max_steps=max_steps
        )
        if trajectory and len(trajectory['positions']) >= 4:
            # Split trajectory temporally
            mid_point = len(trajectory['positions']) // 2
            test_trajectory = {
                'activations': trajectory['activations'][mid_point:],
                'positions': trajectory['positions'][mid_point:],
                'next_positions': trajectory['next_positions'][mid_point:],
                'world_id': world_id,
                'split_type': 'same_rollout',
                'trajectory_id': traj_idx
            }
            trajectories_by_split['same_rollout'].append(test_trajectory)
    
    # 2. Same world trajectories (spatial split)
    print("- Same world (spatial split)...")
    world_id = train_worlds[1]  # Use second training world
    for traj_idx in range(n_rollouts_per_split):
        # Use different seeds to get different starting positions in same world
        trajectory = collect_single_trajectory(
            env, agent, module, world_id,
            seed=SEED + world_id * 1000 + (traj_idx + 10) * 100,  # Different from training seeds
            max_steps=max_steps
        )
        if trajectory and len(trajectory['positions']) >= 2:
            trajectories_by_split['same_world'].append({
                **trajectory,
                'world_id': world_id,
                'split_type': 'same_world',
                'trajectory_id': traj_idx
            })
    
    # 3. Different world trajectories (environmental split)
    print("- Different world (environmental split)...")
    for traj_idx in range(n_rollouts_per_split):
        world_id = test_worlds[traj_idx % len(test_worlds)]
        trajectory = collect_single_trajectory(
            env, agent, module, world_id,
            seed=SEED + world_id * 1000 + traj_idx * 100,
            max_steps=max_steps
        )
        if trajectory and len(trajectory['positions']) >= 2:
            trajectories_by_split['different_world'].append({
                **trajectory,
                'world_id': world_id,
                'split_type': 'different_world', 
                'trajectory_id': traj_idx
            })
    
    env.close()
    return trajectories_by_split

def collect_single_trajectory(env, agent, module, world_id, seed, max_steps):
    """Collect a single trajectory with activations and positions."""
    try:
        # Skip world file loading to allow proper seeding/randomization
        obs = env.reset(seed=seed)
        agent.reset()
        
        activations = []
        positions = []
        next_positions = []
        
        def grab(m, inp, out):
            x = out[1] if isinstance(out, tuple) else out
            activations.append(x.detach().cpu().numpy().ravel())
        
        h = module.register_forward_hook(grab)
        
        done = False
        prev_pos = None
        
        for step in range(max_steps):
            if done:
                break
                
            current_pos = env.agent_pos[:2].copy()
            positions.append(current_pos.copy())
            
            # Take action (this records activation)
            a = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, _, done, _ = env.step(a)
            
            # Record next position
            next_pos = env.agent_pos[:2].copy()
            next_positions.append(next_pos.copy())
        
        h.remove()
        
        if len(positions) >= 2:
            return {
                'activations': np.array(activations),
                'positions': np.array(positions),
                'next_positions': np.array(next_positions),
                'zone_pos': dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
            }
        return None
        
    except Exception as e:
        print(f"Error collecting trajectory for world {world_id}: {e}")
        return None

def train_next_step_probe(trajectories_by_split, layer_name):
    """Train a probe on training data to predict next positions."""
    print("Training next-step position probe...")
    
    # Collect training data from same_rollout trajectories (first half of each)
    X_train, y_train = [], []
    
    for trajectory in trajectories_by_split['same_rollout']:
        activations = trajectory['activations']
        next_positions = trajectory['next_positions']
        
        # Use first half for training (temporal split)
        mid_point = len(activations) // 2
        train_activations = activations[:mid_point]
        train_next_pos = next_positions[:mid_point]
        
        if len(train_activations) > 0:
            X_train.extend(train_activations.tolist())
            y_train.extend(train_next_pos.tolist())
    
    if len(X_train) == 0:
        raise ValueError("No training data collected")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training probe on {len(X_train)} samples...")
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    
    return probe

def predict_trajectory_positions(probe, activations, start_position):
    """Predict trajectory positions step by step from activations."""
    if len(activations) == 0:
        return np.array([start_position])
    
    predicted_positions = [start_position]
    
    for i in range(len(activations)):
        # Predict next position from current activation
        next_pos_pred = probe.predict(activations[i:i+1].reshape(1, -1))[0]
        predicted_positions.append(next_pos_pred)
    
    return np.array(predicted_positions)

def draw_zone_background(ax, zone_pos, alpha=0.3):
    """Draw zone backgrounds if available."""
    if not zone_pos:
        return
    
    colors = {'red': 'red', 'blue': 'blue', 'green': 'green', 'yellow': 'gold',
              'orange': 'orange', 'purple': 'purple', 'brown': 'brown', 'pink': 'pink'}
    
    for zone_name, pos in zone_pos.items():
        if len(pos) >= 3:  # x, y, radius
            color = colors.get(zone_name.split('_')[0], 'gray')
            circle = Circle(pos[:2], pos[2], color=color, alpha=alpha, zorder=1)
            ax.add_patch(circle)

def create_trajectory_visualization(trajectories_by_split, probe, save_path=None):
    """Create comprehensive trajectory visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Next-Step Trajectory Predictions Across Generalization Types', fontsize=16, fontweight='bold')
    
    split_names = ['same_rollout', 'same_world', 'different_world']
    split_titles = ['Same Rollout\n(Temporal)', 'Same World\n(Spatial)', 'Different World\n(Environmental)']
    
    for split_idx, split_name in enumerate(split_names):
        trajectories = trajectories_by_split[split_name]
        
        if not trajectories:
            axes[0, split_idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[0, split_idx].transAxes)
            axes[1, split_idx].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[1, split_idx].transAxes)
            continue
        
        # Show up to 2 example trajectories per split
        for traj_idx in range(min(2, len(trajectories))):
            ax = axes[traj_idx, split_idx]
            trajectory = trajectories[traj_idx]
            
            # Get data
            activations = trajectory['activations']
            actual_positions = trajectory['positions']
            actual_next_positions = trajectory['next_positions']
            zone_pos = trajectory.get('zone_pos', {})
            
            # Predict trajectory
            predicted_positions = predict_trajectory_positions(probe, activations, actual_positions[0])
            
            # Draw zone backgrounds
            draw_zone_background(ax, zone_pos, alpha=0.2)
            
            # Plot actual trajectory
            if len(actual_positions) > 1:
                ax.plot(actual_positions[:, 0], actual_positions[:, 1], 
                       'o-', color='blue', linewidth=2, markersize=4, 
                       label='Actual', alpha=0.8, zorder=3)
            
            # Plot predicted trajectory
            if len(predicted_positions) > 1:
                ax.plot(predicted_positions[1:, 0], predicted_positions[1:, 1], 
                       's--', color='red', linewidth=2, markersize=4,
                       label='Predicted', alpha=0.8, zorder=3)
            
            # Mark start and end points
            ax.plot(actual_positions[0, 0], actual_positions[0, 1], 
                   'go', markersize=8, label='Start', zorder=4)
            if len(actual_positions) > 1:
                ax.plot(actual_positions[-1, 0], actual_positions[-1, 1], 
                       'ro', markersize=8, label='End', zorder=4)
            
            # Calculate trajectory prediction error
            if len(predicted_positions) > 1 and len(actual_next_positions) > 0:
                min_len = min(len(predicted_positions)-1, len(actual_next_positions))
                pred_next = predicted_positions[1:min_len+1]
                actual_next = actual_next_positions[:min_len]
                mse = np.mean(np.sum((pred_next - actual_next)**2, axis=1))
                ax.text(0.02, 0.98, f'MSE: {mse:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=10)
            
            # Set title and labels
            if traj_idx == 0:
                ax.set_title(f'{split_titles[split_idx]}\nExample {traj_idx+1}', fontweight='bold')
            else:
                ax.set_title(f'Example {traj_idx+1}')
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Set consistent axis limits
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            
            if traj_idx == 0 and split_idx == 0:
                ax.legend(bbox_to_anchor=(0, -0.1), loc='upper left', ncol=4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {save_path}")
    
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', required=True, help='Layer to probe')
    p.add_argument('--n-rollouts', type=int, default=N_ROLLOUTS, help='Number of trajectories per split type')
    p.add_argument('--max-steps', type=int, default=MAX_STEPS, help='Maximum steps per trajectory')
    p.add_argument('--out', type=str, help='Output file path')
    args = p.parse_args()

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED) 
    torch.manual_seed(SEED)
    
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    print("Loading model...")
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, sampler, sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── collect test trajectories by split type ─────────────────────────────────
    trajectories_by_split = collect_test_trajectories_by_split(
        model, args.layer, sampler, 
        n_rollouts_per_split=args.n_rollouts,
        max_steps=args.max_steps
    )
    
    # Print collection summary
    print("\nTrajectory collection summary:")
    for split_name, trajectories in trajectories_by_split.items():
        print(f"  {split_name}: {len(trajectories)} trajectories")
        if trajectories:
            avg_length = np.mean([len(t['positions']) for t in trajectories])
            print(f"    Average length: {avg_length:.1f} steps")
    
    # ── train probe ─────────────────────────────────────────────────────────────
    if trajectories_by_split['same_rollout']:
        probe = train_next_step_probe(trajectories_by_split, args.layer)
        
        # ── create visualization ──────────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_name = args.layer.replace(".", "_")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = args.out or f'{results_dir}/trajectory_predictions_{layer_name}_{timestamp}.png'
        
        create_trajectory_visualization(trajectories_by_split, probe, save_path=output_file)
        
        print(f"\n✅ Trajectory visualization complete!")
        print(f"   - Shows actual vs predicted trajectories for each generalization type")
        print(f"   - Blue lines: Actual trajectories")  
        print(f"   - Red dashed: Predicted trajectories")
        print(f"   - MSE values show prediction accuracy")
        
    else:
        print("❌ No trajectories collected for training. Cannot create visualization.")

if __name__ == '__main__':
    main() 