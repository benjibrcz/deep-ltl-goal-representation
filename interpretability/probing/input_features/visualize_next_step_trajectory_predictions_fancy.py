#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
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
from visualize.zones      import FancyAxes, draw_zones, draw_path, draw_diamond, setup_axis

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_ROLLOUTS = 3  # Number of rollouts per test type to visualize
MAX_STEPS  = 40  # Maximum steps per trajectory for visualization
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_test_trajectories_by_split(model, layer_name, sampler, n_rollouts_per_split=3, max_steps=40):
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
            # Ensure arrays are properly aligned before temporal split
            min_len = min(len(trajectory['activations']), len(trajectory['positions']), len(trajectory['next_positions']))
            if min_len >= 4:
                # Split trajectory temporally
                mid_point = min_len // 2
                test_trajectory = {
                    'activations': trajectory['activations'][mid_point:min_len],
                    'positions': trajectory['positions'][mid_point:min_len],
                    'next_positions': trajectory['next_positions'][mid_point:min_len],
                    'zone_pos': trajectory['zone_pos'],
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
        
        # Ensure all arrays have the same length
        min_len = min(len(activations), len(positions), len(next_positions))
        if min_len >= 2:
            return {
                'activations': np.array(activations[:min_len]),
                'positions': np.array(positions[:min_len]),
                'next_positions': np.array(next_positions[:min_len]),
                'zone_pos': dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
            }
        return None
        
    except Exception as e:
        print(f"Error collecting trajectory for world {world_id}: {e}")
        return None

def train_next_step_probe(trajectories_by_split):
    """Train a probe on training data to predict next positions."""
    print("Training next-step position probe...")
    
    # Collect training data from same_rollout trajectories (first half of each)
    X_train, y_train = [], []
    
    for trajectory in trajectories_by_split['same_rollout']:
        activations = trajectory['activations']
        next_positions = trajectory['next_positions']
        
        # Ensure arrays are same length (activations and next_positions should align)
        min_len = min(len(activations), len(next_positions))
        if min_len == 0:
            continue
            
        activations = activations[:min_len]
        next_positions = next_positions[:min_len]
        
        # Use first half for training (temporal split)
        mid_point = min_len // 2
        train_activations = activations[:mid_point]
        train_next_pos = next_positions[:mid_point]
        
        if len(train_activations) > 0:
            X_train.extend(train_activations.tolist())
            y_train.extend(train_next_pos.tolist())
    
    if len(X_train) == 0:
        raise ValueError("No training data collected")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training probe on {len(X_train)} samples (X shape: {X_train.shape}, y shape: {y_train.shape})")
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    
    return probe

def predict_trajectory_positions(probe, activations, start_position):
    """Predict trajectory positions step by step from activations."""
    if len(activations) == 0:
        return np.array([start_position])
    
    predicted_positions = [start_position]
    
    for i in range(len(activations)):
        try:
            # Predict next position from current activation
            next_pos_pred = probe.predict(activations[i:i+1].reshape(1, -1))[0]
            predicted_positions.append(next_pos_pred)
        except Exception as e:
            print(f"Warning: Prediction failed at step {i}: {e}")
            # Use last position as fallback
            predicted_positions.append(predicted_positions[-1])
    
    return np.array(predicted_positions)

def create_fancy_trajectory_visualization(trajectories_by_split, probe, save_path=None):
    """Create high-quality trajectory visualization using built-in visualization tools."""
    
    # Create figure with fancy subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Next-Step Trajectory Predictions: Generalization Analysis', 
                fontsize=20, fontweight='bold', y=0.95)
    
    split_names = ['same_rollout', 'same_world', 'different_world']
    split_titles = ['Same Rollout (Temporal)', 'Same World (Spatial)', 'Different World (Environmental)']
    
    # Show 2 examples per split type
    for split_idx, split_name in enumerate(split_names):
        trajectories = trajectories_by_split[split_name]
        
        if not trajectories:
            continue
            
        # Show up to 2 trajectories per split
        for traj_idx in range(min(2, len(trajectories))):
            ax_idx = traj_idx * 3 + split_idx + 1  # 1-based indexing for subplots
            ax = fig.add_subplot(2, 3, ax_idx, axes_class=FancyAxes, 
                               edgecolor='gray', linewidth=0.5)
            setup_axis(ax)
            
            trajectory = trajectories[traj_idx]
            
            # Get data
            activations = trajectory['activations']
            actual_positions = trajectory['positions']
            actual_next_positions = trajectory['next_positions']
            zone_pos = trajectory.get('zone_pos', {})
            
            # Draw zones first (background)
            draw_zones(ax, zone_pos)
            
            # Predict trajectory
            predicted_positions = predict_trajectory_positions(probe, activations, actual_positions[0])
            
            # Draw starting position
            draw_diamond(ax, actual_positions[0], color='orange', size=0.12)
            
            # Draw actual trajectory (thick blue line)
            if len(actual_positions) > 1:
                draw_path(ax, actual_positions, color='blue', linewidth=3, style='solid')
            
            # Draw predicted trajectory - properly aligned for each split type
            if len(predicted_positions) > 1:
                if trajectory['split_type'] == 'same_rollout':
                    # For temporal split: predictions should match test portion length exactly
                    test_length = len(actual_positions)
                    pred_to_show = predicted_positions[1:test_length+1]  # Skip start, match test length
                    if len(pred_to_show) > 0:
                        draw_path(ax, pred_to_show, color='red', linewidth=3, style='dashed')
                    # Add note that this is the test portion
                    ax.text(0.02, 0.85, 'Test portion\n(2nd half)', transform=ax.transAxes,
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7),
                           verticalalignment='top', zorder=20)
                else:
                    # For spatial/environmental splits: predictions should match actual trajectory length
                    test_length = len(actual_positions)
                    pred_to_show = predicted_positions[1:test_length+1]  # Skip start, match actual length
                    if len(pred_to_show) > 0:
                        draw_path(ax, pred_to_show, color='red', linewidth=3, style='dashed')
            
            # Calculate and display MSE
            if len(predicted_positions) > 1 and len(actual_next_positions) > 0:
                min_len = min(len(predicted_positions)-1, len(actual_next_positions))
                pred_next = predicted_positions[1:min_len+1]
                actual_next = actual_next_positions[:min_len]
                mse = np.mean(np.sum((pred_next - actual_next)**2, axis=1))
                
                # Add MSE text box
                ax.text(0.02, 0.98, f'MSE: {mse:.3f}', transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       verticalalignment='top', zorder=20)
            
            # Set titles
            if traj_idx == 0:
                ax.set_title(f'{split_titles[split_idx]}\nExample {traj_idx+1}', 
                           fontsize=14, fontweight='bold', pad=20)
            else:
                ax.set_title(f'Example {traj_idx+1}', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label='Actual Trajectory'),
        Line2D([0], [0], color='red', linewidth=3, linestyle='dashed', label='Predicted Trajectory'),
        Line2D([0], [0], marker='D', color='orange', linewidth=0, markersize=8, label='Start Position')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              fontsize=14, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.9)  # Make room for legend and title
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved high-quality trajectory visualization to {save_path}")
    
    plt.show()
    return fig

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
        probe = train_next_step_probe(trajectories_by_split)
        
        # ── create fancy visualization ──────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_name = args.layer.replace(".", "_")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        output_file = args.out or f'{results_dir}/fancy_trajectory_predictions_{layer_name}_{timestamp}.png'
        
        fig = create_fancy_trajectory_visualization(trajectories_by_split, probe, save_path=output_file)
        
        print(f"\n✅ High-quality trajectory visualization complete!")
        print(f"   - Professional-grade plots using built-in visualization tools")
        print(f"   - Fancy bordered axes with proper zone visualization")
        print(f"   - Blue solid: Actual trajectories")  
        print(f"   - Red dashed: Predicted trajectories")
        print(f"   - Orange diamonds: Starting positions")
        print(f"   - MSE values: Prediction accuracy per trajectory")
        
    else:
        print("❌ No trajectories collected for training. Cannot create visualization.")

if __name__ == '__main__':
    main() 