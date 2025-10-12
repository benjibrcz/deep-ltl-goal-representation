#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from visualize.zones      import draw_zones, draw_path, draw_diamond, setup_axis, FancyAxes

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

def collect_probe_training_data(model, layer_name, sampler, world_ids, n_rollouts=3, max_steps=50):
    """Collect training data for delta prediction probes."""
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    all_activations = []
    all_deltas = []

    for world_id in world_ids:
        # Skip world file loading to allow proper seeding/randomization
        # world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
        # if not os.path.exists(world_file):
        #     continue
        # env.load_world_info(world_file)
        print(f"Processing world_id {world_id} with random world generation")
        
        for rollout_idx in range(n_rollouts):
            try:
                obs = env.reset(seed=SEED + world_id * 1000 + rollout_idx * 100)
            except:
                continue
                
            agent.reset()
            
            step_activations = []
            def grab(m, inp, out):
                x = out[1] if isinstance(out, tuple) else out
                step_activations.append(x.detach().cpu().numpy().ravel())
            
            h = module.register_forward_hook(grab)
            
            prev_pos = None
            rollout_deltas = []
            
            done = False
            for step in range(max_steps):
                if done:
                    break
                    
                current_pos = env.agent_pos[:2].copy()
                
                if prev_pos is not None:
                    delta = current_pos - prev_pos
                    rollout_deltas.append(delta)
                
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                prev_pos = current_pos.copy()
            
            h.remove()
            
            # Align activations with deltas
            if len(step_activations) > 1 and len(rollout_deltas) > 0:
                min_len = min(len(step_activations) - 1, len(rollout_deltas))
                all_activations.extend(step_activations[1:min_len+1])
                all_deltas.extend(rollout_deltas[:min_len])
    
    env.close()
    return np.array(all_activations), np.array(all_deltas)

def collect_detailed_trajectory(model, layer_name, sampler, world_id, rollout_seed, max_steps=30):
    """Collect detailed trajectory with step-by-step activations, positions, and deltas."""
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    # Skip world file loading to allow proper seeding/randomization
    # world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
    # env.load_world_info(world_file)
    zone_pos = dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
    
    obs = env.reset(seed=rollout_seed)
    agent.reset()
    
    step_activations = []
    def grab(m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        step_activations.append(x.detach().cpu().numpy().ravel())
    
    h = module.register_forward_hook(grab)
    
    positions = []
    actual_deltas = []
    prev_pos = None
    
    done = False
    for step in range(max_steps):
        if done:
            break
            
        current_pos = env.agent_pos[:2].copy()
        positions.append(current_pos.copy())
        
        # Calculate actual delta if we have previous position
        if prev_pos is not None:
            delta = current_pos - prev_pos
            actual_deltas.append(delta)
        
        a = agent.get_action(obs, {}, deterministic=True).flatten()
        obs, _, done, _ = env.step(a)
        prev_pos = current_pos.copy()
    
    h.remove()
    env.close()
    
    return {
        'positions': np.array(positions),
        'actual_deltas': np.array(actual_deltas),
        'activations': np.array(step_activations),
        'zone_pos': zone_pos
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', required=True, help='Neural network layer to probe')
    p.add_argument('--world-id', type=int, default=0, help='World ID for same-rollout test (use training world)')
    p.add_argument('--seed-offset', type=int, default=5000, help='Seed offset for trajectory generation')
    args = p.parse_args()

    # Setup
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = "FG blue"
    sampler = FixedSampler.partial(formula)

    # Load model
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, sampler, sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()

    # Train probe (using one world)
    train_worlds = [args.world_id]  # Train world
    
    print(f"Training delta prediction probe for {args.layer}...")
    X_train, y_train = collect_probe_training_data(
        model, args.layer, sampler, train_worlds, n_rollouts=5, max_steps=200)
    
    if len(X_train) == 0:
        print("No training data collected!")
        return
    
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    print(f"Trained probe on {len(X_train)} samples from world {args.world_id}")
    
    # Collect ONE long trajectory and split it temporally (same rollout scenario)
    trajectory_seed = SEED + args.world_id * 1000 + 200
    full_traj_data = collect_detailed_trajectory(
        model, args.layer, sampler, args.world_id, trajectory_seed, max_steps=200)
    
    # Split the trajectory temporally: first 50% for "train", last 50% for "test" 
    # This ensures both portions have diverse movement patterns
    total_steps = len(full_traj_data['positions'])
    split_point = int(0.5 * total_steps)
    
    print(f"Splitting trajectory at step {split_point} (total steps: {total_steps})")
    
    # Create train portion (first part of the trajectory)
    train_traj_data = {
        'positions': full_traj_data['positions'][:split_point],
        'actual_deltas': full_traj_data['actual_deltas'][:split_point-1] if len(full_traj_data['actual_deltas']) >= split_point-1 else full_traj_data['actual_deltas'],
        'activations': full_traj_data['activations'][:split_point],
        'zone_pos': full_traj_data['zone_pos']
    }
    
    # Create test portion (last part of the trajectory)
    test_traj_data = {
        'positions': full_traj_data['positions'][split_point:],
        'actual_deltas': full_traj_data['actual_deltas'][split_point-1:] if len(full_traj_data['actual_deltas']) > split_point-1 else [],
        'activations': full_traj_data['activations'][split_point:],
        'zone_pos': full_traj_data['zone_pos']
    }
    
    if len(test_traj_data['positions']) < 3 or len(train_traj_data['positions']) < 3:
        print("Insufficient trajectory data!")
        return
    
    # Analyze movement characteristics
    def analyze_movement_diversity(deltas, name):
        if len(deltas) == 0:
            return
        speeds = np.linalg.norm(deltas, axis=1)
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angle_changes = np.abs(np.diff(angles))
        # Handle angle wraparound
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        
        print(f"{name} movement analysis:")
        print(f"  Steps: {len(deltas)}")
        print(f"  Speed range: {speeds.min():.3f} - {speeds.max():.3f} (mean: {speeds.mean():.3f})")
        print(f"  Total distance: {speeds.sum():.2f}")
        print(f"  Direction changes (mean): {angle_changes.mean():.3f} rad ({np.degrees(angle_changes.mean()):.1f}°)")
        print(f"  Max direction change: {angle_changes.max():.3f} rad ({np.degrees(angle_changes.max()):.1f}°)")
        print(f"  Position range: x=[{(deltas.cumsum(axis=0) + full_traj_data['positions'][0])[:, 0].min():.2f}, {(deltas.cumsum(axis=0) + full_traj_data['positions'][0])[:, 0].max():.2f}], y=[{(deltas.cumsum(axis=0) + full_traj_data['positions'][0])[:, 1].min():.2f}, {(deltas.cumsum(axis=0) + full_traj_data['positions'][0])[:, 1].max():.2f}]")
    
    if len(train_traj_data['actual_deltas']) > 0:
        analyze_movement_diversity(train_traj_data['actual_deltas'], "Train")
    
    if len(test_traj_data['actual_deltas']) > 0:
        analyze_movement_diversity(test_traj_data['actual_deltas'], "Test")
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot training predictions (left)
    ax1 = plt.subplot(1, 2, 1, axes_class=FancyAxes, edgecolor='gray', linewidth=0.5)
    setup_axis(ax1)
    plot_trajectory_with_deltas(ax1, train_traj_data, probe, "TRAIN (early steps)", args.layer, args.world_id)
    
    # Plot test predictions (right)  
    ax2 = plt.subplot(1, 2, 2, axes_class=FancyAxes, edgecolor='gray', linewidth=0.5)
    setup_axis(ax2)
    plot_trajectory_with_deltas(ax2, test_traj_data, probe, "TEST (later steps)", args.layer, args.world_id)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    layer_name_str = args.layer.replace(".", "_")
    plot_file = f'{results_dir}/train_test_step_deltas_world{args.world_id}_{layer_name_str}_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved train/test step-by-step delta visualization to {plot_file}")
    plt.show()

def plot_trajectory_with_deltas(ax, traj_data, probe, data_type, layer_name, world_id):
    """Plot a single trajectory with step-by-step delta predictions."""
    
    # Draw zone map
    draw_zones(ax, traj_data['zone_pos'])
    
    # Draw actual trajectory (thick blue line with markers)
    draw_path(ax, traj_data['positions'], color='blue', linewidth=3, style='solid')
    
    # Also draw trajectory points explicitly
    ax.plot(traj_data['positions'][:, 0], traj_data['positions'][:, 1], 'bo', markersize=4, alpha=0.6)
    
    # Draw starting position (larger)
    draw_diamond(ax, traj_data['positions'][0], color='orange', size=0.2)
    
    # Draw ending position
    ax.plot(traj_data['positions'][-1, 0], traj_data['positions'][-1, 1], 'rs', markersize=10, label='End')
    
    # Draw step-by-step delta predictions
    positions = traj_data['positions']
    actual_deltas = traj_data['actual_deltas']
    activations = traj_data['activations']
    
    # Show every 10th step
    step_interval = 20
    
    print(f"Drawing {data_type} delta arrows for {len(actual_deltas)} steps...")
    
    for i in range(0, len(actual_deltas), step_interval):
        if i + 1 >= len(activations):
            break
            
        # Current position and actual delta
        current_pos = positions[i]
        actual_delta = actual_deltas[i]
        actual_next_pos = positions[i + 1]
        
        # Predict delta using activation from step i+1
        activation = activations[i + 1].reshape(1, -1)
        predicted_delta = probe.predict(activation)[0]
        predicted_next_pos = current_pos + predicted_delta
        
        # Scale up arrows for visibility if movement is very small
        scale_factor = max(10.0, 0.3 / max(np.linalg.norm(actual_delta), 0.001))
        scale_factor = min(scale_factor, 20.0)  # Cap the scaling
        
        # Draw actual delta arrow (green) - scaled for visibility
        scaled_actual_end = current_pos + actual_delta * scale_factor
        ax.annotate('', xy=scaled_actual_end, xytext=current_pos,
                   arrowprops=dict(arrowstyle='->', color='green', lw=3, alpha=0.8))
        
        # Draw predicted delta arrow (red) - scaled for visibility
        scaled_pred_end = current_pos + predicted_delta * scale_factor
        ax.annotate('', xy=scaled_pred_end, xytext=current_pos,
                   arrowprops=dict(arrowstyle='->', color='red', lw=5, alpha=0.7))
        
        # Add step number
        ax.text(current_pos[0], current_pos[1], str(i), fontsize=8, 
               ha='center', va='center', 
               bbox=dict(boxstyle="circle,pad=0.1", facecolor="yellow", alpha=0.8))
    
    # Add all step errors for statistics (not just the displayed ones)
    all_step_errors = []
    for i in range(len(actual_deltas)):
        if i + 1 >= len(activations):
            break
        activation = activations[i + 1].reshape(1, -1)
        predicted_delta = probe.predict(activation)[0]
        actual_delta = actual_deltas[i]
        step_error = np.linalg.norm(predicted_delta - actual_delta)
        all_step_errors.append(step_error)
    
    # Add error statistics
    if all_step_errors:
        mean_error = np.mean(all_step_errors)
        max_error = np.max(all_step_errors)
        error_text = f"{data_type} Delta Errors:\nMean: {mean_error:.3f}\nMax: {max_error:.3f}\nTotal Steps: {len(all_step_errors)}\nShowing every {step_interval} steps\n(Arrows scaled {scale_factor:.1f}x)"
        ax.text(0.02, 0.98, error_text, transform=ax.transAxes, va='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    
    ax.set_title(f'{data_type} Predictions - {layer_name}\nWorld {world_id}', fontsize=12)
    
    # Create legend (only for the first plot to avoid duplication)
    if data_type == "TRAIN":
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3, label='Actual Trajectory'),
            Line2D([0], [0], marker='D', color='orange', linewidth=0, markersize=8, label='Start Position'),
            Line2D([0], [0], color='green', linewidth=3, label='Actual Delta'),
            Line2D([0], [0], color='red', linewidth=2.5, label='Predicted Delta'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85))

if __name__ == '__main__':
    main() 