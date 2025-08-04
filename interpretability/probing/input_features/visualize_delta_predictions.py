#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
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

def collect_probe_training_data(model, layer_name, sampler, world_ids, n_rollouts=5, max_steps=100):
    """
    Collect training data for delta prediction probes.
    """
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    all_activations = []
    all_deltas = []
    all_world_info = []

    for world_id in world_ids:
        # Skip world file loading to allow proper seeding/randomization
        # world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
        # if not os.path.exists(world_file):
        #     continue
        # env.load_world_info(world_file)
        print(f"Processing world_id {world_id} with random world generation")
        zone_pos = dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
        
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
                
                # Calculate delta if we have previous position
                if prev_pos is not None:
                    delta = current_pos - prev_pos
                    rollout_deltas.append(delta)
                
                # Take action
                a = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(a)
                
                prev_pos = current_pos.copy()
            
            h.remove()
            
            # Align activations with deltas
            if len(step_activations) > 1 and len(rollout_deltas) > 0:
                min_len = min(len(step_activations) - 1, len(rollout_deltas))
                all_activations.extend(step_activations[1:min_len+1])  # Skip first activation
                all_deltas.extend(rollout_deltas[:min_len])
                all_world_info.extend([(world_id, rollout_idx)] * min_len)
    
    env.close()
    return np.array(all_activations), np.array(all_deltas), all_world_info

def collect_test_trajectory(model, layer_name, sampler, world_id, rollout_seed, max_steps=50):
    """
    Collect a single trajectory for visualization with activations and positions.
    """
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
        
        # Take action
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

def predict_trajectory_from_deltas(probe, start_pos, activations):
    """
    Predict trajectory by accumulating predicted deltas.
    """
    predicted_positions = [start_pos.copy()]
    current_pos = start_pos.copy()
    
    for i in range(len(activations) - 1):  # Skip first activation (no previous position)
        activation = activations[i + 1].reshape(1, -1)
        predicted_delta = probe.predict(activation)[0]
        current_pos = current_pos + predicted_delta
        predicted_positions.append(current_pos.copy())
    
    return np.array(predicted_positions)

def visualize_delta_predictions(model, layer_name, sampler, args):
    """
    Main visualization function.
    """
    # Train probe on training worlds
    train_worlds = [0, 1, 2, 3, 4]  # Same as in our comprehensive script
    test_worlds = [5, 6, 7, 8, 9]
    
    print("Training delta prediction probe...")
    X_train, y_train, _ = collect_probe_training_data(
        model, layer_name, sampler, train_worlds, n_rollouts=3, max_steps=50)
    
    if len(X_train) == 0:
        print("No training data collected!")
        return
    
    # Train probe
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    print(f"Trained probe on {len(X_train)} samples")
    
    # Create visualizations for different test scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Delta Prediction Visualization - {layer_name}', fontsize=16)
    
    scenarios = [
        ("Same World (Train)", train_worlds[0], SEED + 1000),
        ("Same World (Test)", train_worlds[0], SEED + 2000), 
        ("Different World", test_worlds[0], SEED + 3000),
        ("Different World", test_worlds[1], SEED + 4000),
        ("Different World", test_worlds[2], SEED + 5000),
        ("Different World", test_worlds[3], SEED + 6000),
    ]
    
    for idx, (scenario_name, world_id, seed) in enumerate(scenarios):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        try:
            # Collect test trajectory
            traj_data = collect_test_trajectory(model, layer_name, sampler, world_id, seed, max_steps=30)
            
            if len(traj_data['positions']) < 3:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{scenario_name}\nWorld {world_id}')
                continue
            
            # Predict trajectory using probe
            predicted_positions = predict_trajectory_from_deltas(
                probe, traj_data['positions'][0], traj_data['activations'])
            
            # Set up the plot
            ax = plt.subplot(2, 3, idx + 1, axes_class=FancyAxes, edgecolor='gray', linewidth=0.5)
            setup_axis(ax)
            
            # Draw zone map
            draw_zones(ax, traj_data['zone_pos'])
            
            # Draw starting position
            draw_diamond(ax, traj_data['positions'][0], color='orange', size=0.15)
            
            # Draw actual trajectory
            draw_path(ax, traj_data['positions'], color='blue', linewidth=3, style='solid')
            
            # Draw predicted trajectory (only if we have enough predictions)
            if len(predicted_positions) > 1:
                draw_path(ax, predicted_positions, color='red', linewidth=2, style='dashed')
            
            # Calculate and display error
            if len(predicted_positions) == len(traj_data['positions']):
                final_error = np.linalg.norm(predicted_positions[-1] - traj_data['positions'][-1])
                ax.text(0.02, 0.98, f'Final Error: {final_error:.3f}', 
                       transform=ax.transAxes, va='top', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(f'{scenario_name}\nWorld {world_id}', fontsize=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{scenario_name}\nWorld {world_id}')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=3, label='Actual Trajectory'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Predicted Trajectory'),
        Line2D([0], [0], marker='D', color='orange', linewidth=0, markersize=8, label='Start Position')
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save the plot
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_file = f'{results_dir}/delta_prediction_trajectories_{layer_name.replace(".", "_")}_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory visualization to {plot_file}")
    plt.show()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layer', required=True, help='Neural network layer to probe')
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

    # Create visualizations
    visualize_delta_predictions(model, args.layer, sampler, args)

if __name__ == '__main__':
    main() 