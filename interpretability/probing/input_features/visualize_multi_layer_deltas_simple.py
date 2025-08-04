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

def collect_multi_layer_trajectory(model, layer_names, sampler, world_id, rollout_seed, max_steps=40):
    """Collect a single trajectory with activations from multiple layers."""
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    
    # Get modules for all layers
    modules = {layer_name: dict(model.named_modules())[layer_name] for layer_name in layer_names}

    # Skip world file loading to allow proper seeding/randomization
    # world_file = f"{WORLD_DIR}/world_info_{world_id}.pkl"
    # env.load_world_info(world_file)
    zone_pos = dict(env.zone_positions) if hasattr(env, 'zone_positions') else {}
    
    obs = env.reset(seed=rollout_seed)
    agent.reset()
    
    # Collect activations for all layers
    layer_activations = {layer_name: [] for layer_name in layer_names}
    
    def create_hook(layer_name):
        def grab(m, inp, out):
            x = out[1] if isinstance(out, tuple) else out
            layer_activations[layer_name].append(x.detach().cpu().numpy().ravel())
        return grab
    
    # Register hooks for all layers
    hooks = {}
    for layer_name in layer_names:
        hooks[layer_name] = modules[layer_name].register_forward_hook(create_hook(layer_name))
    
    positions = []
    
    done = False
    for step in range(max_steps):
        if done:
            break
            
        current_pos = env.agent_pos[:2].copy()
        positions.append(current_pos.copy())
        
        a = agent.get_action(obs, {}, deterministic=True).flatten()
        obs, _, done, _ = env.step(a)
    
    # Remove hooks
    for hook in hooks.values():
        hook.remove()
    
    env.close()
    
    return {
        'positions': np.array(positions),
        'layer_activations': {layer_name: np.array(activations) for layer_name, activations in layer_activations.items()},
        'zone_pos': zone_pos
    }

def predict_trajectory_from_deltas(probe, start_pos, activations):
    """Predict trajectory by accumulating predicted deltas."""
    predicted_positions = [start_pos.copy()]
    current_pos = start_pos.copy()
    
    for i in range(len(activations) - 1):
        activation = activations[i + 1].reshape(1, -1)
        predicted_delta = probe.predict(activation)[0]
        current_pos = current_pos + predicted_delta
        predicted_positions.append(current_pos.copy())
    
    return np.array(predicted_positions)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', nargs='+', required=True, help='Neural network layers to probe')
    p.add_argument('--world-id', type=int, default=5, help='World ID to test on')
    p.add_argument('--seed-offset', type=int, default=3000, help='Seed offset for trajectory generation')
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

    # Train probes for each layer
    train_worlds = [0, 1, 2, 3, 4]
    probes = {}
    
    print(f"Training delta prediction probes for {len(args.layers)} layers...")
    for layer_name in args.layers:
        print(f"  Training probe for {layer_name}...")
        X_train, y_train = collect_probe_training_data(
            model, layer_name, sampler, train_worlds, n_rollouts=3, max_steps=50)
        
        if len(X_train) == 0:
            print(f"    No training data for {layer_name}")
            continue
        
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        probes[layer_name] = probe
        print(f"    Trained on {len(X_train)} samples")
    
    if not probes:
        print("No probes could be trained!")
        return

    # Define colors for different layers
    layer_colors = [
        '#e74c3c',  # Red
        '#3498db',  # Blue  
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Teal
    ]
    
    layer_color_map = {layer: layer_colors[i % len(layer_colors)] 
                       for i, layer in enumerate(args.layers)}
    
    # Collect test trajectory
    seed = SEED + args.seed_offset
    traj_data = collect_multi_layer_trajectory(
        model, args.layers, sampler, args.world_id, seed, max_steps=35)
    
    if len(traj_data['positions']) < 3:
        print("Insufficient trajectory data!")
        return
    
    # Create single plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, axes_class=FancyAxes, edgecolor='gray', linewidth=0.5)
    setup_axis(ax)
    
    # Draw zone map
    draw_zones(ax, traj_data['zone_pos'])
    
    # Draw starting position
    draw_diamond(ax, traj_data['positions'][0], color='orange', size=0.15)
    
    # Draw actual trajectory (thick black line)
    draw_path(ax, traj_data['positions'], color='black', linewidth=4, style='solid')
    
    # Draw predicted trajectories for each layer
    layer_errors = {}
    for layer_name in args.layers:
        if layer_name not in probes:
            continue
            
        layer_activations = traj_data['layer_activations'][layer_name]
        if len(layer_activations) < 2:
            continue
        
        # Predict trajectory using this layer's probe
        predicted_positions = predict_trajectory_from_deltas(
            probes[layer_name], traj_data['positions'][0], layer_activations)
        
        # Draw predicted trajectory
        color = layer_color_map[layer_name]
        if len(predicted_positions) > 1:
            draw_path(ax, predicted_positions, color=color, linewidth=3, style='dashed')
        
        # Calculate final error
        if len(predicted_positions) == len(traj_data['positions']):
            final_error = np.linalg.norm(predicted_positions[-1] - traj_data['positions'][-1])
            layer_errors[layer_name] = final_error
    
    # Add error text box
    if layer_errors:
        error_text = "Final Prediction Errors:\n" + "\n".join([
            f"{layer.split('.')[-1]}: {error:.2f}" 
            for layer, error in sorted(layer_errors.items())
        ])
        ax.text(0.02, 0.98, error_text, transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    
    ax.set_title(f'Multi-Layer Delta Predictions - World {args.world_id}', fontsize=14)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=4, label='Actual Trajectory'),
        Line2D([0], [0], marker='D', color='orange', linewidth=0, markersize=8, label='Start Position')
    ]
    
    # Add layer-specific legend entries
    for layer_name in args.layers:
        if layer_name in probes:
            color = layer_color_map[layer_name]
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=3, linestyle='--', 
                       label=f'{short_name} Prediction')
            )
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    layer_names_str = "_".join([layer.replace(".", "_") for layer in args.layers])
    plot_file = f'{results_dir}/multi_layer_comparison_world{args.world_id}_{layer_names_str}_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved multi-layer comparison to {plot_file}")
    plt.show()

if __name__ == '__main__':
    main() 