#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

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
N_WORLDS   = 50
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

CMAP       = {"blue":0, "green":1, "yellow":2, "pink":3}
CMAP_RGB   = {
    0: "#1f77b4",   # blue
    1: "#2ca02c",   # green
    2: "#ff7f0e",   # yellow
    3: "#d62728",   # red
    4: "#9467bd",   # purple
    5: "#8c564b",   # brown
    6: "#e377c2",   # pink
}
COLOR_NAMES= ["blue","green","yellow","red", "purple", "brown", "pink"]

def get_current_goal(agent, step):
    """Get the current goal the agent is pursuing at a given step"""
    if hasattr(agent, 'sequence') and agent.sequence and len(agent.sequence) > 0:
        goal_set = agent.sequence[0][0]
        if len(goal_set) == 1:
            assignment = next(iter(goal_set))
            true_props = {p for p, v in assignment.assignment if v}
            if len(true_props) == 1:
                prop = next(iter(true_props))
                if prop in CMAP:
                    return CMAP[prop]
    return None

def calculate_directional_accuracy(pred_pos, true_pos, agent_pos):
    """Calculate how well the predicted direction matches the true direction"""
    # Calculate true direction vector
    true_dir = true_pos - agent_pos
    true_dir_norm = np.linalg.norm(true_dir)
    if true_dir_norm == 0:
        return 0.0, 0.0, 0.0
    
    true_dir_unit = true_dir / true_dir_norm
    
    # Calculate predicted direction vector
    pred_dir = pred_pos - agent_pos
    pred_dir_norm = np.linalg.norm(pred_dir)
    if pred_dir_norm == 0:
        return 0.0, 0.0, 0.0
    
    pred_dir_unit = pred_dir / pred_dir_norm
    
    # Calculate cosine similarity (dot product of unit vectors)
    cosine_sim = np.dot(true_dir_unit, pred_dir_unit)
    
    # Calculate angular error in degrees
    angular_error = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) * 180 / np.pi
    
    # Calculate distance error
    distance_error = abs(pred_dir_norm - true_dir_norm)
    
    return cosine_sim, angular_error, distance_error

def collect_data_for_layer(model, layer_name, sampler):
    X, Ypos, Ycol = [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    world_dir_path = f"{WORLD_DIR}"
    if not os.path.exists(world_dir_path):
        print(f"World directory not found: {world_dir_path}, skipping data collection.")
        env.close()
        return None, None, None

    for i in trange(N_WORLDS, desc="Collect worlds"):
        world_file = f"{world_dir_path}/world_info_{i}.pkl"
        if not os.path.exists(world_file):
            continue
        env.load_world_info(world_file)
        obs = env.reset(seed=SEED+i)
        agent.reset()

        feats = []
        def grab(m, inp, out):
            x = out[1] if isinstance(out, tuple) else out
            feats.append(x.detach().cpu().numpy().ravel())
        h = module.register_forward_hook(grab)
        try: agent.get_action(obs, {}, deterministic=True)
        except: pass
        h.remove()

        X.append(feats[0])

        layout = env.task.world_info.layout
        zs, cs, keys = [], [], []
        for k,v in layout.items():
            if 'zone' in k:
                for name,idx in CMAP.items():
                    if name in k:
                        zs.append(np.asarray(v, float).tolist())
                        cs.append(idx); keys.append(k)
                        break
        order = sorted(range(len(zs)), key=lambda j: keys[j])
        if not order: # Handle case with no zones
            Ypos.append(np.array([]))
            Ycol.append([])
        else:
            Ypos.append(np.array([zs[j] for j in order]).flatten())
            Ycol.append([cs[j] for j in order])

    env.close()

    if not X:
        return None, None, None

    return np.stack(X), np.stack(Ypos), np.stack(Ycol)

def decode_true_layout(env):
    layout = env.task.world_info.layout
    zs, cs, keys = [], [], []
    for k,v in layout.items():
        if 'zone' in k:
            for name,idx in CMAP.items():
                if name in k:
                    zs.append(np.asarray(v, float).tolist())
                    cs.append(idx); keys.append(k)
                    break
    order = sorted(range(len(zs)), key=lambda j: keys[j])
    return np.array([zs[j] for j in order]), [cs[j] for j in order]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--formula',     type=str, default="GF blue & GF green")
    p.add_argument('--layer',       required=True)
    p.add_argument('--world-idx',   type=int, default=0)
    p.add_argument('--max-steps',   type=int, default=1000)
    p.add_argument('--out',         type=str, default='compass_accuracy.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = args.formula
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── train probes ─────────────────────────────────────────────────────────────
    X, Ypos, Ycol = collect_data_for_layer(model, args.layer, sampler)
    if X is None:
        print("Could not collect data for probe training. Exiting.")
        return

    ridge = Ridge().fit(X, Ypos)
    clfs  = []
    if Ycol is not None and Ycol.shape[1] > 0:
        for z in range(Ycol.shape[1]):
            y = Ycol[:,z]
            clf = None
            if len(np.unique(y))>1:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(multi_class="multinomial", max_iter=500)
                clf.fit(X, y)
            clfs.append(clf)

    # ── rollout and collect accuracy data ───────────────────────────────────────
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    env.load_world_info(f"{WORLD_DIR}/world_info_{args.world_idx}.pkl")
    obs = env.reset(seed=SEED+args.world_idx)
    true_zpos, true_zcol = decode_true_layout(env)

    props  = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)

    feats, traj, goals = [], [], []
    module = dict(model.named_modules())[args.layer]
    def grab(m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        feats.append(x.detach().cpu().numpy().ravel())
    h = module.register_forward_hook(grab)

    agent.reset()
    traj.append(env.agent_pos[:2].copy())
    goals.append(get_current_goal(agent, 0))
    done=False
    
    for step in range(args.max_steps):
        try:
            if done:
                a = env.action_space.sample()
            else:
                a = agent.get_action(obs, {}, deterministic=True).flatten()
        except ValueError as e:
            print(f"Error getting action at step {step}: {e}. Using random action.")
            a = env.action_space.sample()

        obs, _, done, _ = env.step(a)
        traj.append(env.agent_pos[:2].copy())
        goals.append(get_current_goal(agent, step + 1))

        if done and step > 100:  # Ensure we have enough data
            break
            
    h.remove()
    env.close()

    feats = np.stack(feats)
    traj  = np.stack(traj)
    goals = np.array(goals)

    # ── analyze directional accuracy ────────────────────────────────────────────
    print(f"\nAnalyzing directional accuracy for {len(feats)} timesteps...")
    
    # Ensure arrays have the same length
    min_len = min(len(feats), len(traj))
    feats = feats[:min_len]
    traj = traj[:min_len]
    goals = goals[:min_len]
    
    # Collect accuracy metrics for each zone and timestep
    accuracy_data = {
        'cosine_similarities': [],
        'angular_errors': [],
        'distance_errors': [],
        'timesteps': [],
        'zone_colors': [],
        'is_current_goal': []
    }
    
    for t in range(len(feats)):
        x = feats[t:t+1]
        ppos = ridge.predict(x).reshape(-1,2)
        agent_pos = traj[t]
        current_goal = goals[t]
        
        for z_idx, ((true_zx, true_zy), true_c_idx) in enumerate(zip(true_zpos, true_zcol)):
            pred_zx, pred_zy = ppos[z_idx]
            
            # Calculate directional accuracy
            cosine_sim, angular_error, distance_error = calculate_directional_accuracy(
                np.array([pred_zx, pred_zy]), 
                np.array([true_zx, true_zy]), 
                agent_pos
            )
            
            accuracy_data['cosine_similarities'].append(cosine_sim)
            accuracy_data['angular_errors'].append(angular_error)
            accuracy_data['distance_errors'].append(distance_error)
            accuracy_data['timesteps'].append(t)
            accuracy_data['zone_colors'].append(COLOR_NAMES[true_c_idx])
            accuracy_data['is_current_goal'].append(current_goal == true_c_idx)

    # ── print summary statistics ────────────────────────────────────────────────
    print("\n=== COMPASS PHENOMENON QUANTIFICATION ===")
    
    # Overall statistics
    cosine_sims = np.array(accuracy_data['cosine_similarities'])
    angular_errors = np.array(accuracy_data['angular_errors'])
    distance_errors = np.array(accuracy_data['distance_errors'])
    
    print(f"\nOverall Directional Accuracy:")
    print(f"  Mean cosine similarity: {np.mean(cosine_sims):.3f} ± {np.std(cosine_sims):.3f}")
    print(f"  Mean angular error: {np.mean(angular_errors):.1f}° ± {np.std(angular_errors):.1f}°")
    print(f"  Mean distance error: {np.mean(distance_errors):.3f} ± {np.std(distance_errors):.3f}")
    
    # By zone color
    for color in COLOR_NAMES:
        if color in accuracy_data['zone_colors']:
            mask = np.array(accuracy_data['zone_colors']) == color
            if np.any(mask):
                color_cosine = cosine_sims[mask]
                color_angular = angular_errors[mask]
                print(f"\n{color.capitalize()} zone:")
                print(f"  Mean cosine similarity: {np.mean(color_cosine):.3f} ± {np.std(color_cosine):.3f}")
                print(f"  Mean angular error: {np.mean(color_angular):.1f}° ± {np.std(color_angular):.1f}°")
    
    # Current goal vs other zones
    is_current_goal = np.array(accuracy_data['is_current_goal'])
    if np.any(is_current_goal):
        current_goal_cosine = cosine_sims[is_current_goal]
        other_zones_cosine = cosine_sims[~is_current_goal]
        
        print(f"\nCurrent Goal vs Other Zones:")
        print(f"  Current goal cosine similarity: {np.mean(current_goal_cosine):.3f} ± {np.std(current_goal_cosine):.3f}")
        print(f"  Other zones cosine similarity: {np.mean(other_zones_cosine):.3f} ± {np.std(other_zones_cosine):.3f}")
        print(f"  Difference: {np.mean(current_goal_cosine) - np.mean(other_zones_cosine):.3f}")

    # ── create visualization ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Compass Phenomenon Quantification - {formula}", fontsize=16)
    
    # Plot 1: Cosine similarity over time
    ax1 = axes[0,0]
    for color in COLOR_NAMES:
        if color in accuracy_data['zone_colors']:
            mask = np.array(accuracy_data['zone_colors']) == color
            if np.any(mask):
                timesteps = np.array(accuracy_data['timesteps'])[mask]
                cosine_vals = cosine_sims[mask]
                ax1.plot(timesteps, cosine_vals, 'o-', label=color, alpha=0.7, markersize=3)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Directional Accuracy Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Angular error over time
    ax2 = axes[0,1]
    for color in COLOR_NAMES:
        if color in accuracy_data['zone_colors']:
            mask = np.array(accuracy_data['zone_colors']) == color
            if np.any(mask):
                timesteps = np.array(accuracy_data['timesteps'])[mask]
                angular_vals = angular_errors[mask]
                ax2.plot(timesteps, angular_vals, 'o-', label=color, alpha=0.7, markersize=3)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Angular Error (degrees)')
    ax2.set_title('Angular Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Current goal vs other zones
    ax3 = axes[1,0]
    current_goal_mask = np.array(accuracy_data['is_current_goal'])
    if np.any(current_goal_mask):
        current_goal_cosine = cosine_sims[current_goal_mask]
        other_zones_cosine = cosine_sims[~current_goal_mask]
        
        ax3.hist(current_goal_cosine, bins=20, alpha=0.7, label='Current Goal', color='green')
        ax3.hist(other_zones_cosine, bins=20, alpha=0.7, label='Other Zones', color='red')
        ax3.set_xlabel('Cosine Similarity')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Directional Accuracy: Current Goal vs Other Zones')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distance error over time
    ax4 = axes[1,1]
    for color in COLOR_NAMES:
        if color in accuracy_data['zone_colors']:
            mask = np.array(accuracy_data['zone_colors']) == color
            if np.any(mask):
                timesteps = np.array(accuracy_data['timesteps'])[mask]
                distance_vals = distance_errors[mask]
                ax4.plot(timesteps, distance_vals, 'o-', label=color, alpha=0.7, markersize=3)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Distance Error')
    ax4.set_title('Distance Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved accuracy analysis to {args.out}")

if __name__ == '__main__':
    main() 