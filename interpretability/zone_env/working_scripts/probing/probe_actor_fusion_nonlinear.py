#!/usr/bin/env python3
"""
Non-linear probe experiment for DeepLTL Actor Fusion Layer
-------------------------------------------------------------------
Test if planning information is linearly entangled but accessible with non-linear decoders.

This script uses small MLPs instead of linear models to probe the fusion layer,
focusing on targets that showed poor linear performance (pose_k5, pose_k10, etc.).

Key differences from linear probe:
- Uses small MLPs (2-3 layers) instead of Ridge/LogisticRegression
- Focuses on targets that showed negative R² with linear probes
- Tests if planning information is non-linearly encoded
"""

import os, sys, argparse, random, numpy as np, torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from tqdm import trange

# ─── Deep-LTL imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from envs                   import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store      import ModelStore
from config                 import model_configs
from model.model            import build_model
from sequence.search        import ExhaustiveSearch
from model.agent            import Agent
import preprocessing

# ─── Dataset parameters ────────────────────────────────────────────────────────
ENV, EXP  = "PointLtl2-v0", "big_test"
SEED      = 0
N_WORLDS  = 10
N_ROLLOUT = 10
MAX_STEP  = 200

# ── Goal variation parameters ───────────────────────────────────────────────────
COLOURS = ["blue", "green", "yellow", "magenta"]
COLOUR2IDX = {colour: idx for idx, colour in enumerate(COLOURS)}
GOALS = [f"FG {c}" for c in COLOURS]

# ── Non-linear probe targets (focus on poor linear performers) ────────────────
NONLINEAR_TARGETS = [
    # Multi-step planning (poor linear performance)
    "pose_k5", "pose_k10",
    # Egocentric planning (poor linear performance)  
    "delta_body_1step", "delta_body_5step",
    # Physics-based (moderate linear performance)
    "speed_xy", "acc_xy", "speed_xy_sign",
    # Future prediction (good linear performance for comparison)
    "next_wall_lidar"
]

# ── MLP Architecture ───────────────────────────────────────────────────────────
class NonLinearProbe(nn.Module):
    """Small MLP for non-linear probing."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ── Helper functions ────────────────────────────────────────────────────────────
def get_target(env, obs, name):
    """Extract target features from environment/observation."""
    features = obs['features']
    
    # Direct inputs from the 80D feature vector
    if name == "agent_sensors":
        # acc(3) + gyro(3) + vel(3) = 9D sensor data
        acc_features = np.clip(features[0:3], -10.0, 10.0)  # Clip to ±10g
        gyro_features = features[38:41]  # Angular velocity
        vel_features = features[35:38]   # Linear velocity
        return np.concatenate([acc_features, gyro_features, vel_features]).copy()
    elif name == "zone_lidar":
        return features[19:35].copy()  # zone(19-34) - 16D
    elif name == "wall_lidar":
        return features[3:19].copy()  # wall(3-18) - 16D
    elif name == "wall_sensor":
        return features[41:45].copy()  # contact forces (4D)
    elif name == "agent_pos":
        return env.agent_pos[:2].copy()  # (x,y) regression
    elif name == "zone_id":
        current_props = obs.get('propositions', [])
        zone_colors = ['blue', 'green', 'yellow', 'magenta']
        zone_idx = 0  # Default to no zone
        for i, color in enumerate(zone_colors):
            if color in current_props:
                zone_idx = i + 1  # 1-indexed zone IDs
                break
        return np.array([zone_idx], dtype=int)
    elif name == "current_goal_colour":
        if hasattr(env, 'current_goal_colour'):
            colour = env.current_goal_colour
        else:
            current_props = obs.get('propositions', [])
            for colour in COLOURS:
                if colour in current_props:
                    break
            else:
                colour = "blue"
        return np.array([COLOUR2IDX.get(colour, 0)], dtype=int)
    elif name == "zone_distances":
        agent_pos = env.agent_pos[:2]
        if hasattr(env, 'zone_positions') and env.zone_positions:
            distances = []
            for zone_name in sorted(env.zone_positions.keys()):
                zone_pos = env.zone_positions[zone_name]
                dist = np.linalg.norm(agent_pos - zone_pos[:2])
                distances.append(dist)
            return np.array(distances)
        else:
            zone_lidar = features[19:35]
            max_intensity = np.max(zone_lidar)
            distance = 1.0 - max_intensity if max_intensity > 0 else 1.0
            return np.array([distance])
    elif name == "zone_directions":
        agent_pos = env.agent_pos[:2]
        if hasattr(env, 'zone_positions') and env.zone_positions:
            directions = []
            for zone_name in sorted(env.zone_positions.keys()):
                zone_pos = env.zone_positions[zone_name]
                direction = zone_pos[:2] - agent_pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                directions.extend(direction)
            return np.array(directions)
        else:
            return np.zeros(8)
    elif name == "successor_value":
        # Placeholder - would need to compute actual successor value
        return np.array([0.5], dtype=np.float32)
    elif name == "next_wall_lidar":
        # Placeholder - would need to compute actual next wall lidar
        return features[3:19].copy()  # Current wall lidar as placeholder
    else:
        return np.array([0.0], dtype=np.float32)

# ── Training functions ──────────────────────────────────────────────────────────
def train_mlp_probe(X_train, X_test, y_train, y_test, target_name, device="cpu"):
    """Train and evaluate a non-linear MLP probe."""
    
    # Determine output dimension and task type
    
    # Check if it's classification (integer targets)
    if len(y_train.shape) > 1:
        # For 2D arrays, check if it's actually a single column
        if y_train.shape[1] == 1:
            y_train_flat = y_train.flatten()
        else:
            output_dim = y_train.shape[1]
            is_classification = False  # Multi-dimensional regression
            # print(f"DEBUG: Multi-dimensional regression detected")
    else:
        y_train_flat = y_train
    
    # Check classification for single-column data
    if len(y_train.shape) == 1 or (len(y_train.shape) > 1 and y_train.shape[1] == 1):
        unique_vals = np.unique(y_train_flat)
        is_classification = (target_name.endswith("_sign") or 
                           (len(unique_vals) <= 10 and 
                            all(isinstance(v, (int, np.integer)) for v in unique_vals) and
                            set(unique_vals).issubset({0, 1})))
        
        # Force classification for speed_xy_sign
        if target_name == "speed_xy_sign":
            is_classification = True
        
        if is_classification:
            n_classes = int(y_train_flat.max()) + 1  # works for labels 0 … K-1
            output_dim = n_classes
        else:
            output_dim = 1
    
    # Standardize inputs and outputs
    scaler_x = StandardScaler().fit(X_train)
    X_train_scaled = scaler_x.transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    if not is_classification:
        scaler_y = StandardScaler().fit(y_train)
        y_train_scaled = scaler_y.transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        y_train_scaled = y_train
        y_test_scaled = y_test
    
    # Create model
    input_dim = X_train.shape[1]
    model = NonLinearProbe(input_dim, output_dim).to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    # Loss function and optimizer
    if is_classification:
        criterion = nn.CrossEntropyLoss()
        # Convert to class indices
        y_train_tensor = y_train_tensor.long().squeeze()
        y_test_tensor = y_test_tensor.long().squeeze()
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop - fixed epochs first, no early stopping
    model.train()
    best_score = -np.inf
    
    for epoch in range(50):  # Fixed 50 epochs
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        
        if is_classification:
            loss = criterion(outputs, y_train_tensor)
        else:
            loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            
            if is_classification:
                val_preds = torch.argmax(val_outputs, dim=1)
                score = accuracy_score(y_test_tensor.cpu().numpy(), val_preds.cpu().numpy())
            else:
                # For regression, unscale predictions before computing R²
                val_preds_unscaled = scaler_y.inverse_transform(val_outputs.cpu().numpy())
                y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)
                score = r2_score(y_test_unscaled, val_preds_unscaled)
        
        # Track best score
        if score > best_score:
            best_score = score
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        
        if is_classification:
            final_preds = torch.argmax(final_outputs, dim=1)
            final_score = accuracy_score(y_test_tensor.cpu().numpy(), final_preds.cpu().numpy())
            metric = "accuracy"
        else:
            # Unscale predictions for final evaluation
            final_preds_unscaled = scaler_y.inverse_transform(final_outputs.cpu().numpy())
            y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)
            final_score = r2_score(y_test_unscaled, final_preds_unscaled)
            metric = "R²"
    
    return final_score, metric

# ── Data collection (reuse from linear probe) ──────────────────────────────────
def collect_data():
    """Collect data for non-linear probing."""
    print("🔄 Collecting data for non-linear probing...")
    
    # Build model
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Data collection buffers
    buf_actor_input = []
    buf_actor_output = []
    buf_labels = {target: [] for target in NONLINEAR_TARGETS}
    world_ids = []
    
    # Global position buffer (persists across rollouts)
    global_pos_buffer = []
    global_embed_buffer = []
    
    # K-step pose data
    X_k5, Y_k5, world_ids_k5 = [], [], []
    X_k10, Y_k10, world_ids_k10 = [], [], []
    
    # Hook variables (use global for easier access)
    global current_embedding, collect_this_step
    current_embedding = None
    collect_this_step = False
    
    # Register hooks
    def fusion_hook(_, __, out):
        global collect_this_step  # easier than nonlocal dance
        if not collect_this_step:
            return
        collect_this_step = False  # reset immediately
        
        # Store the output (distribution parameters)
        if isinstance(out, tuple):
            dist = out[0]
            try:
                if hasattr(dist, 'loc') and hasattr(dist, 'scale'):
                    # For Normal distribution: capture (μ, log σ)
                    mu = dist.loc
                    log_sigma = torch.log(dist.scale)
                    dist_params = torch.cat([mu, log_sigma], dim=-1)
                    buf_actor_output.append(dist_params.detach().cpu().ravel().numpy())
                elif hasattr(dist, 'logits'):
                    # For discrete distributions
                    buf_actor_output.append(dist.logits.detach().cpu().ravel().numpy())
                elif hasattr(dist, 'dist') and hasattr(dist.dist, 'loc'):
                    # For wrapped distributions
                    mu = dist.dist.loc
                    log_sigma = torch.log(dist.dist.scale)
                    dist_params = torch.cat([mu, log_sigma], dim=-1)
                    buf_actor_output.append(dist_params.detach().cpu().ravel().numpy())
                else:
                    # Fallback: try to extract action parameters
                    buf_actor_output.append(dist.detach().cpu().ravel().numpy())
            except Exception as e:
                # If all else fails, just capture the action
                print(f"Warning: Could not extract distribution parameters: {e}")
                buf_actor_output.append(np.zeros(4))  # 2D action + 2D log_std
        else:
            buf_actor_output.append(out.detach().cpu().ravel().numpy())
    
    def hooked_compute_embedding(obs):
        global current_embedding
        embedding = original_compute_embedding(obs)
        if collect_this_step:
            current_embedding = embedding
        return embedding
    
    # Store original function and register hooks
    original_compute_embedding = model.compute_embedding
    model.register_forward_hook(fusion_hook)
    model.compute_embedding = hooked_compute_embedding
    
    # Progress bar
    total_rollouts = N_WORLDS * N_ROLLOUT
    pbar = trange(total_rollouts, desc="Collecting data", unit="rollout")
    
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            ltl_goal = GOALS[(wid * N_ROLLOUT + rid) % len(GOALS)]
            env = make_env(ENV, FixedSampler.partial(ltl_goal), sequence=False)
            props = set(env.get_propositions())
            planner = ExhaustiveSearch(model, props, num_loops=2)
            agent = Agent(model, planner, propositions=props, verbose=False)
            
            obs = env.reset(seed=SEED + 100*wid + rid)
            agent.reset()
            done = False
            step = 0
            
            while not done and step < MAX_STEP:
                # Collect pre-step state
                pre_state = {
                    'pos': env.agent_pos[:2].copy(),
                    'vel': getattr(env, 'agent_vel', np.zeros(2))
                }
                
                # Store position for k-step prediction (global buffer)
                global_pos_buffer.append(env.agent_pos[:2].copy())
                
                # Set flag to collect activation for this step
                collect_this_step = True
                
                # Store embedding for k-step prediction (global buffer)
                if current_embedding is not None:
                    global_embed_buffer.append(current_embedding.detach().cpu().numpy())
                
                # Get action and step environment
                action = agent.get_action(obs, {}, deterministic=True)
                
                # Ensure action is in correct format
                if isinstance(action, np.ndarray):
                    action = action.flatten()
                elif isinstance(action, torch.Tensor):
                    action = int(action.item())
                elif isinstance(action, (int, float)):
                    action = int(action)
                else:
                    action = int(action)
                
                obs, _, done, _ = env.step(action)
                
                # Collect post-step state
                post_state = {
                    'pos': env.agent_pos[:2].copy(),
                    'vel': getattr(env, 'agent_vel', np.zeros(2))
                }
                
                # Collect actor fusion input (embedding)
                if current_embedding is not None:
                    buf_actor_input.append(current_embedding.detach().cpu().numpy())
                    world_ids.append(wid)
                
                # Collect labels for all targets
                for target_name in NONLINEAR_TARGETS:
                    if target_name == "pose_k5" or target_name == "pose_k10":
                        continue  # Handle separately
                    
                    # Get target value
                    if target_name in ["speed_xy", "speed_xy_sign", "acc_xy", "delta_body_1step", "delta_body_5step"]:
                        # These need position history
                        if target_name == "speed_xy":
                            # Use 5-step horizon for less noisy speed calculation
                            if len(global_pos_buffer) >= 5:
                                delta_pos_5 = global_pos_buffer[-1] - global_pos_buffer[-5]
                                speed_xy = np.linalg.norm(delta_pos_5) / (5 * 0.02)  # Convert to m/s
                            elif len(global_pos_buffer) >= 2:
                                # Use 1-step if we have at least 2 positions
                                delta_pos = global_pos_buffer[-1] - global_pos_buffer[-2]
                                speed_xy = np.linalg.norm(delta_pos) / 0.02
                            else:
                                # Skip this step if not enough history
                                continue
                            label = np.array([speed_xy], dtype=np.float32)
                        elif target_name == "speed_xy_sign":
                            # Binary classification: moving vs not moving
                            if len(global_pos_buffer) >= 5:
                                delta_pos_5 = global_pos_buffer[-1] - global_pos_buffer[-5]
                                speed_xy = np.linalg.norm(delta_pos_5) / (5 * 0.02)
                            elif len(global_pos_buffer) >= 2:
                                # Use 1-step if we have at least 2 positions
                                delta_pos = global_pos_buffer[-1] - global_pos_buffer[-2]
                                speed_xy = np.linalg.norm(delta_pos) / 0.02
                            else:
                                # Skip this step if not enough history
                                continue
                            # Threshold at 0.1 m/s (reasonable threshold for meaningful movement)
                            is_moving = 1 if speed_xy > 0.1 else 0
                            label = np.array([is_moving], dtype=np.int64)
                            # print(f"DEBUG: speed_xy={speed_xy:.6f}, is_moving={is_moving}, label={label}")
                        elif target_name == "delta_body_1step":
                            # Body-frame 1-step displacement
                            if len(global_pos_buffer) >= 2:
                                # Get current and previous positions
                                pos_now = global_pos_buffer[-1]
                                pos_prev = global_pos_buffer[-2]
                                delta_world = pos_now - pos_prev
                                
                                # Get current heading from angular velocity (wz)
                                features = obs['features']
                                wz = features[40]  # yaw rate
                                heading = np.arctan2(wz, 0.001)  # Avoid division by zero
                                
                                # Convert to body frame
                                cos_h = np.cos(heading)
                                sin_h = np.sin(heading)
                                delta_body_x = delta_world[0] * cos_h + delta_world[1] * sin_h
                                delta_body_y = -delta_world[0] * sin_h + delta_world[1] * cos_h
                                
                                label = np.array([delta_body_x, delta_body_y], dtype=np.float32)
                            else:
                                # Skip this step if not enough history
                                continue
                        elif target_name == "delta_body_5step":
                            # Body-frame 5-step displacement
                            if len(global_pos_buffer) >= 6:
                                # Get current and 5-step-back positions
                                pos_now = global_pos_buffer[-1]
                                pos_prev = global_pos_buffer[-6]
                                delta_world = pos_now - pos_prev
                                
                                # Get current heading from angular velocity (wz)
                                features = obs['features']
                                wz = features[40]  # yaw rate
                                heading = np.arctan2(wz, 0.001)  # Avoid division by zero
                                
                                # Convert to body frame
                                cos_h = np.cos(heading)
                                sin_h = np.sin(heading)
                                delta_body_x = delta_world[0] * cos_h + delta_world[1] * sin_h
                                delta_body_y = -delta_world[0] * sin_h + delta_world[1] * cos_h
                                
                                label = np.array([delta_body_x, delta_body_y], dtype=np.float32)
                            else:
                                # Skip this step if not enough history
                                continue
                        elif target_name == "acc_xy":
                            features = obs['features']
                            ax, ay = features[0], features[1]
                            ax, ay = np.clip(ax, -10.0, 10.0), np.clip(ay, -10.0, 10.0)
                            acc_xy = np.sqrt(ax**2 + ay**2)
                            label = np.array([acc_xy], dtype=np.float32)
                    else:
                        # Use existing target functions
                        label = get_target(env, obs, target_name)
                    
                    buf_labels[target_name].append(label)
                
                step += 1
            
            env.close()
            
            # K-step pose prediction post-processing
            if len(global_pos_buffer) > 5 and len(global_embed_buffer) > 5:
                pose_now_5 = np.array(global_pos_buffer[:-5])
                pose_later_5 = np.array(global_pos_buffer[5:])
                embed_now_5 = np.array(global_embed_buffer[:-5])
                
                Y_k5.extend(pose_later_5 - pose_now_5)
                X_k5.extend(embed_now_5)
                world_ids_k5.extend([wid] * len(pose_later_5))
            
            if len(global_pos_buffer) > 10 and len(global_embed_buffer) > 10:
                pose_now_10 = np.array(global_pos_buffer[:-10])
                pose_later_10 = np.array(global_pos_buffer[10:])
                embed_now_10 = np.array(global_embed_buffer[:-10])
                
                Y_k10.extend(pose_later_10 - pose_now_10)
                X_k10.extend(embed_now_10)
                world_ids_k10.extend([wid] * len(pose_later_10))
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    
    # Convert to numpy arrays
    X_actor_input = np.vstack(buf_actor_input) if buf_actor_input else np.array([])
    X_actor_output = np.vstack(buf_actor_output) if buf_actor_output else np.array([])
    
    # Process k-step data
    if len(Y_k5) > 0:
        Y_k5 = np.vstack(Y_k5)
        X_k5 = np.vstack(X_k5)
    else:
        Y_k5 = None
        X_k5 = None
    
    if len(Y_k10) > 0:
        Y_k10 = np.vstack(Y_k10)
        X_k10 = np.vstack(X_k10)
    else:
        Y_k10 = None
        X_k10 = None
    
    return {
        'X_input': X_actor_input,
        'X_output': X_actor_output,
        'labels': buf_labels,
        'world_ids': np.array(world_ids),
        'k5_data': (X_k5, Y_k5, world_ids_k5) if Y_k5 is not None else None,
        'k10_data': (X_k10, Y_k10, world_ids_k10) if Y_k10 is not None else None
    }

# ── Main function ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="specific target to probe")
    ap.add_argument("--all", action="store_true", help="probe all non-linear targets")
    ap.add_argument("--device", default="cpu", help="device to use (cpu/cuda)")
    ap.add_argument("--no-holdout", action="store_true", help="use random 80/20 split instead of world hold-out")
    args = ap.parse_args()
    
    if not args.target and not args.all:
        ap.error("Please specify either --target or --all")
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Collect data
    data = collect_data()
    
    # Determine targets to probe
    targets_to_probe = NONLINEAR_TARGETS if args.all else [args.target]
    
    print(f"\n🎯 Non-linear probing {len(targets_to_probe)} target(s)...")
    print("=" * 80)
    
    results = []
    
    # Progress bar for probing
    probe_pbar = trange(len(targets_to_probe), desc="Probing targets", unit="target")
    
    for target_name in targets_to_probe:
        print(f"\n🔍 Non-linear probing: {target_name}")
        print("-" * 50)
        
        # Get target data
        if target_name == "pose_k5":
            if data['k5_data'] is None:
                print(f"  ❌ No k5 data available")
                continue
            Y = data['k5_data'][1]
            X = data['k5_data'][0]
            world_ids_for_target = data['k5_data'][2]
        elif target_name == "pose_k10":
            if data['k10_data'] is None:
                print(f"  ❌ No k10 data available")
                continue
            Y = data['k10_data'][1]
            X = data['k10_data'][0]
            world_ids_for_target = data['k10_data'][2]
        else:
            Y = np.vstack(data['labels'][target_name])
            X = data['X_input']
            world_ids_for_target = data['world_ids']
        
        # Align data
        min_length = min(len(X), len(Y))
        Y = Y[:min_length]
        X = X[:min_length]
        world_ids_for_target = world_ids_for_target[:min_length]
        
        # Split data
        if args.no_holdout:
            # Random 80/20 split
            n_samples = len(Y)
            indices = np.random.permutation(n_samples)
            split_idx = int(0.8 * n_samples)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            Y_train, Y_test = Y[train_indices], Y[test_indices]
            X_train, X_test = X[train_indices], X[test_indices]
        else:
            # Hold out worlds 3, 7
            held_out_worlds = [3, 7]
            train_mask = ~np.isin(world_ids_for_target, held_out_worlds)
            test_mask = np.isin(world_ids_for_target, held_out_worlds)
            
            Y_train, Y_test = Y[train_mask], Y[test_mask]
            X_train, X_test = X[train_mask], X[test_mask]
        
        # Ensure consistent lengths
        min_train_length = min(len(X_train), len(Y_train))
        min_test_length = min(len(X_test), len(Y_test))
        
        X_train = X_train[:min_train_length]
        X_test = X_test[:min_test_length]
        Y_train = Y_train[:min_train_length]
        Y_test = Y_test[:min_test_length]
        
        # Data quality check
        print(f"📊 {target_name} Data Quality:")
        print(f"  Samples: {len(Y)}")
        print(f"  Shape: {Y.shape}")
        print(f"  Range: [{Y.min():.3f}, {Y.max():.3f}]")
        print(f"  Mean: {Y.mean():.3f}")
        print(f"  Std: {Y.std():.3f}")
        
        if Y.std() < 1e-6:
            print(f"  ⚠️  WARNING: Very low variance (std={Y.std():.6f})")
            continue
        
        # Train non-linear probe
        try:
            input_score, input_metric = train_mlp_probe(
                X_train, X_test, Y_train, Y_test, target_name, args.device
            )
            
                        # For OUTPUT, use actor output data
            if target_name in ["pose_k5", "pose_k10"]:
                output_score, output_metric = 0.0, "N/A"
            else:
                # Get the correct train/test indices for output data
                if args.no_holdout:
                    train_idx, test_idx = train_indices, test_indices
                else:
                    train_idx = np.flatnonzero(train_mask)
                    test_idx = np.flatnonzero(test_mask)
                
                X_out_train = data['X_output'][train_idx][:len(Y_train)]
                X_out_test = data['X_output'][test_idx][:len(Y_test)]
                output_score, output_metric = train_mlp_probe(
                    X_out_train, X_out_test, Y_train, Y_test, target_name, args.device
                )
            
            # Store results
            result = {
                'target': target_name,
                'input_score': input_score,
                'output_score': output_score,
                'metric': input_metric,
                'shape': Y.shape[1] if len(Y.shape) > 1 else 1
            }
            results.append(result)
            
            # Print results
            print(f"  INPUT  (MLP)            {input_metric}: {input_score:.3f}")
            if output_metric != "N/A":
                print(f"  OUTPUT (MLP)            {output_metric}: {output_score:.3f}")
            else:
                print(f"  OUTPUT (MLP)            {output_metric}")
            
            # Debug: print what metric was returned

            
        except Exception as e:
            print(f"  ❌ Error probing {target_name}: {e}")
            continue
        
        # Update progress bar
        probe_pbar.update(1)
    
    probe_pbar.close()
    
    # Print summary table
    if results:
        print(f"\n📋 NON-LINEAR PROBE SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Target':<20} {'Shape':<6} {'INPUT':<12} {'OUTPUT':<12} {'Metric':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result['target']:<20} {result['shape']:<6} "
                  f"{result['input_score']:<12.3f} {result['output_score']:<12.3f} "
                  f"{result['metric']:<8}")

if __name__ == "__main__":
    main() 