#!/usr/bin/env python3
"""
Comprehensive linear-probe experiment for DeepLTL Actor Fusion Layer
-------------------------------------------------------------------
Probe what the actor network fusion layer has learned to represent.

This script probes the fusion output (where geometry and task information are combined)
to understand what information the policy has access to when making decisions.

Available Targets:
-----------------
ACTOR-SPECIFIC TARGETS:
- action_logits: Raw action logits from policy head (regression)
- action_index: Chosen action index (classification)
- td_value: Value estimate from critic head (regression)
- delta_xy: 1-step displacement vector from action (regression)
- collision_imminence: Binary collision prediction (classification)

ENVIRONMENT TARGETS (reused from env-net):
- agent_pos: Agent (x,y) position (regression)
- zone_id: Which zone agent is in (classification)
- current_goal_colour: Current goal color (classification)
- zone_distances: Distances to zone centers (regression)
- zone_directions: Direction vectors to zones (regression)

SENSOR TARGETS:
- agent_sensors: Accelerometer, velocimeter, gyro (regression)
- zone_lidar: Zone lidar readings (regression)
- wall_lidar: Wall lidar readings (regression)
- wall_sensor: Wall sensor readings (regression)

Examples
--------
# Probe all features
python probe_actor_fusion.py --all

# Probe specific feature
python probe_actor_fusion.py --target action_logits
"""
import os, sys, argparse, random, numpy as np, torch
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge, LogisticRegression
from sklearn.metrics       import r2_score, accuracy_score, roc_auc_score
from tqdm                  import trange

# ─── Deep-LTL imports (adapt path if your repo layout differs) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from envs                   import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store      import ModelStore
from config                 import model_configs
from model.model            import build_model
from sequence.search        import ExhaustiveSearch
from model.agent            import Agent
import preprocessing

# ─── Small, fast default dataset sizes ─────────────────────────────────────────
ENV, EXP  = "PointLtl2-v0", "big_test"
SEED      = 0
N_WORLDS  = 10
N_ROLLOUT = 10
MAX_STEP  = 200

# ── Goal variation parameters ───────────────────────────────────────────────────
COLOURS = ["blue", "green", "yellow", "magenta"]  # Basic colors likely available
COLOUR2IDX = {colour: idx for idx, colour in enumerate(COLOURS)}

# Add at the top, after COLOURS definition
GOALS = [f"FG {c}" for c in COLOURS]

# ── Action to displacement mapping ─────────────────────────────────────────────
# Assuming 4 discrete actions: [up, right, down, left]
ACTION_TO_DELTA = {
    0: [0, 1],   # up
    1: [1, 0],   # right
    2: [0, -1],  # down
    3: [-1, 0],  # left
}

# ── All available targets ───────────────────────────────────────────────────────
ALL_TARGETS = [
    # Actor-specific targets
    "action_logits", "action_index", "td_value", "delta_xy", "delta_xy_class", "collision_imminence",
    # Multi-step planning targets
    "pose_k5", "pose_k10",
    # Environment targets
    "agent_pos", "zone_id", "current_goal_colour", "zone_distances", "zone_directions",
    # Sensor targets
    "agent_sensors", "zone_lidar", "wall_lidar", "wall_sensor"
]

# ── Helpers ─────────────────────────────────────────────────────────────────────
def is_classification(y: np.ndarray) -> bool:
    "Integer labels with ≤32 unique values → classification."
    return np.issubdtype(y.dtype, np.integer) and np.unique(y).size <= 32

def get_target(env, obs, name):
    """Extract target features from environment/observation."""
    features = obs['features']
    
    # Direct inputs from the 80D feature vector
    # Layout: acc(0-2), wall(3-18), zone(19-34), vel(35-37), gyro(38-40), contact(41-46), remaining(47-79)
    if name == "agent_sensors":
        # acc(3) + gyro(3) + vel(3) = 9D sensor data
        return np.concatenate([features[0:3], features[38:41], features[35:38]]).copy()  # acc + gyro + vel
    elif name == "zone_lidar":
        return features[19:35].copy()  # zone(19-34) - 16D
    elif name == "wall_lidar":
        return features[3:19].copy()  # wall(3-18) - 16D
    elif name == "wall_sensor":
        # Use contact forces instead of velocity/gyro
        return features[41:45].copy()  # contact forces (4D)
    
    # Agent position (not in feature vector, accessed via env)
    elif name == "agent_pos":
        return env.agent_pos[:2].copy()  # (x,y) regression
    
    # Zone classification (derived from propositions)
    elif name == "zone_id":
        current_props = obs.get('propositions', [])
        zone_colors = ['blue', 'green', 'yellow', 'magenta']
        zone_idx = 0  # Default to no zone
        for i, color in enumerate(zone_colors):
            if color in current_props:
                zone_idx = i + 1  # 1-indexed zone IDs
                break
        return np.array([zone_idx], dtype=int)  # 1-D int label
    
    # Goal-related targets
    elif name == "current_goal_colour":
        # Get current goal color from environment
        if hasattr(env, 'current_goal_colour'):
            colour = env.current_goal_colour
        else:
            # Fallback: try to extract from propositions or environment state
            current_props = obs.get('propositions', [])
            for colour in COLOURS:
                if colour in current_props:
                    break
            else:
                colour = "blue"  # Default fallback
        return np.array([COLOUR2IDX.get(colour, 0)], dtype=int)
    
    # Derived features (computed from environment)
    elif name == "zone_distances":
        # Compute distances from agent to zone centers
        agent_pos = env.agent_pos[:2]
        if hasattr(env, 'zone_positions') and env.zone_positions:
            distances = []
            for zone_name in sorted(env.zone_positions.keys()):
                zone_pos = env.zone_positions[zone_name]
                dist = np.linalg.norm(agent_pos - zone_pos[:2])
                distances.append(dist)
            return np.array(distances)
        else:
            # Fallback: estimate from zone lidar intensities
            zone_lidar = features[19:35]  # zone(19-34) - 16D
            # Use max intensity as proxy for zone proximity
            max_intensity = np.max(zone_lidar)
            distance = 1.0 - max_intensity if max_intensity > 0 else 1.0
            return np.array([distance])
    
    elif name == "zone_directions":
        # Compute direction vectors from agent to zone centers
        agent_pos = env.agent_pos[:2]
        if hasattr(env, 'zone_positions') and env.zone_positions:
            directions = []
            for zone_name in sorted(env.zone_positions.keys()):
                zone_pos = env.zone_positions[zone_name]
                direction = zone_pos[:2] - agent_pos
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                directions.extend(direction)  # Add x, y components
            return np.array(directions)
        else:
            return np.zeros(8)  # 4 zones × 2D = 8D
    
    elif name == "delta_xy":
        # Use executed displacement (position change) instead of action
        # This avoids the circular dependency of predicting the actor's own output
        if hasattr(env, 'agent_pos') and hasattr(env, 'last_pos'):
            # Calculate actual displacement from position change
            current_pos = env.agent_pos[:2]
            last_pos = env.last_pos[:2]
            return current_pos - last_pos
        else:
            # Fallback: use action but this creates circular dependency
            if isinstance(action, np.ndarray) and len(action) == 2:
                # Continuous action case - use the action directly
                return action
            else:
                # Discrete action case - compute displacement from action
                if isinstance(action, np.ndarray):
                    action_idx = np.argmax(action)
                else:
                    action_idx = action
                
                if action_idx in ACTION_TO_DELTA:
                    dx, dy = ACTION_TO_DELTA[action_idx]
                    return np.array([dx, dy])
                else:
                    return np.array([0.0, 0.0])
    
    elif name == "collision_imminence":
        # Binary collision prediction based on wall lidar
        features = obs['features']
        wall_lidar = features[3:19]  # wall(3-18) - 16D
        min_distance = np.min(wall_lidar)
        collision_imminent = 1 if min_distance < 0.1 else 0
        return np.array([collision_imminent], dtype=int)
    
    elif name == "pose_k5":
        # The label is already stored in buf_lbl_dict by the rollout post-processing,
        # so here we just return a dummy – it will never be called.
        return np.zeros(2)
    
    elif name == "pose_k10":
        # The label is already stored in buf_lbl_dict by the rollout post-processing,
        # so here we just return a dummy – it will never be called.
        return np.zeros(2)
    
    else:
        raise ValueError(f"Target '{name}' not found in environment or feature vector.")

def get_actor_target(env, obs, action, agent, name):
    """Extract actor-specific target features."""
    if name == "action_logits":
        if hasattr(agent, 'last_logits'):
            return agent.last_logits.cpu().numpy()
        else:
            # Fallback: dummy logits
            return np.array([[0.0, 0.0, 0.0, 0.0]])
    
    elif name == "action_index":
        # Convert action to index
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return np.array([int(action.item())])
            else:
                return np.array([np.argmax(action)])
        else:
            return np.array([int(action)])
    
    elif name == "td_value":
        if hasattr(agent, 'last_value'):
            return agent.last_value.cpu().numpy()
        else:
            # Fallback: dummy value
            return np.array([0.0])
    
    elif name == "delta_xy":
        # Use executed displacement (position change) instead of action
        # This avoids the circular dependency of predicting the actor's own output
        if hasattr(env, 'agent_pos') and hasattr(env, 'last_pos'):
            # Calculate actual displacement from position change
            current_pos = env.agent_pos[:2]
            last_pos = env.last_pos[:2]
            return current_pos - last_pos
        else:
            # Fallback: use action but this creates circular dependency
            if isinstance(action, np.ndarray) and len(action) == 2:
                # Continuous action case - use the action directly
                return action
            else:
                # Discrete action case - compute displacement from action
                if isinstance(action, np.ndarray):
                    action_idx = np.argmax(action)
                else:
                    action_idx = action
                
                if action_idx in ACTION_TO_DELTA:
                    dx, dy = ACTION_TO_DELTA[action_idx]
                    return np.array([dx, dy])
                else:
                    return np.array([0.0, 0.0])
    
    elif name == "delta_xy_class":
        # Classification version of delta_xy - convert delta vector to discrete class
        if hasattr(env, 'agent_pos') and hasattr(env, 'last_pos'):
            # Calculate actual displacement from position change
            current_pos = env.agent_pos[:2]
            last_pos = env.last_pos[:2]
            delta = current_pos - last_pos
        else:
            # Fallback: use action
            if isinstance(action, np.ndarray) and len(action) == 2:
                delta = action
            else:
                # Discrete action case - compute displacement from action
                if isinstance(action, np.ndarray):
                    action_idx = np.argmax(action)
                else:
                    action_idx = action
                
                if action_idx in ACTION_TO_DELTA:
                    dx, dy = ACTION_TO_DELTA[action_idx]
                    delta = np.array([dx, dy])
                else:
                    delta = np.array([0.0, 0.0])
        
        # Convert delta vector to angle-based discrete class
        # Calculate angle in degrees
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        
        # Map angle to discrete classes
        if -45 <= angle < 45:
            cls = 1  # right
        elif 45 <= angle < 135:
            cls = 0  # up
        elif angle >= 135 or angle < -135:
            cls = 3  # left
        else:
            cls = 2  # down
        
        return np.array([cls])
    
    elif name == "collision_imminence":
        # Binary collision prediction based on wall lidar
        features = obs['features']
        wall_lidar = features[3:19]  # wall(3-18) - 16D
        min_distance = np.min(wall_lidar)
        collision_imminent = 1 if min_distance < 0.1 else 0
        return np.array([collision_imminent], dtype=int)
    
    else:
        # Fall back to environment targets
        return get_target(env, obs, name)

def validate_target_data(Y, target_name):
    """Validate target data quality."""
    print(f"\n📊 {target_name} Data Quality:")
    print(f"  Samples: {len(Y)}")
    print(f"  Shape: {Y.shape}")
    print(f"  Range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  Mean: {Y.mean():.3f}")
    print(f"  Std: {Y.std():.3f}")
    
    # Check for all-zeros or constant targets
    if Y.std() < 1e-6:
        print(f"  ⚠️  WARNING: Very low variance (std={Y.std():.6f})")
        return False
    if np.allclose(Y, 0):
        print(f"  ❌ ERROR: All zeros!")
        return False
    
    # Target-specific checks
    if target_name == "zone_id":
        unique_vals = np.unique(Y)
        print(f"  Zone IDs: {unique_vals}")
        if len(unique_vals) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_vals)} unique zone ID(s)")
    elif target_name in ["zone_lidar", "wall_lidar"]:
        non_zero_bins = np.sum(Y > 0.01, axis=1)
        print(f"  Avg non-zero bins: {non_zero_bins.mean():.1f}")
        if non_zero_bins.mean() < 1:
            print(f"  ⚠️  WARNING: Very few non-zero readings")
    elif target_name == "agent_pos":
        pos_range = np.linalg.norm(Y, axis=1)
        print(f"  Position range: [{pos_range.min():.3f}, {pos_range.max():.3f}]")
        if pos_range.max() < 0.1:
            print(f"  ⚠️  WARNING: Positions very close to origin")
    elif target_name in ["current_goal_colour", "action_index"]:
        unique_vals = np.unique(Y)
        print(f"  Values: {unique_vals}")
        if len(unique_vals) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_vals)} unique value(s)")
    elif target_name == "collision_imminence":
        unique_vals = np.unique(Y)
        print(f"  Collision values: {unique_vals}")
        if len(unique_vals) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_vals)} unique collision value(s)")
    elif target_name == "delta_xy_class":
        unique_vals = np.unique(Y)
        print(f"  Delta classes: {unique_vals}")
        if len(unique_vals) < 2:
            print(f"  ⚠️  WARNING: Only {len(unique_vals)} unique delta class(es)")
    
    return True

def train_and_evaluate_probe(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate a probe for a specific target."""
    clf_task = is_classification(y_train)
    
    # Check if this is an identity target (X == y)
    y_is_raw_input = target_name in ["action_logits", "agent_sensors", "zone_lidar", "wall_lidar", "wall_sensor", "raw_features"]
    
    # For multi-dimensional targets, we need to handle each dimension separately
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"  DEBUG: Multi-dimensional target detected: {y_train.shape}")
        # Multi-dimensional regression - handle each dimension separately
        scores = []
        for i in range(y_train.shape[1]):
            y_train_dim = y_train[:, i]
            y_test_dim = y_test[:, i]
            
            pipe_dim = make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if not y_is_raw_input else Ridge(alpha=1.0)
            pipe_dim.fit(X_train, y_train_dim)
            y_pred_dim = pipe_dim.predict(X_test)
            
            # Calculate R² for this dimension
            ss_res = np.sum((y_test_dim - y_pred_dim) ** 2)
            ss_tot = np.sum((y_test_dim - np.mean(y_test_dim)) ** 2)
            if ss_tot > 0:
                score_dim = 1 - (ss_res / ss_tot)
            else:
                score_dim = 1.0
            scores.append(score_dim)
        
        # Return average R² across dimensions
        return np.mean(scores), "R²"
    
    # Create pipeline for classification or single-dimensional regression
    if clf_task:
        pipe = make_pipeline(StandardScaler(), 
                           LogisticRegression(max_iter=1000, class_weight="balanced"))
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    else:
        if y_is_raw_input:
            # Skip scaling for identity targets to avoid overflow
            pipe = Ridge(alpha=1.0)
        else:
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        # Single-dimensional regression
        y_train_flat = y_train.ravel()
        y_test_flat = y_test.ravel()
    
    # Train and predict (for classification or single-dimensional regression)
    pipe.fit(X_train, y_train_flat)
    y_pred = pipe.predict(X_test)
    
    # Calculate score
    if clf_task:
        if target_name == "collision_imminence":
            # Use AUROC for binary classification
            score = roc_auc_score(y_test_flat, y_pred)
            metric = "AUROC"
        else:
            score = accuracy_score(y_test_flat, y_pred)
            metric = "accuracy"
    else:
        if y_is_raw_input:
            # For identity targets, use simple R² calculation
            # Calculate R² manually to avoid multioutput issues
            ss_res = np.sum((y_test_flat - y_pred) ** 2)
            ss_tot = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
            if ss_tot > 0:
                score = 1 - (ss_res / ss_tot)
            else:
                score = 1.0
            metric = "R²"
        else:
            score = r2_score(y_test_flat, y_pred, multioutput="uniform_average")
            metric = "R²"
    
    return score, metric

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="specific target to decode")
    ap.add_argument("--all", action="store_true", help="probe all available targets")
    args = ap.parse_args()

    if not args.target and not args.all:
        ap.error("Please specify either --target or --all")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # Build dummy env + model
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[ENV]
    store = ModelStore(ENV, EXP, SEED); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── Capture buffers ─────────────────────────────────────────────
    buf_actor_input = []  # Actor fusion input activations (128D embedding)
    buf_actor_output = []  # Actor fusion output activations (distribution logits)
    buf_lbl_dict = {t: [] for t in ALL_TARGETS} if args.all else {}
    buf_lbl_single = [] if not args.all else None  # For single target
    current_env = None
    current_obs = None
    current_action = None
    current_agent = None
    current_wid = 0  # Track current world ID
    world_ids = []  # Record world IDs in post_hook
    goal_sequences = []  # Record goal sequences
    current_embedding = None  # Store the embedding for input probing

    # Data collection buffers
    buf_actor_input = []
    buf_actor_output = []
    buf_lbl_dict = {target: [] for target in ALL_TARGETS}
    buf_lbl_single = []
    world_ids = []
    
    # K-step pose prediction buffers
    pos_buffer = []  # (x,y) at every env step
    embed_buffer = []  # actor fusion activations
    buf_labels_k5 = []  # future pose targets for k=5
    buf_labels_k10 = []  # future pose targets for k=10
    X_k5 = []  # separate embeddings for k=5
    Y_k5 = []  # separate labels for k=5
    world_ids_k5 = []  # separate world_ids for k=5
    X_k10 = []  # separate embeddings for k=10
    Y_k10 = []  # separate labels for k=10
    world_ids_k10 = []  # separate world_ids for k=10

    # Global flag to control when to collect activations
    global collect_this_step
    collect_this_step = False

    def fusion_hook(_, __, out):
        global collect_this_step  # easier than nonlocal dance
        if not collect_this_step:
            return
        collect_this_step = False  # reset immediately
        
        # 1. store activations (actor pass only)
        if current_embedding is not None:
            buf_actor_input.append(current_embedding.detach().cpu().ravel().numpy())
        
        # Store the output (distribution logits)
        if isinstance(out, tuple):
            dist = out[0]
            if hasattr(dist, 'logits'):
                buf_actor_output.append(dist.logits.detach().cpu().ravel().numpy())
            elif hasattr(dist, 'dist') and hasattr(dist.dist, 'logits'):
                buf_actor_output.append(dist.dist.logits.detach().cpu().ravel().numpy())
            elif hasattr(dist, 'loc'):
                buf_actor_output.append(dist.loc.detach().cpu().ravel().numpy())
            else:
                buf_actor_output.append(dist.dist.loc.detach().cpu().ravel().numpy())
        else:
            buf_actor_output.append(out.detach().cpu().ravel().numpy())
        
        world_ids.append(current_wid)
        
        # 2. store labels **here** so lengths always match
        if current_env is not None and current_obs is not None and current_action is not None and current_agent is not None:
            if args.all:
                # Collect all targets
                for target_name in ALL_TARGETS:
                    buf_lbl_dict[target_name].append(get_actor_target(current_env, current_obs, current_action, current_agent, target_name))
            else:
                # Collect single target
                buf_lbl_single.append(get_actor_target(current_env, current_obs, current_action, current_agent, args.target))

    # Hook the compute_embedding method to capture the input
    original_compute_embedding = model.compute_embedding
    
    def hooked_compute_embedding(obs):
        nonlocal current_embedding
        embedding = original_compute_embedding(obs)
        if collect_this_step:
            current_embedding = embedding
        return embedding
    
    model.compute_embedding = hooked_compute_embedding

    # Register hook on the model's forward method
    model.register_forward_hook(fusion_hook)

    # Roll-out data with varied goals
    print("🔄 Collecting data with varied goals...")
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            ltl_goal = GOALS[(wid * N_ROLLOUT + rid) % len(GOALS)]
            env = make_env(ENV, FixedSampler.partial(ltl_goal), sequence=False)
            props = set(env.get_propositions())
            planner = ExhaustiveSearch(model, props, num_loops=2)
            agent = Agent(model, planner, propositions=props, verbose=False)
            goal_sequences.append(ltl_goal)
            obs = env.reset(seed=SEED + 100*wid + rid)
            agent.reset(); done=False; step=0
            last_pos = None # Initialize last_pos for delta_xy calculation
            while not done and step < MAX_STEP:
                current_env = env
                current_obs = obs
                current_wid = wid
                # Set flag to collect activation for this step
                collect_this_step = True
                
                # Store current position for delta_xy calculation
                if hasattr(env, 'agent_pos'):
                    current_pos = env.agent_pos[:2].copy()
                    if last_pos is not None:
                        env.last_pos = last_pos  # Store for delta_xy calculation
                    last_pos = current_pos
                
                # Store position for k-step pose prediction
                pos_buffer.append(env.agent_pos[:2].copy())
                if current_embedding is not None:
                    embed_buffer.append(current_embedding.detach().cpu().numpy())
                
                action = agent.get_action(obs, {}, deterministic=True)
                current_action = action
                current_agent = agent
                
                # Store logits and value in agent for later access
                # This needs to be done after the model forward pass
                if hasattr(agent, 'last_logits'):
                    agent.last_logits = agent.last_logits
                if hasattr(agent, 'last_value'):
                    agent.last_value = agent.last_value
                
                # Ensure action is in the correct format for the environment
                # The action should be an integer for discrete actions or an array for continuous
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        action = int(action.item())
                    else:
                        # For continuous actions, flatten the array but keep as numpy array
                        action = action.flatten()
                elif isinstance(action, torch.Tensor):
                    action = int(action.item())
                elif isinstance(action, (int, float)):
                    action = int(action)
                else:
                    # Fallback: try to convert to int
                    action = int(action)
                obs, _, done, _ = env.step(action)
                step += 1
            env.close()

            # Record goal sequence
            if hasattr(env, 'goal_sequence'):
                goal_sequences.append(env.goal_sequence)
            
            # K-step pose prediction post-processing
            if len(pos_buffer) > 5:  # k=5
                pose_now_5 = np.array(pos_buffer[:-5])  # drop last 5
                pose_later_5 = np.array(pos_buffer[5:])  # shifted
                embed_now_5 = np.array(embed_buffer[:-5])
                
                X_k5.extend(embed_now_5)
                # Use relative displacement instead of absolute position
                Y_k5.extend(pose_later_5 - pose_now_5)
                world_ids_k5.extend([wid] * len(pose_later_5))
            
            if len(pos_buffer) > 10:  # k=10
                pose_now_10 = np.array(pos_buffer[:-10])  # drop last 10
                pose_later_10 = np.array(pos_buffer[10:])  # shifted
                embed_now_10 = np.array(embed_buffer[:-10])
                
                X_k10.extend(embed_now_10)
                # Use relative displacement instead of absolute position
                Y_k10.extend(pose_later_10 - pose_now_10)
                world_ids_k10.extend([wid] * len(pose_later_10))
            
            # Clear buffers for next rollout
            pos_buffer = []
            embed_buffer = []

    # Align data
    X_actor_input = np.vstack(buf_actor_input)
    X_actor_output = np.vstack(buf_actor_output)
    world_ids = np.array(world_ids)
    
    print(f"📊 Collected {len(X_actor_input)} samples")
    print(f"  Actor fusion input shape: {X_actor_input.shape}")
    print(f"  Actor fusion output shape: {X_actor_output.shape}")
    print(f"  World IDs shape: {world_ids.shape}")
    
    # Ensure all arrays have the same length
    min_length = min(len(X_actor_input), len(X_actor_output), len(world_ids))
    X_actor_input = X_actor_input[:min_length]
    X_actor_output = X_actor_output[:min_length]
    world_ids = world_ids[:min_length]
    
    print(f"  Aligned length: {min_length}")

    # Analyze goal variety
    unique_goals = set(tuple(seq) for seq in goal_sequences)
    print(f"  Unique goal sequences: {len(unique_goals)}")
    print(f"  Total goal sequences: {len(goal_sequences)}")

    # Split data using held-out worlds for better generalization
    # Use last 2 worlds for testing (out of 10 total)
    held_out_worlds = [8, 9]  # Last 2 worlds for testing
    
    # Use recorded world_ids from post_hook
    # world_ids = np.array(world_ids) # This line is now redundant as world_ids is already numpy
    
    train_mask = ~np.isin(world_ids, held_out_worlds)
    test_mask = np.isin(world_ids, held_out_worlds)
    
    X_actor_input_train, X_actor_input_test = X_actor_input[train_mask], X_actor_input[test_mask]
    X_actor_output_train, X_actor_output_test = X_actor_output[train_mask], X_actor_output[test_mask]
    
    print(f"  Train samples: {np.sum(train_mask)}")
    print(f"  Test samples: {np.sum(test_mask)}")

    # Determine targets to probe
    targets_to_probe = ALL_TARGETS if args.all else [args.target]
    
    print(f"\n🎯 Probing {len(targets_to_probe)} target(s)...")
    print("=" * 80)
    
    results = []
    
    for target_name in targets_to_probe:
        print(f"\n🔍 Probing: {target_name}")
        print("-" * 40)
        
        # Get target data
        if target_name == "pose_k5":
            Y = np.vstack(Y_k5)
            X = np.vstack(X_k5)
            world_ids_for_target = np.array(world_ids_k5)
        elif target_name == "pose_k10":
            Y = np.vstack(Y_k10)
            X = np.vstack(X_k10)
            world_ids_for_target = np.array(world_ids_k10)
        elif args.all:
            # Use pre-collected labels
            Y = np.vstack(buf_lbl_dict[target_name])
            X = X_actor_input  # Use the main actor input
            world_ids_for_target = world_ids  # Use the main world_ids
        else:
            # Use pre-collected single target labels
            Y = np.vstack(buf_lbl_single)
            X = X_actor_input  # Use the main actor input
            world_ids_for_target = world_ids  # Use the main world_ids
        
        # Align labels with activations
        min_length = min(len(X), len(Y))
        Y = Y[:min_length]
        X = X[:min_length]
        world_ids_for_target = world_ids_for_target[:min_length]
        
        # Decide which ID array this target uses and build fresh masks
        if target_name == "pose_k5":
            world_ids_target = np.array(world_ids_k5)[:min_length]
        elif target_name == "pose_k10":
            world_ids_target = np.array(world_ids_k10)[:min_length]
        else:
            world_ids_target = world_ids_for_target  # default (1-step data)
        
        # Build fresh masks from the correct array
        train_mask_target = ~np.isin(world_ids_target, held_out_worlds)
        test_mask_target = np.isin(world_ids_target, held_out_worlds)
        
        # Validate data quality
        if not validate_target_data(Y, target_name):
            print(f"  ❌ Skipping {target_name} due to poor data quality")
            continue
        
        # Split target data using the correct masks
        Y_train, Y_test = Y[train_mask_target], Y[test_mask_target]
        X_train, X_test = X[train_mask_target], X[test_mask_target]
        
        # Ensure all arrays have consistent lengths
        min_train_length = min(len(X_train), len(Y_train))
        min_test_length = min(len(X_test), len(Y_test))
        
        X_train = X_train[:min_train_length]
        X_test = X_test[:min_test_length]
        Y_train = Y_train[:min_train_length]
        Y_test = Y_test[:min_test_length]
        
        # Train and evaluate probe
        try:
            # Probe INPUT (actor fusion input)
            input_score, input_metric = train_and_evaluate_probe(X_train, X_test, Y_train, Y_test, target_name)
            
            # For OUTPUT, we need to use the actor output data
            # Get the corresponding actor output data
            if target_name in ["pose_k5", "pose_k10"]:
                # For k-step targets, we need to get the corresponding actor output data
                # This is more complex - for now, let's skip OUTPUT probing for k-step targets
                output_score, output_metric = 0.0, "N/A"
            else:
                # Use the main actor output data
                X_out_train = X_actor_output_train[:len(Y_train)]
                X_out_test = X_actor_output_test[:len(Y_test)]
                output_score, output_metric = train_and_evaluate_probe(X_out_train, X_out_test, Y_train, Y_test, target_name)
            
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
            print(f"  INPUT  (128D emb)       {input_metric}: {input_score:.3f}")
            if output_metric != "N/A":
                print(f"  OUTPUT (dist logits)     {output_metric}: {output_score:.3f}")
            else:
                print(f"  OUTPUT (dist logits)     {output_metric}")
            
        except Exception as e:
            print(f"  ❌ Error probing {target_name}: {e}")
            continue
    
    # Print summary table
    if results:
        print(f"\n📋 SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Target':<20} {'Shape':<6} {'INPUT':<12} {'OUTPUT':<12} {'Metric':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result['target']:<20} {result['shape']:<6} "
                  f"{result['input_score']:<12.3f} {result['output_score']:<12.3f} "
                  f"{result['metric']:<8}")

if __name__ == "__main__":
    main() 