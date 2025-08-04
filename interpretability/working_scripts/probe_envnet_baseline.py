#!/usr/bin/env python3
"""
Comprehensive linear-probe experiment for DeepLTL Env-Net
--------------------------------------------------------
Probe what the environment network has learned to represent.

Available Targets:
-----------------
DIRECT INPUTS (from 80D feature vector):
- agent_sensors: Accelerometer, velocimeter, gyro (3D)
- zone_lidar: Zone lidar readings (16D)
- wall_lidar: Wall lidar readings (16D) 
- wall_sensor: Wall sensor readings (4D)
- remaining_features: Remaining 41 dimensions
- raw_features: Entire 80D feature vector
- features_subset: Last 16 elements
- velocity_features: First 6 elements

AGENT STATE (from environment):
- agent_pos: Agent (x,y) position (2D regression)

ZONE CLASSIFICATION (from propositions):
- zone_id: Which zone agent is in (1D classification)

DERIVED FEATURES (computed from environment):
- zone_distances: Distances to zone centers (regression)
- zone_directions: Direction vectors to zones (regression)
- zone_differences: Differences between zone lidars (regression)

INDIVIDUAL ZONE LIDAR:
- blue_zone_lidar: Blue zone lidar readings (4D)
- green_zone_lidar: Green zone lidar readings (4D)
- yellow_zone_lidar: Yellow zone lidar readings (4D)
- magenta_zone_lidar: Magenta zone lidar readings (4D)

Examples
--------
# Probe all features
python probe_envnet_baseline.py --all

# Probe specific feature
python probe_envnet_baseline.py --target agent_pos
"""
import os, sys, argparse, random, numpy as np, torch
from sklearn.pipeline      import make_pipeline
from sklearn.preprocessing import StandardScaler                 # :contentReference[oaicite:0]{index=0}
from sklearn.linear_model  import Ridge, LogisticRegression      # :contentReference[oaicite:1]{index=1}
from sklearn.metrics       import r2_score, accuracy_score
from tqdm                  import trange

# ─── Deep-LTL imports (adapt path if your repo layout differs) ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from envs                   import make_env
from ltl                    import FixedSampler
from utils.model_store      import ModelStore
from config                 import model_configs
from model.model            import build_model
from sequence.search        import ExhaustiveSearch
from model.agent            import Agent

# ─── Small, fast default dataset sizes ─────────────────────────────────────────
ENV, EXP  = "PointLtl2-v0", "big_test"
SEED      = 0
N_WORLDS  = 10
N_ROLLOUT = 10
MAX_STEP  = 200

# ── All available targets ───────────────────────────────────────────────────────
ALL_TARGETS = [
    # Direct inputs from 80D feature vector
    "agent_sensors", "zone_lidar", "wall_lidar", "wall_sensor", 
    "remaining_features", "raw_features", "features_subset", "velocity_features",
    # Agent state
    "agent_pos",
    # Zone classification  
    "zone_id",
    # Derived features
    "zone_distances", "zone_directions", "zone_differences",
    # Individual zone lidar
    "blue_zone_lidar", "green_zone_lidar", "yellow_zone_lidar", "magenta_zone_lidar"
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
        # This might be in the remaining features, let's check the actual layout
        return features[35:39].copy()  # vel + gyro_x as wall sensor
    elif name == "remaining_features":
        return features[47:80].copy()  # remaining(47-79) - 33D
    elif name == "raw_features":
        return features.copy()  # Entire 80D vector
    elif name == "features_subset":
        return features[-16:].copy()  # Last 16 elements
    elif name == "velocity_features":
        return features[35:41].copy()  # vel(35-37) + gyro(38-40) - 6D
    
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
            zone_lidar = features[3:19]
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
    
    elif name == "zone_differences":
        # Differences between zone lidar readings
        zone_lidar = features[19:35]  # zone(19-34) - 16D
        # Split into 4 zones of 4 readings each
        if len(zone_lidar) >= 8:
            zone1 = zone_lidar[:4]  # First zone (blue)
            zone2 = zone_lidar[4:8]  # Second zone (green)
            return zone1 - zone2
        else:
            return np.zeros(4)
    
    # Individual zone lidar readings
    elif name == "blue_zone_lidar":
        return features[19:23].copy()  # zone(19-22) - First 4 zone lidar readings
    elif name == "green_zone_lidar":
        return features[23:27].copy()  # zone(23-26) - Next 4 zone lidar readings
    elif name == "yellow_zone_lidar":
        return features[27:31].copy()  # zone(27-30) - Next 4 zone lidar readings
    elif name == "magenta_zone_lidar":
        return features[31:35].copy()  # zone(31-34) - Last 4 zone lidar readings
    
    else:
        raise KeyError(f"unknown target {name}")

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
    
    return True

def train_and_evaluate_probe(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate a probe for a specific target."""
    clf_task = is_classification(y_train)
    
    # Check if this is an identity target (X == y)
    y_is_raw_input = target_name in ["raw_features", "agent_sensors", "velocity_features"]
    
    # Create pipeline
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
        y_train_flat = y_train
        y_test_flat = y_test
    
    # Train and predict
    pipe.fit(X_train, y_train_flat)
    y_pred = pipe.predict(X_test)
    
    # Calculate score
    if clf_task:
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
    buf_in, buf_out = [], []
    buf_lbl_dict = {t: [] for t in ALL_TARGETS} if args.all else {}
    buf_lbl_single = [] if not args.all else None  # For single target
    pre_cache = []  # holds exactly one actor-input tensor
    last_actor_step = -1  # monotonically increasing by 1 per env.step
    current_env = None
    current_obs = None
    current_wid = 0  # Track current world ID
    world_ids = []  # Record world IDs in post_hook

    def pre_hook(_, inp):
        """Runs *before* env_net forward. Save raw input for the next actor pass."""
        nonlocal pre_cache
        pre_cache[:] = [inp[0].detach().cpu().ravel().numpy()]

    def post_hook(_, __, out):
        """Runs *after* env_net forward. Keep only the *first* call each env step (actor)."""
        nonlocal last_actor_step, pre_cache, current_wid
        if len(buf_out) == last_actor_step:  # critic → skip
            return
        last_actor_step = len(buf_out)

        buf_out.append(out.detach().cpu().ravel().numpy())  # 64-D embedding
        
        # Safety guard for pre_cache
        if not pre_cache:
            return  # safety just before buf_in.append()
        
        buf_in.append(pre_cache.pop())  # matching 80-D input
        world_ids.append(current_wid)  # Record current world ID
        
        # Collect labels for this timestep
        if current_env is not None and current_obs is not None:
            if args.all:
                # Collect all targets
                for target_name in ALL_TARGETS:
                    buf_lbl_dict[target_name].append(get_target(current_env, current_obs, target_name))
            else:
                # Collect single target
                buf_lbl_single.append(get_target(current_env, current_obs, args.target))

    model.env_net.register_forward_pre_hook(pre_hook)
    model.env_net.register_forward_hook(post_hook)

    # Roll-out data
    env   = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False)
    props = set(env.get_propositions())
    planner = ExhaustiveSearch(model, props, num_loops=2)
    agent   = Agent(model, planner, propositions=props, verbose=False)

    print("🔄 Collecting data...")
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            obs = env.reset(seed=SEED + 100*wid + rid)
            agent.reset(); done=False; step=0
            while not done and step < MAX_STEP:
                # Set current environment and observation for hooks
                current_env = env
                current_obs = obs
                current_wid = wid # Update current world ID
                
                action = agent.get_action(obs, {}, deterministic=True).flatten()
                obs, _, done, _ = env.step(action)
                step += 1

    env.close()

    # Align data
    X_in, X_out = map(np.vstack, (buf_in, buf_out))
    print(f"📊 Collected {len(X_in)} samples")
    print(f"  Input shape: {X_in.shape}")
    print(f"  Output shape: {X_out.shape}")

    # Split data using held-out worlds for better generalization
    # Use last 2 worlds for testing (out of 10 total)
    held_out_worlds = [8, 9]  # Last 2 worlds for testing
    
    # Use recorded world_ids from post_hook
    world_ids = np.array(world_ids)
    
    train_mask = ~np.isin(world_ids, held_out_worlds)
    test_mask = np.isin(world_ids, held_out_worlds)
    
    X_in_train, X_in_test = X_in[train_mask], X_in[test_mask]
    X_out_train, X_out_test = X_out[train_mask], X_out[test_mask]
    
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
        if args.all:
            # Use pre-collected labels
            Y = np.vstack(buf_lbl_dict[target_name])
        else:
            # Use pre-collected single target labels
            Y = np.vstack(buf_lbl_single)
        
        # Validate data quality
        if not validate_target_data(Y, target_name):
            print(f"  ❌ Skipping {target_name} due to poor data quality")
            continue
        
        # Split target data
        Y_train, Y_test = Y[train_mask], Y[test_mask]
        
        # Train and evaluate probes
        try:
            # Probe INPUT
            input_score, input_metric = train_and_evaluate_probe(
                X_in_train, X_in_test, Y_train, Y_test, target_name
            )
            
            # Probe OUTPUT  
            output_score, output_metric = train_and_evaluate_probe(
                X_out_train, X_out_test, Y_train, Y_test, target_name
            )
            
            # Store results
            results.append({
                'target': target_name,
                'input_score': input_score,
                'output_score': output_score,
                'metric': input_metric,
                'shape': Y.shape[1] if len(Y.shape) > 1 else 1
            })
            
            # Print results
            print(f"  INPUT  (80D obs)       {input_metric}: {input_score:.3f}")
            print(f"  OUTPUT (64D emb)       {output_metric}: {output_score:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error probing {target_name}: {e}")
            continue
    
    # Print summary table
    if results:
        print(f"\n📋 SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Target':<20} {'Shape':<6} {'INPUT':<10} {'OUTPUT':<10} {'Metric':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result['target']:<20} {result['shape']:<6} "
                  f"{result['input_score']:<10.3f} {result['output_score']:<10.3f} "
                  f"{result['metric']:<8}")

if __name__ == "__main__":
    main()
