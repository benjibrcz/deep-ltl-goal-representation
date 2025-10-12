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

# ─── Larger dataset for comprehensive probing ──────────────────────────────────
ENV, EXP  = "PointLtl2-v0", "big_test"
SEED      = 0
N_WORLDS  = 10
N_ROLLOUT = 10
MAX_STEP  = 500

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

# ── Label helper functions (improved velocity handling) ───────────────────────
def body_vel(env, obs):
    """Extract body-frame velocity from observation features."""
    return obs['features'][35:38]  # vx, vy, vz in body frame

def world_speed(env, obs):
    """Calculate world-frame speed from body-frame velocity."""
    vx, vy, _ = body_vel(env, obs)
    # rotate by current yaw (if available) to world frame
    yaw = getattr(env, 'agent_yaw', 0.0)
    c, s = np.cos(yaw), np.sin(yaw)
    v_world = np.array([c*vx - s*vy, s*vx + c*vy])
    return np.linalg.norm(v_world)

def get_yaw_rate(env, obs):
    """Extract yaw rate from observation features."""
    return obs['features'][40]  # wz directly

def time_diff_velocity(env, obs):
    """Calculate acceleration from velocity time difference."""
    if not hasattr(time_diff_velocity, 'prev_vel'):
        time_diff_velocity.prev_vel = body_vel(env, obs)
        return np.array([0.0, 0.0, 0.0])
    
    v_now = body_vel(env, obs)
    v_prev = time_diff_velocity.prev_vel
    acc = (v_now - v_prev) / 0.02  # Assuming 20ms timesteps
    time_diff_velocity.prev_vel = v_now.copy()
    return acc

# ── All available targets ───────────────────────────────────────────────────────
ALL_TARGETS = [
    # Actor-specific targets
    "action_logits", "action_index", "td_value", "delta_xy", "delta_xy_class", "collision_imminence",
    # Multi-step planning targets
    "pose_k5", "pose_k10",
    # Egocentric planning probes
    "delta_body_1step", "delta_body_5step", "heading_change",
    # Velocity probes (recommended by user)
    "vz", "wz", "wz_sign", "speed_xy", "speed_xy_sign", "acc_xy",  # Core velocity quantities
    # Legacy physics-based probes (keeping for comparison)
    "vx_vy", "speed_sq", "fwd_speed", "side_speed", "vel_3d", "vel_stats", "yaw_rate",
    # Future prediction probes
    "next_wall_lidar", "successor_value", "automaton_edge",
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
        # Clip acceleration values to prevent overflow in StandardScaler
        # Accelerometer can spike to ±20g, causing numerical issues
        acc_features = np.clip(features[0:3], -10.0, 10.0)  # Clip to ±10g
        gyro_features = features[38:41]  # Angular velocity
        vel_features = features[35:38]   # Linear velocity
        return np.concatenate([acc_features, gyro_features, vel_features]).copy()
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
    
    elif name == "delta_body_1step":
        # Egocentric displacement in body frame (1 step ahead)
        # This requires storing position and yaw from previous step
        if not hasattr(get_target, 'prev_pos') or not hasattr(get_target, 'prev_yaw'):
            get_target.prev_pos = env.agent_pos[:2].copy()
            get_target.prev_yaw = getattr(env, 'agent_yaw', 0.0)
            return np.array([0.0, 0.0])  # First step
        
        # Compute global displacement
        current_pos = env.agent_pos[:2].copy()
        current_yaw = getattr(env, 'agent_yaw', 0.0)
        delta_global = current_pos - get_target.prev_pos
        
        # Transform to body frame
        yaw = get_target.prev_yaw  # Use previous yaw as reference
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s], [s, c]])
        delta_body = R @ delta_global
        
        # Update stored values
        get_target.prev_pos = current_pos.copy()
        get_target.prev_yaw = current_yaw
        
        return delta_body
    
    elif name == "delta_body_5step":
        # Egocentric displacement in body frame (5 steps ahead)
        # This requires a buffer of positions and yaws
        if not hasattr(get_target, 'pos_buffer'):
            get_target.pos_buffer = []
            get_target.yaw_buffer = []
        
        current_pos = env.agent_pos[:2].copy()
        current_yaw = getattr(env, 'agent_yaw', 0.0)
        
        get_target.pos_buffer.append(current_pos)
        get_target.yaw_buffer.append(current_yaw)
        
        # Need at least 6 positions to compute 5-step displacement
        if len(get_target.pos_buffer) < 6:
            return np.array([0.0, 0.0])
        
        # Compute 5-step displacement
        pos_t = get_target.pos_buffer[-6]  # 5 steps ago
        yaw_t = get_target.yaw_buffer[-6]
        pos_tp5 = get_target.pos_buffer[-1]  # current position
        
        delta_global = pos_tp5 - pos_t
        
        # Transform to body frame at time t
        c, s = np.cos(-yaw_t), np.sin(-yaw_t)
        R = np.array([[c, -s], [s, c]])
        delta_body = R @ delta_global
        
        return delta_body
    
    elif name == "heading_change":
        # Estimate heading from successive positions
        if not hasattr(get_target, "heading_change_prev_pos"):
            get_target.heading_change_prev_pos  = env.agent_pos[:2].copy()
            get_target.heading_change_prev_hdg  = 0.0
            return np.array([8], dtype=int)        # neutral first step

        # current heading
        delta = env.agent_pos[:2] - get_target.heading_change_prev_pos
        if np.linalg.norm(delta) < 1e-6:
            hdg = get_target.heading_change_prev_hdg             # no movement → keep old heading
        else:
            hdg = np.arctan2(delta[1], delta[0])  # radians [-π,π)

        # heading change
        dpsi = ((hdg - get_target.heading_change_prev_hdg + np.pi) % (2*np.pi)) - np.pi
        cls  = int(np.floor((dpsi + np.pi) / (2*np.pi/16)))  # 16 bins (0-15)

        # Debug: check if we're getting variety
        if not hasattr(get_target, 'heading_change_debug_count'):
            get_target.heading_change_debug_count = 0
        get_target.heading_change_debug_count += 1
        if get_target.heading_change_debug_count <= 10:  # Print first 10 steps
            # print(f"DEBUG: delta={delta}, hdg={hdg:.3f}, prev_hdg={get_target.heading_change_prev_hdg:.3f}, dpsi={dpsi:.3f}, cls={cls}")
            # Also check action space type
            if get_target.heading_change_debug_count == 1:
                # print(f"DEBUG: env.action_space = {env.action_space}")
                # print(f"DEBUG: env.action_space.shape = {getattr(env.action_space, 'shape', 'N/A')}")
                # print(f"DEBUG: env.action_space.n = {getattr(env.action_space, 'n', 'N/A')}")
                pass

        # update
        get_target.heading_change_prev_pos = env.agent_pos[:2].copy()
        get_target.heading_change_prev_hdg = hdg

        return np.array([cls], dtype=int)
    
    elif name == "next_wall_lidar":
        # Next-step wall-lidar (16-beam vector at t+1)
        # This requires storing the current observation and getting the next one
        if not hasattr(get_target, 'current_obs'):
            get_target.current_obs = obs
            return np.zeros(16)  # First step - no next observation yet
        
        # Store current observation for next step comparison
        next_obs = get_target.current_obs
        get_target.current_obs = obs
        
        # Extract wall lidar from next observation
        if 'features' in next_obs:
            wall_lidar = next_obs['features'][3:19]  # wall(3-18) - 16D
            return wall_lidar
        else:
            return np.zeros(16)
    
    elif name == "successor_value":
        # Successor-value (V_t+5) - run critic 5 steps ahead
        # This is a placeholder - would need to implement Monte Carlo rollout
        # For now, use a simple heuristic based on current observation
        features = obs['features']
        
        # Simple heuristic: use distance to goal zones as proxy for future value
        zone_lidar = features[19:35]  # zone(19-34) - 16D
        max_zone_intensity = np.max(zone_lidar)
        
        # Convert to value estimate (0-1)
        value = max_zone_intensity if max_zone_intensity > 0 else 0.1
        return np.array([value])
    
    elif name == "automaton_edge":
        # Predicted automaton edge (which LTL transition fires next)
        # This requires tracking the automaton state transitions
        if not hasattr(get_target, 'prev_automaton_state'):
            get_target.prev_automaton_state = obs.get('ldba_state', 0)
            return np.array([0], dtype=int)  # First step
        
        current_state = obs.get('ldba_state', 0)
        prev_state = get_target.prev_automaton_state
        
        # Check if state changed (edge fired)
        edge_fired = 1 if current_state != prev_state else 0
        
        get_target.prev_automaton_state = current_state
        
        return np.array([edge_fired], dtype=int)
    
    elif name == "next_velocity":
        # Next velocity v_{t+1} - what the physics engine produces from the action
        # This requires storing velocity from the previous step
        if not hasattr(get_target, 'prev_velocity'):
            get_target.prev_velocity = np.zeros(2)  # First step
            return np.zeros(2)
        
        # Current velocity (after physics step)
        current_velocity = getattr(env, 'agent_vel', np.zeros(2))
        
        # Debug: check if velocity is accessible
        if not hasattr(get_target, 'debug_vel_count'):
            get_target.debug_vel_count = 0
        get_target.debug_vel_count += 1
        if get_target.debug_vel_count <= 5:
            # print(f"DEBUG: env.agent_vel exists: {hasattr(env, 'agent_vel')}")
            # print(f"DEBUG: current_velocity: {current_velocity}")
            # print(f"DEBUG: env attributes with 'vel': {[attr for attr in dir(env) if 'vel' in attr.lower()]}")
            # print(f"DEBUG: env attributes with 'agent': {[attr for attr in dir(env) if 'agent' in attr.lower()]}")
            pass
        
        # Store for next step
        next_velocity = current_velocity.copy()
        get_target.prev_velocity = current_velocity.copy()
        
        return next_velocity
    
    elif name == "speed":
        # Signed speed ||v_{t+1}|| - use velocity from observation features
        features = obs['features']
        # Layout: acc(0-2), wall(3-18), zone(19-34), vel(35-37), gyro(38-40), contact(41-46), remaining(47-79)
        velocity_features = features[35:38]  # vel(35-37) - 3D velocity
        speed = np.linalg.norm(velocity_features)
        
        return np.array([speed])
    
    elif name == "yaw_rate":
        # Yaw rate ω_t = atan2(vy, vx) - atan2(prev_vy, prev_vx)
        if not hasattr(get_target, 'prev_velocity'):
            get_target.prev_velocity = np.zeros(2)
            return np.array([0.0])
        
        current_velocity = getattr(env, 'agent_vel', np.zeros(2))
        prev_velocity = get_target.prev_velocity
        
        # Compute yaw rates
        current_yaw = np.arctan2(current_velocity[1], current_velocity[0])
        prev_yaw = np.arctan2(prev_velocity[1], prev_velocity[0])
        
        # Handle wrapping
        yaw_rate = ((current_yaw - prev_yaw + np.pi) % (2*np.pi)) - np.pi
        
        get_target.prev_velocity = current_velocity.copy()
        
        return np.array([yaw_rate])
    
    elif name == "delta_pos_world":
        # Δpos in world frame p_{t+1} - p_t
        if not hasattr(get_target, 'prev_pos'):
            get_target.prev_pos = env.agent_pos[:2].copy()
            return np.zeros(2)
        
        current_pos = env.agent_pos[:2].copy()
        prev_pos = get_target.prev_pos
        
        delta_pos = current_pos - prev_pos
        get_target.prev_pos = current_pos.copy()
        
        return delta_pos
    
    elif name == "delta_pos_body":
        # Δpos in body frame (rotation-invariant)
        if not hasattr(get_target, 'delta_pos_body_prev_pos') or not hasattr(get_target, 'delta_pos_body_prev_hdg'):
            get_target.delta_pos_body_prev_pos = env.agent_pos[:2].copy()
            get_target.delta_pos_body_prev_hdg = 0.0
            return np.zeros(2)
        
        current_pos = env.agent_pos[:2].copy()
        prev_pos = get_target.delta_pos_body_prev_pos
        
        # Compute heading from movement direction
        delta_world = current_pos - prev_pos
        if np.linalg.norm(delta_world) < 1e-6:
            heading = get_target.delta_pos_body_prev_hdg  # no movement → keep old heading
        else:
            heading = np.arctan2(delta_world[1], delta_world[0])
        
        # Transform to body frame
        c, s = np.cos(-heading), np.sin(-heading)
        R = np.array([[c, -s], [s, c]])
        delta_body = R @ delta_world
        
        get_target.delta_pos_body_prev_pos = current_pos.copy()
        get_target.delta_pos_body_prev_hdg = heading
        
        return delta_body
    
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
        # For continuous actions, the logits are the action parameters (mean, std)
        # The fusion hook captures the distribution parameters (μ, log σ)
        if isinstance(action, np.ndarray):
            # Handle both 1D and 2D arrays
            if action.ndim == 2:
                # 2D array [[x, y]] -> flatten to [x, y]
                return action.flatten()
            elif action.ndim == 1 and len(action) == 2:
                # 1D array [x, y] -> return as is
                return action
            else:
                # Fallback: dummy logits
                return np.array([0.0, 0.0])
        else:
            # Fallback: dummy logits
            return np.array([0.0, 0.0])
    
    # Debug: check what action we're getting
    if name == "action_logits" and not hasattr(get_actor_target, 'debug_action_count'):
        get_actor_target.debug_action_count = 0
    if name == "action_logits":
        get_actor_target.debug_action_count += 1
        if get_actor_target.debug_action_count <= 5:
            # print(f"DEBUG: action type: {type(action)}, shape: {getattr(action, 'shape', 'N/A')}, value: {action}")
            pass
    
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

def get_actor_target_with_state(env, obs, action, agent, name, pre_state=None, post_state=None):
    """Extract actor-specific target features using pre/post state."""
    
    # Physics-based targets that need post-step state
    if name == "delta_pos_world":
        # Skip this probe - it's an arbitrary heuristic
        return np.zeros(2)
    
    elif name == "next_velocity":
        # Skip this probe - it's redundant with vx_vy
        return np.zeros(2)
    
    elif name == "vx_vy":
        # Body-frame velocity components (vx, vy)
        features = obs['features']
        vx = features[35]  # forward + / back - (body-frame)
        vy = features[36]  # left + / right - (body-frame)
        
        # Debug: print velocity components for first few samples
        if not hasattr(get_actor_target_with_state, 'debug_vx_vy_count'):
            get_actor_target_with_state.debug_vx_vy_count = 0
        get_actor_target_with_state.debug_vx_vy_count += 1
        if get_actor_target_with_state.debug_vx_vy_count <= 3:
            # print(f"DEBUG: vx_vy [vx, vy] = [{vx:.3f}, {vy:.3f}]")
            # print(f"DEBUG: Direction signs: vx={np.sign(vx)}, vy={np.sign(vy)}")
            pass
        
        return np.array([vx, vy], dtype=np.float32)  # Return vx, vy components
    
    elif name == "speed_sq":
        # Speed squared (quadratic but linear in squared components)
        features = obs['features']
        acc_features = features[0:3]  # acc(0-2) - 3D (use acceleration as proxy for velocity)
        speed_sq = acc_features[0]**2 + acc_features[1]**2 + acc_features[2]**2
        return np.array([speed_sq])
    
    elif name == "fwd_speed":
        # body-frame forward speed (signed; forward ≈ negative)
        features = obs['features']
        fwd_speed = features[37]  # body-frame forward speed
        
        # Debug: print velocity statistics for first few samples
        if not hasattr(get_actor_target_with_state, 'debug_vel_stats_count'):
            get_actor_target_with_state.debug_vel_stats_count = 0
        get_actor_target_with_state.debug_vel_stats_count += 1
        if get_actor_target_with_state.debug_vel_stats_count <= 10:
            vx = features[35]  # world-frame x (should be 0)
            vy = features[36]  # world-frame y (should be 0)
            vz = features[37]  # body-frame forward speed
            # print(f"DEBUG: vx={vx:.3f}, vy={vy:.3f}, vz={vz:.3f}")
            pass
        
        return np.array([fwd_speed], dtype=np.float32)
    
    elif name == "side_speed":
        # lateral speed (side-stepping) - body-frame velocity components
        features = obs['features']
        # Body-frame velocity: vx (forward/back), vy (left/right), vz (up/down)
        vx = features[35]  # forward + / back - (body-frame)
        vy = features[36]  # left + / right - (body-frame)
        vz = features[37]  # up + / down - (body-frame)
        
        # Debug: print velocity direction statistics for first few samples
        if not hasattr(get_actor_target_with_state, 'debug_vel_direction_count'):
            get_actor_target_with_state.debug_vel_direction_count = 0
        get_actor_target_with_state.debug_vel_direction_count += 1
        if get_actor_target_with_state.debug_vel_direction_count <= 5:
            # print(f"DEBUG: Body-frame velocity [35:38] = {features[35:38]}")
            # print(f"DEBUG: Angular velocity [38:41] = {features[38:41]}")
            # print(f"DEBUG: vx={vx:.3f} (forward/back), vy={vy:.3f} (left/right), vz={vz:.3f} (up/down)")
            pass
        
        # Return lateral velocity (left/right movement)
        return np.array([vy], dtype=np.float32)  # Left + / right - movement
    
    elif name == "vel_3d":
        # Full 3D velocity vector (body-frame)
        features = obs['features']
        # Body-frame velocity: vx (forward/back), vy (left/right), vz (up/down)
        vx = features[35]  # forward + / back - (body-frame)
        vy = features[36]  # left + / right - (body-frame)
        vz = features[37]  # up + / down - (body-frame)
        
        # Debug: print 3D velocity for first few samples
        if not hasattr(get_actor_target_with_state, 'debug_vel_3d_count'):
            get_actor_target_with_state.debug_vel_3d_count = 0
        get_actor_target_with_state.debug_vel_3d_count += 1
        if get_actor_target_with_state.debug_vel_3d_count <= 3:
            # print(f"DEBUG: 3D velocity [vx, vy, vz] = [{vx:.3f}, {vy:.3f}, {vz:.3f}]")
            # print(f"DEBUG: Speed = {np.sqrt(vx**2 + vy**2 + vz**2):.3f}")
            # print(f"DEBUG: Direction signs: vx={np.sign(vx)}, vy={np.sign(vy)}, vz={np.sign(vz)}")
            pass
        
        # Return full 3D velocity vector
        return np.array([vx, vy, vz], dtype=np.float32)
    
    elif name == "vel_stats":
        # Comprehensive velocity statistics for all components
        features = obs['features']
        
        # Collect all velocity-related components
        vx = features[35]  # world-frame x (always 0)
        vy = features[36]  # world-frame y (always 0) 
        vz = features[37]  # body-frame forward speed (varies)
        ax = features[0]   # acceleration x
        ay = features[1]   # acceleration y
        az = features[2]   # acceleration z
        gyro_x = features[38]  # angular velocity x
        gyro_y = features[39]  # angular velocity y
        
        # Debug: check for angle-related features
        if not hasattr(get_actor_target_with_state, 'debug_angle_count'):
            get_actor_target_with_state.debug_angle_count = 0
        get_actor_target_with_state.debug_angle_count += 1
        if get_actor_target_with_state.debug_angle_count <= 3:
            # print(f"DEBUG: Full features [0:80] = {features}")
            # print(f"DEBUG: Looking for angle features...")
            # Check if there are any features that could represent angles
            # for i in range(0, 80, 10):
            #     print(f"DEBUG: Features [{i}:{i+10}] = {features[i:i+10]}")
            pass
        
        # Return comprehensive velocity statistics
        # This will help us understand the full velocity encoding
        return np.array([vx, vy, vz, ax, ay, az, gyro_x, gyro_y], dtype=np.float32)
    
    elif name == "speed":
        # Original speed (non-linear, should be harder for linear probe)
        features = obs['features']
        acc_features = features[0:3]  # acc(0-2) - 3D (use acceleration as proxy for velocity)
        speed = np.linalg.norm(acc_features)
        
        # Debug: check what acceleration features we're getting
        if not hasattr(get_actor_target_with_state, 'debug_speed_count'):
            get_actor_target_with_state.debug_speed_count = 0
        get_actor_target_with_state.debug_speed_count += 1
        if get_actor_target_with_state.debug_speed_count <= 5:
            # print(f"DEBUG: acc_features = {acc_features}")
            # print(f"DEBUG: speed = {speed}")
            pass
        
        return np.array([speed])
    
    elif name == "yaw_rate":
        # Use the SAME observation that produced the fusion embedding
        # For now, use gyro features as a proxy for yaw rate
        features = obs['features']
        gyro_features = features[38:41]  # gyro(38-40) - 3D
        # Use z-component as yaw rate (simplified)
        yaw_rate = gyro_features[2] if len(gyro_features) > 2 else 0.0
        return np.array([yaw_rate])
    
    elif name == "delta_pos_body":
        # Skip this probe - it's an arbitrary heuristic
        return np.zeros(2)
    
    # New velocity probes (improved implementation)
    elif name == "vz":
        # Signed forward speed (vertical velocimeter)
        features = obs['features']
        vz = features[37]  # body-frame forward speed (signed)
        return np.array([vz], dtype=np.float32)
    
    elif name == "wz":
        # Yaw rate (from gyro)
        wz = get_yaw_rate(env, obs)
        return np.array([wz], dtype=np.float32)
    
    elif name == "wz_sign":
        # Yaw rate sign (binary classification)
        wz = get_yaw_rate(env, obs)
        wz_sign = 1 if wz > 0 else 0  # Binary: turning left (1) or right (0)
        return np.array([wz_sign], dtype=int)
    
    elif name == "speed_xy":
        # Real horizontal speed derived from positions (coarser horizon)
        if pre_state is None or post_state is None:
            return np.array([0.0], dtype=np.float32)
        
        # Calculate displacement magnitude over 5 steps for better resolution
        # This reduces numerical noise from single-step measurements
        delta_pos = post_state['pos'] - pre_state['pos']
        speed_xy = np.linalg.norm(delta_pos)
        
        # Scale to m/s (assuming 20ms timesteps)
        speed_xy = speed_xy / 0.02  # Convert from m/step to m/s
        
        return np.array([speed_xy], dtype=np.float32)
    
    elif name == "speed_xy_sign":
        # Binary classification: moving vs not moving
        if pre_state is None or post_state is None:
            return np.array([0], dtype=int)
        
        # Calculate speed from position change (same as speed_xy)
        delta_pos = post_state['pos'] - pre_state['pos']
        speed_xy = np.linalg.norm(delta_pos) / 0.02  # Convert to m/s
        
        # Threshold at 0.1 m/s for meaningful movement
        moving = 1 if speed_xy > 0.1 else 0
        return np.array([moving], dtype=int)
    
    elif name == "acc_xy":
        # Horizontal acceleration from velocity time difference
        acc = time_diff_velocity(env, obs)
        acc_xy = np.sqrt(acc[0]**2 + acc[1]**2)  # Magnitude of horizontal acceleration
        return np.array([acc_xy], dtype=np.float32)
    
    else:
        # Fall back to original function for other targets
        return get_actor_target(env, obs, action, agent, name)

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
    
    # Determine appropriate metric based on target type
    if target_name.endswith('_sign') or target_name in ['zone_id', 'current_goal_colour', 'action_index', 'delta_xy_class', 'heading_change']:
        metric = 'accuracy'
        # Use balanced class weights for imbalanced datasets
        pipe = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, class_weight="balanced"))
    elif target_name == "collision_imminence":
        metric = 'AUROC'
        pipe = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000, class_weight="balanced"))
    else:
        metric = 'R²'
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    
    # For multi-dimensional targets, we need to handle each dimension separately
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        # print(f"  DEBUG: Multi-dimensional target detected: {y_train.shape}")
        # Multi-dimensional regression - handle each dimension separately
        scores = []
        for i in range(y_train.shape[1]):
            y_train_dim = y_train[:, i]
            y_test_dim = y_test[:, i]
            
            pipe_dim = make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if not y_is_raw_input else Ridge(alpha=1.0)
            pipe_dim.fit(X_train, y_train_dim)
            y_pred_dim = pipe_dim.predict(X_test)
            
            # Calculate R² for this dimension (with epsilon to prevent overflow)
            ss_res = np.sum((y_test_dim - y_pred_dim) ** 2)
            ss_tot = np.sum((y_test_dim - np.mean(y_test_dim)) ** 2)
            if ss_tot > 1e-12:  # Add epsilon to prevent division by tiny values
                score_dim = 1 - (ss_res / ss_tot)
            else:
                score_dim = 1.0
            scores.append(score_dim)
        
        # Return average R² across dimensions
        return np.mean(scores), "R²"
    
    # Single-dimensional target
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()
    
    # Train and predict
    pipe.fit(X_train, y_train_flat)
    y_pred = pipe.predict(X_test)
    
    # Calculate score based on metric type
    if metric == 'accuracy':
        score = accuracy_score(y_test_flat, y_pred)
    elif metric == 'AUROC':
        # For AUROC, we need probability predictions
        if hasattr(pipe, 'predict_proba'):
            y_pred_proba = pipe.predict_proba(X_test)[:, 1]  # Probability of positive class
        else:
            y_pred_proba = y_pred  # Fallback to binary predictions
        score = roc_auc_score(y_test_flat, y_pred_proba)
    else:  # R²
        if y_is_raw_input:
            # For identity targets, use simple R² calculation
            ss_res = np.sum((y_test_flat - y_pred) ** 2)
            ss_tot = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
            if ss_tot > 1e-12:  # Add epsilon to prevent division by tiny values
                score = 1 - (ss_res / ss_tot)
            else:
                score = 1.0
        else:
            score = r2_score(y_test_flat, y_pred, multioutput="uniform_average")
    
    return score, metric

def validate_label_quality(X_raw, y, target_name):
    """Quick sanity check: train Ridge on raw 80-D features to validate label quality."""
    if target_name in ["action_logits", "agent_sensors", "zone_lidar", "wall_lidar", "wall_sensor"]:
        return  # Skip identity targets
    
    # Use Ridge regression on raw features
    pipe_raw = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe_raw.fit(X_raw, y.ravel())
    y_pred_raw = pipe_raw.predict(X_raw)
    
    # Calculate R² on raw features
    score_raw = r2_score(y.ravel(), y_pred_raw)
    
    print(f"  📊 Raw feature R²: {score_raw:.3f}")
    if score_raw < 0.1:
        print(f"  ⚠️  WARNING: Low raw feature R² - label may be flawed")
    
    return score_raw

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="specific target to decode")
    ap.add_argument("--all", action="store_true", help="probe all available targets")
    ap.add_argument("--cv", action="store_true", help="use 5-fold cross-validation over worlds")
    ap.add_argument("--mlp", action="store_true", help="use MLP instead of Ridge for non-linear probing")
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
    
    def reset_buffer_state():
        """Reset buffer state to prevent cross-rollout contamination."""
        # Reset function-level static variables
        if hasattr(get_target, 'prev_pos'):
            delattr(get_target, 'prev_pos')
        if hasattr(get_target, 'prev_yaw'):
            delattr(get_target, 'prev_yaw')
        if hasattr(get_target, 'pos_buffer'):
            delattr(get_target, 'pos_buffer')
        if hasattr(get_target, 'yaw_buffer'):
            delattr(get_target, 'yaw_buffer')
        if hasattr(get_target, 'heading_change_prev_pos'):
            delattr(get_target, 'heading_change_prev_pos')
        if hasattr(get_target, 'heading_change_prev_hdg'):
            delattr(get_target, 'heading_change_prev_hdg')
        if hasattr(get_target, 'current_obs'):
            delattr(get_target, 'current_obs')
        if hasattr(get_target, 'prev_automaton_state'):
            delattr(get_target, 'prev_automaton_state')
        
        # Reset helper function state
        if hasattr(time_diff_velocity, 'prev_vel'):
            delattr(time_diff_velocity, 'prev_vel')
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
        
        world_ids.append(current_wid)
        
        # 2. store labels **here** so lengths always match
        if current_env is not None and current_obs is not None and current_action is not None and current_agent is not None:
            # Debug: check action values (commented out for cleaner output)
            # if not hasattr(fusion_hook, 'debug_count'):
            #     fusion_hook.debug_count = 0
            # fusion_hook.debug_count += 1
            # if fusion_hook.debug_count <= 5:
            #     print(f"DEBUG: current_action type: {type(current_action)}, value: {current_action}")
            
            if args.all:
                # Collect all targets using post-step state
                for target_name in ALL_TARGETS:
                    label = get_actor_target_with_state(current_env, current_obs, current_action, current_agent, target_name, current_pre_state, current_post_state)
                    buf_lbl_dict[target_name].append(label)
            else:
                # Collect single target
                label = get_actor_target_with_state(current_env, current_obs, current_action, current_agent, args.target, current_pre_state, current_post_state)
                buf_lbl_single.append(label)

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
    
    # Progress bar for data collection
    total_rollouts = N_WORLDS * N_ROLLOUT
    pbar = trange(total_rollouts, desc="Collecting data", unit="rollout")
    
    for wid in range(N_WORLDS):
        for rid in range(N_ROLLOUT):
            ltl_goal = GOALS[(wid * N_ROLLOUT + rid) % len(GOALS)]
            env = make_env(ENV, FixedSampler.partial(ltl_goal), sequence=False)
            
            # Reset buffer state to prevent cross-rollout contamination
            reset_buffer_state()
            
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
                
                # Collect pre-step state
                pre_state = {}
                pre_state['pos'] = env.agent_pos[:2].copy()
                pre_state['vel'] = getattr(env, 'agent_vel', np.zeros(2))
                
                # Set flag to collect activation for this step
                collect_this_step = True
                
                # Store position for k-step pose prediction
                pos_buffer.append(env.agent_pos[:2].copy())
                if current_embedding is not None:
                    embed_buffer.append(current_embedding.detach().cpu().numpy())
                
                action = agent.get_action(obs, {}, deterministic=True)
                current_action = action
                current_agent = agent
                
                # Ensure action is in the correct format for the environment
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        action = int(action.item())
                    else:
                        action = action.flatten()
                elif isinstance(action, torch.Tensor):
                    action = int(action.item())
                elif isinstance(action, (int, float)):
                    action = int(action)
                else:
                    action = int(action)
                
                # Step the environment
                obs, _, done, _ = env.step(action)
                
                # Collect post-step state
                post_state = {}
                post_state['pos'] = env.agent_pos[:2].copy()
                post_state['vel'] = getattr(env, 'agent_vel', np.zeros(2))
                
                # Store post-step state for label generation
                current_post_state = post_state
                current_pre_state = pre_state
                
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
            
            # Update progress bar
            pbar.update(1)

    # Close progress bar
    pbar.close()
    
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
    held_out_worlds = [3,7]  # Hold out world 7 for testing (with larger dataset)
    
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
    
    # Progress bar for probing
    probe_pbar = trange(len(targets_to_probe), desc="Probing targets", unit="target")
    
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
        
        # Update progress bar
        probe_pbar.update(1)
    
    # Close progress bar
    probe_pbar.close()
    
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