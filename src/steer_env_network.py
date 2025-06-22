#!/usr/bin/env python3
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from envs.flatworld    import FlatWorld
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

ENV       = "PointLtl2-v0"
EXP       = "big_test"
SEED      = 1
MAX_STEPS = 500
STEER_FREQ = 0.5  # How often to steer (0.5 = 50% of the time)

# pick whatever formula you like here:
FORMULA = "GF blue & GF green"

class EnvNetworkSteerer:
    def __init__(self, model, blue_probe, green_probe):
        self.model = model
        self.blue_probe = blue_probe
        self.green_probe = green_probe
        self.steer_count = 0
        self.original_hidden = None
        self.agent = None
        
    def should_steer(self, current_goal):
        """Decide if we should steer based on current goal and frequency"""
        if current_goal is None:
            return False
        return random.random() < STEER_FREQ
    
    def get_steered_hidden(self, hidden_state, current_goal):
        """Edit hidden state to change goal from blue->green or green->blue"""
        if current_goal == 'blue':
            # Steer towards green
            target_goal = 'green'
            probe = self.green_probe
        elif current_goal == 'green':
            # Steer towards blue
            target_goal = 'blue'
            probe = self.blue_probe
        else:
            return hidden_state
            
        # Get the direction that increases probability of target goal
        if hasattr(probe, 'coef_'):
            steer_direction = probe.coef_[0]
        else:
            # If no probe available, use a simple perturbation
            steer_direction = np.random.randn(hidden_state.shape[0]) * 0.1
            
        # Apply steering (increase activation in target direction)
        steered_hidden = hidden_state + steer_direction * 0.5
        
        print(f"STEERING ENV: {current_goal} -> {target_goal}")
        self.steer_count += 1
        
        return steered_hidden
    
    def hook_fn(self, mod, inp, out):
        """Hook function that edits environment network hidden states during forward pass"""
        # Get the output from environment network
        if hasattr(out, 'detach'):
            hidden_state = out.detach().squeeze().cpu().numpy()
        else:
            hidden_state = out.squeeze().cpu().numpy()
        
        # Get current goal from agent
        if self.agent and hasattr(self.agent, 'sequence'):
            seq = self.agent.sequence
            if seq and len(seq) > 0:
                goal_set = seq[0][0]
                if len(goal_set) == 1:
                    assignment = next(iter(goal_set))
                    true_props = {p for p, v in assignment.assignment if v}
                    if len(true_props) == 1:
                        current_goal = next(iter(true_props))
                        
                        # Check if we should steer
                        if self.should_steer(current_goal):
                            # Store original for comparison
                            self.original_hidden = hidden_state.copy()
                            
                            # Get steered hidden state
                            steered_hidden = self.get_steered_hidden(hidden_state, current_goal)
                            
                            # Replace the output
                            new_out = torch.tensor(steered_hidden, dtype=out.dtype, device=out.device)
                            if len(out.shape) > 0:
                                new_out = new_out.reshape(out.shape)
                            
                            return new_out
        
        # Return original output if no steering
        return out

def train_env_goal_probes(model, env, sampler_fn):
    """Train probes to predict blue vs green goals from environment network"""
    print("Training environment network goal probes...")
    
    # Collect data for probing
    feats = []
    labels = []
    
    # Hook to collect features from environment network
    def collect_hook(mod, inp, out):
        if hasattr(out, 'detach'):
            arr = out.detach().squeeze().cpu().numpy()
        else:
            arr = out.squeeze().cpu().numpy()
        feats.append(arr)
    
    handle = None
    if hasattr(model, 'env_net') and hasattr(model.env_net, 'mlp'):
        handle = model.env_net.mlp.register_forward_hook(collect_hook)
    
    # Rollout to collect data
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    
    search = ExhaustiveSearch(model, set(['blue', 'green']), num_loops=2)
    agent = Agent(model, search=search, propositions=set(['blue', 'green']), verbose=False)
    agent.reset()
    
    for step in range(200):  # Shorter rollout for probe training
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Get current goal
        seq = getattr(agent, "sequence", None)
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    prop = next(iter(true_props))
                    if prop in ['blue', 'green']:
                        labels.append(1 if prop == 'blue' else 0)
                    else:
                        labels.append(-1)
                else:
                    labels.append(-1)
            else:
                labels.append(-1)
        else:
            labels.append(-1)
            
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    if handle:
        handle.remove()
    rollout_env.close()
    
    # Train probes
    X = np.array(feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X) > len(y):
        X = X[:len(y)]
    X, y = X[valid_idxs], y[valid_idxs]
    
    if len(np.unique(y)) <= 1:
        print("Warning: Only one class found for probe training")
        return None, None
    
    # Train blue probe (y=1 for blue, y=0 for green)
    from sklearn.linear_model import LogisticRegression
    blue_probe = LogisticRegression(max_iter=1000)
    blue_probe.fit(X, y)
    
    # Train green probe (y=1 for green, y=0 for blue)
    green_probe = LogisticRegression(max_iter=1000)
    green_probe.fit(X, 1 - y)  # Invert labels
    
    print(f"Environment blue probe accuracy: {blue_probe.score(X, y):.2%}")
    print(f"Environment green probe accuracy: {green_probe.score(X, 1-y):.2%}")
    
    return blue_probe, green_probe

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1) Load model and train environment network probes
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Train environment network goal probes
    blue_probe, green_probe = train_env_goal_probes(model, ENV, sampler_fn)
    if blue_probe is None:
        print("Failed to train environment network probes, exiting")
        return
    
    # 2) Create agent and steerer
    search = ExhaustiveSearch(model, set(['blue', 'green']), num_loops=2)
    agent = Agent(model, search=search, propositions=set(['blue', 'green']), verbose=False)
    
    steerer = EnvNetworkSteerer(model, blue_probe, green_probe)
    steerer.agent = agent
    
    # 3) Hook into environment network for steering
    handle = None
    if hasattr(model, 'env_net') and hasattr(model.env_net, 'mlp'):
        handle = model.env_net.mlp.register_forward_hook(steerer.hook_fn)
        print("Registered environment network steering hook")
    else:
        print("Could not find environment network MLP to hook")
        return
    
    # 4) Run steered rollout
    print(f"\nRunning environment network steered rollout (steering frequency: {STEER_FREQ})...")
    rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = rollout_env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    # Track behavior
    positions = []
    goals = []
    
    for step in trange(MAX_STEPS, desc="Environment network steered rollout"):
        # Get agent position
        if hasattr(rollout_env, 'agent_pos'):
            pos = rollout_env.agent_pos
        else:
            pos = rollout_env.unwrapped.agent_pos if hasattr(rollout_env.unwrapped, 'agent_pos') else [0, 0]
        positions.append(pos)
        
        # Get current goal
        seq = getattr(agent, "sequence", None)
        current_goal = None
        if seq and len(seq) > 0:
            goal_set = seq[0][0]
            if len(goal_set) == 1:
                assignment = next(iter(goal_set))
                true_props = {p for p, v in assignment.assignment if v}
                if len(true_props) == 1:
                    current_goal = next(iter(true_props))
        goals.append(current_goal)
        
        # Get action
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Step environment
        ret = rollout_env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        
        if done:
            break
    
    if handle:
        handle.remove()
    rollout_env.close()
    
    # 5) Analyze results
    print(f"\nEnvironment Network Steering Results:")
    print(f"Total steering interventions: {steerer.steer_count}")
    print(f"Steering rate: {steerer.steer_count/len(goals):.1%}")
    
    # Count goal changes
    goal_changes = 0
    for i in range(1, len(goals)):
        if goals[i] != goals[i-1]:
            goal_changes += 1
    
    print(f"Goal changes: {goal_changes}")
    
    # Plot trajectory
    positions = np.array(positions)
    goals = np.array(goals)
    
    plt.figure(figsize=(12, 8))
    
    # Plot trajectory with goal colors
    colors = {'blue': 'blue', 'green': 'green', None: 'gray'}
    for i in range(len(positions)-1):
        goal = goals[i]
        color = colors.get(goal, 'gray')
        plt.plot(positions[i:i+2, 0], positions[i:i+2, 1], color=color, alpha=0.7, linewidth=2)
    
    # Mark start and end
    plt.plot(positions[0, 0], positions[0, 1], 'ko', markersize=10, label='Start')
    plt.plot(positions[-1, 0], positions[-1, 1], 'ks', markersize=10, label='End')
    
    # Add zone circles (approximate)
    zone_positions = {
        'blue': [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)],
        'green': [(0.5, 0.2), (0.2, 0.5), (0.8, 0.5), (0.5, 0.8)]
    }
    
    for color, positions_list in zone_positions.items():
        for pos in positions_list:
            circle = patches.Circle(pos, 0.1, color=color, alpha=0.3, edgecolor='black')
            plt.gca().add_patch(circle)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Agent Trajectory with Environment Network Steering (Frequency: {STEER_FREQ})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    plt.savefig('env_steered_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTrajectory saved as 'env_steered_trajectory.png'")
    print(f"Environment network steering interventions: {steerer.steer_count}")

if __name__ == '__main__':
    main() 