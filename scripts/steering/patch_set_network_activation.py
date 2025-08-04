#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent
from visualize.zones   import draw_trajectories

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
MAX_STEPS = 700
N_ROLLOUTS = 20
COLLECT_FORMULA = "FG yellow"
PATCH_FORMULA = "FG blue"
PATCH_START_STEP = 200  # Start patching from this step onwards

# Create results directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def collect_set_network_activations(model, formula, n_rollouts, max_steps=MAX_STEPS):
    """Collect activations from all steps of each rollout"""
    sampler_fn = FixedSampler.partial(formula)
    all_activations = []  # List of rollouts, each containing step activations
    
    def hook_fn(module, input, output):
        arr = output.detach().cpu().numpy()
        # Always sum over the set dimension (axis=1) to get (1, 16)
        if len(arr.shape) == 3:
            arr = arr.sum(axis=1)
        # Get the current rollout's activations list
        current_rollout_activations = all_activations[-1]
        current_rollout_activations.append(arr)
    
    handle = model.ltl_net.set_network.register_forward_hook(hook_fn)
    
    for i in range(n_rollouts):
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        ret = env.reset(seed=SEED + i)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        props = set(env.get_propositions())
        search = ExhaustiveSearch(model, props, num_loops=2)
        agent = Agent(model, search=search, propositions=props, verbose=False)
        agent.reset()
        
        # Initialize activations list for this rollout
        rollout_activations = []
        all_activations.append(rollout_activations)
        
        # Run the full rollout to collect activations from all steps
        for step in range(max_steps):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            ret = env.step(action)  # Use the actual action from the agent
            if isinstance(ret, tuple) and len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            elif isinstance(ret, tuple) and len(ret) == 4:
                obs, rew, done, info = ret
                term, trunc = done, done
            else:
                # Handle unexpected return format
                obs, rew, done, info = ret, 0, False, {}
                term, trunc = done, done
            if done:
                break
        
        env.close()
    
    handle.remove()
    
    # Convert to numpy arrays and pad shorter rollouts
    max_rollout_length = max(len(rollout) for rollout in all_activations)
    padded_activations = []
    
    for rollout in all_activations:
        # Pad with the last activation if rollout is shorter
        while len(rollout) < max_rollout_length:
            rollout.append(rollout[-1] if rollout else np.zeros((1, 16)))
        padded_activations.append(np.stack(rollout))
    
    # Stack all rollouts: shape = (n_rollouts, max_steps, 1, 16)
    activations = np.stack(padded_activations)
    print(f"Collected activations shape: {activations.shape} for formula '{formula}'")
    print(f"  - {n_rollouts} rollouts")
    print(f"  - {max_rollout_length} steps per rollout")
    print(f"  - {activations.shape[-1]} activation dimensions")
    
    return activations


def patch_set_network(model, step_activations):
    """Patch the set_network output with step-corresponding activations, starting from PATCH_START_STEP"""
    class PatchSetNetwork:
        def __init__(self, step_activations):
            # step_activations shape: (n_rollouts, max_steps, 1, 16)
            # We'll use the mean across rollouts for each step
            self.step_activations = step_activations
            self.current_step = 0
            print(f"Step activations shape: {step_activations.shape}")
            print(f"Patching will start from step {PATCH_START_STEP}")
            
        def hook_fn(self, module, input, output):
            # Only patch if we've reached the start step
            if self.current_step >= PATCH_START_STEP:
                print(f"Step {self.current_step}: Patching set_network output with activation from '{COLLECT_FORMULA}'")
                
                # Get the mean activation for the current step across all rollouts
                # Adjust step index to account for the delay
                adjusted_step = self.current_step - PATCH_START_STEP
                if adjusted_step < self.step_activations.shape[1]:
                    step_activation = np.mean(self.step_activations[:, adjusted_step, :, :], axis=0)  # Average across rollouts
                    step_activation = torch.tensor(step_activation, dtype=torch.float32)
                else:
                    # If we exceed the collected steps, use the last step's activation
                    step_activation = torch.tensor(
                        np.mean(self.step_activations[:, -1, :, :], axis=0), 
                        dtype=torch.float32
                    )
                    print(f"  Using last step activation (adjusted step {adjusted_step} > {self.step_activations.shape[1]})")
                
                # Match the output shape
                batch_size, set_size, embedding_dim = output.shape
                patched = step_activation.unsqueeze(0).expand(batch_size, -1, -1)
                if set_size != patched.shape[1]:
                    patched = patched.repeat(1, set_size, 1)
                patched = patched[:, :set_size, :]
                
                print(f"  Patched output shape: {patched.shape}")
                self.current_step += 1
                return patched
            else:
                # Before the start step, return the original output
                print(f"Step {self.current_step}: No patching yet (start at step {PATCH_START_STEP})")
                self.current_step += 1
                return output
            
    patcher = PatchSetNetwork(step_activations)
    return model.ltl_net.set_network.register_forward_hook(patcher.hook_fn)


def run_unpatched_rollout(model, formula):
    """Run a rollout without any patching for comparison"""
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = env.reset(seed=SEED + 1000)  # Use same seed for fair comparison
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    print(f"[DEBUG] Running unpatched rollout:")
    print(f"  Formula: {formula}")
    print(f"  Environment: {ENV}")
    print(f"  Agent: {agent}")
    positions = []
    raw_obs = []
    for step in range(MAX_STEPS):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Get agent position the same way as steer_subgoals.py
        agent_pos = getattr(env, 'agent', None)
        if agent_pos is not None and hasattr(agent_pos, 'pos'):
            pos = np.array(agent_pos.pos)
        elif hasattr(env, 'agent_pos'):
            pos = np.array(env.agent_pos)
        else:
            pos = None
            
        positions.append(pos)
        raw_obs.append(obs)
        if step < 10:
            print(f"  Step {step}: pos={pos}, obs_type={type(obs)}, obs_keys={list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
            if isinstance(obs, dict) and 'features' in obs:
                print(f"    features[:5]={obs['features'][:5]}")
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        if done:
            break
    
    # Get zone positions before closing the environment
    zone_positions = getattr(env, 'zone_positions', {})
    env.close()
    print(f"Unpatched rollout for formula '{formula}' completed. Trajectory length: {len(positions)}")
    print(f"First 10 positions: {positions[:10]}")
    print(f"Any None positions in first 10? {any(p is None for p in positions[:10])}")
    return positions, zone_positions


def run_patched_rollout(model, formula, patch_activation):
    sampler_fn = FixedSampler.partial(formula)
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    ret = env.reset(seed=SEED + 1000)  # Use a different seed for test
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    agent.reset()
    handle = patch_set_network(model, patch_activation)
    positions = []
    for step in range(MAX_STEPS):
        action = agent.get_action(obs, info, deterministic=True).flatten()
        
        # Get agent position the same way as steer_subgoals.py
        agent_pos = getattr(env, 'agent', None)
        if agent_pos is not None and hasattr(agent_pos, 'pos'):
            pos = np.array(agent_pos.pos)
        elif hasattr(env, 'agent_pos'):
            pos = np.array(env.agent_pos)
        else:
            pos = None
            
        positions.append(pos)
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
            term, trunc = done, done
        if done:
            break
    
    # Get zone positions before closing the environment
    zone_positions = getattr(env, 'zone_positions', {})
    handle.remove()
    env.close()
    print(f"Patched rollout for formula '{formula}' completed. Trajectory length: {len(positions)}")
    
    # Plot and save the trajectory using existing visualization tools
    if positions and any(pos is not None for pos in positions):
        # Filter out None positions
        valid_positions = [pos for pos in positions if pos is not None]
        if valid_positions:
            # Create trajectory plot using existing function
            fig = draw_trajectories([zone_positions], [valid_positions], 1, 1)
            filename = f"patched_trajectory_{COLLECT_FORMULA.replace(' ', '_')}_to_{PATCH_FORMULA.replace(' ', '_')}_start_step_{PATCH_START_STEP}.png"
            fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {os.path.join(RESULTS_DIR, filename)}")
    
    return positions, zone_positions


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    sampler_fn = FixedSampler.partial(COLLECT_FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()

    # 1. Collect step-corresponding activations for COLLECT_FORMULA
    step_activations = collect_set_network_activations(model, COLLECT_FORMULA, N_ROLLOUTS)
    print(f"Step activations shape: {step_activations.shape}")

    # 2. Patch step-corresponding activations into PATCH_FORMULA rollout
    run_patched_rollout(model, PATCH_FORMULA, step_activations)

    # 3. Run unpatched rollout for comparison
    unpatched_positions, unpatched_zone_positions = run_unpatched_rollout(model, PATCH_FORMULA)

    # Plot and save the unpatched trajectory
    if unpatched_positions and any(pos is not None for pos in unpatched_positions):
        # Filter out None positions
        valid_unpatched_positions = [pos for pos in unpatched_positions if pos is not None]
        if valid_unpatched_positions:
            # Create trajectory plot using existing function
            fig = draw_trajectories([unpatched_zone_positions], [valid_unpatched_positions], 1, 1)
            filename = f"unpatched_trajectory_{COLLECT_FORMULA.replace(' ', '_')}.png"
            fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
            print(f"Unpatched trajectory plot saved to: {os.path.join(RESULTS_DIR, filename)}")

if __name__ == '__main__':
    main() 