#!/usr/bin/env python3
"""
Actor Zone Direction Probing

This script tests whether the Actor network has access to spatial goal information
by probing its ability to predict the direction toward target zones.

CRITICAL TEST: If Actor has spatial goal info, this should be TRIVIAL!
- INPUT: Combined embedding (env_embedding + ltl_embedding) 
- TARGET: Direction vector toward target zone (should be easily computable)

If this fails, it means ltl_embedding is purely symbolic (no spatial coordinates).

Usage: python interpretability/probing/probe_actor_zone_direction.py --probe-type input
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

class ActorZoneDirectionProbe:
    """Test if Actor can predict direction toward target zones."""
    
    def __init__(self, model, probe_type, target_goals=None, n_components=None):
        self.model = model
        self.probe_type = probe_type  # 'input' or 'output'
        self.target_goals = target_goals or ["FG blue", "FG red", "FG green", "FG yellow"]
        self.n_components = n_components
        self.pca = None
        
    def collect_data(self, world_ids, n_rollouts_per_world=10, max_steps=200):
        """Collect Actor activations and target zone directions."""
        print(f"Collecting ACTOR {self.probe_type.upper()} data for zone direction prediction")
        print(f"Testing goals: {self.target_goals}")
        
        all_activations = []
        all_zone_directions = []
        all_metadata = {
            'world_ids': [],
            'rollout_ids': [],
            'step_ids': [],
            'goal_types': []
        }
        
        total_samples = 0
        
        # Test each goal type
        for goal in self.target_goals:
            print(f"\n--- Testing goal: {goal} ---")
            
            env = make_env(ENV, FixedSampler.partial(goal), sequence=False, render_mode=None)
            props = set(env.get_propositions())
            agent = Agent(self.model, ExhaustiveSearch(self.model, props, num_loops=2),
                         propositions=props, verbose=False)
            
            # Extract target zone name and find its position
            target_zone_name = goal.split()[-1]  # "FG blue" -> "blue"
            
            for world_id in world_ids[:5]:  # Fewer worlds per goal to keep it manageable
                print(f"Processing world {world_id} for goal {goal}...")
                for rollout_id in trange(n_rollouts_per_world, desc=f"Rollouts for world {world_id}"):
                    # Hook to capture Actor input/output
                    captured_data = []
                    
                    def capture_actor_input_output(module, input, output):
                        if self.probe_type == 'input':
                            # Capture the input tensor (combined embedding: env + ltl)
                            data = input[0].detach().cpu().numpy().ravel()
                        else:  # 'output'
                            # Capture the output from MixedDistribution
                            if hasattr(output, 'dist'):
                                if hasattr(output.dist, 'logits'):
                                    data = output.dist.logits.detach().cpu().numpy().ravel()
                                elif hasattr(output.dist, 'probs'):
                                    data = output.dist.probs.detach().cpu().numpy().ravel()
                                else:
                                    data = output.dist.mean.detach().cpu().numpy().ravel()
                            elif hasattr(output, 'logits'):
                                data = output.logits.detach().cpu().numpy().ravel()
                            elif hasattr(output, 'probs'):
                                data = output.probs.detach().cpu().numpy().ravel()
                            else:
                                data = torch.tensor(output).detach().cpu().numpy().ravel()
                        captured_data.append(data)
                    
                    # Hook the Actor module
                    hook = self.model.actor.register_forward_hook(capture_actor_input_output)
                    
                    done = False
                    obs = env.reset(seed=world_id + rollout_id * 1000)
                    agent.reset()
                    
                    # Find target zone position in this world
                    target_zone_pos = None
                    if hasattr(env, 'zone_positions') and env.zone_positions:
                        for zone_name, zone_pos in env.zone_positions.items():
                            if target_zone_name.lower() in zone_name.lower():
                                target_zone_pos = zone_pos[:2]  # Take x,y coordinates
                                break
                    
                    if target_zone_pos is None:
                        print(f"Warning: Could not find target zone '{target_zone_name}' in world {world_id}")
                        continue
                    
                    for step_id in range(max_steps):
                        if done:
                            break
                            
                        # Store current position and compute direction to target zone
                        current_pos = env.agent_pos[:2].copy()
                        direction_to_target = target_zone_pos - current_pos
                        
                        # Normalize direction (unit vector)
                        norm = np.linalg.norm(direction_to_target)
                        if norm > 0:
                            direction_to_target = direction_to_target / norm
                        else:
                            # Agent is exactly at target - use zero vector
                            direction_to_target = np.zeros(2)
                        
                        # Get action - this triggers the hook to capture Actor activations
                        action = agent.get_action(obs, {}, deterministic=True).flatten()
                        
                        # Store the captured activation and target direction
                        if len(captured_data) > step_id:
                            activation = captured_data[step_id]
                            all_activations.append(activation)
                            all_zone_directions.append(direction_to_target)
                            all_metadata['world_ids'].append(world_id)
                            all_metadata['rollout_ids'].append(rollout_id)
                            all_metadata['step_ids'].append(step_id)
                            all_metadata['goal_types'].append(goal)
                            total_samples += 1
                        
                        # Step environment
                        obs, _, done, info = env.step(action)
                    
                    # Remove hook
                    hook.remove()
            
            env.close()
        
        if total_samples == 0:
            print("No samples collected!")
            return None
            
        print(f"Collected {total_samples} samples across {len(self.target_goals)} goals")
        
        return {
            'activations': np.array(all_activations),
            'zone_directions': np.array(all_zone_directions),
            'metadata': all_metadata
        }
    
    def create_generalization_splits(self, data):
        """Create temporal, spatial, environmental, and GOAL-based generalization splits."""
        activations = data['activations']
        zone_directions = data['zone_directions']
        world_ids = np.array(data['metadata']['world_ids'])
        rollout_ids = np.array(data['metadata']['rollout_ids'])
        step_ids = np.array(data['metadata']['step_ids'])
        goal_types = np.array(data['metadata']['goal_types'])
        
        splits = {}
        
        # 1. Temporal split (early vs late steps)
        median_step = np.median(step_ids)
        train_mask = step_ids <= median_step
        test_mask = step_ids > median_step
        
        splits['temporal'] = {
            'train': {'X': activations[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': activations[test_mask], 'y': zone_directions[test_mask]}
        }
        
        # 2. Spatial split (different rollouts within same worlds)
        unique_combinations = list(set(zip(world_ids, rollout_ids, goal_types)))
        random.shuffle(unique_combinations)
        
        split_idx = len(unique_combinations) // 2
        train_combinations = set(unique_combinations[:split_idx])
        test_combinations = set(unique_combinations[split_idx:])
        
        train_mask = np.array([
            (w, r, g) in train_combinations 
            for w, r, g in zip(world_ids, rollout_ids, goal_types)
        ])
        test_mask = ~train_mask
        
        splits['spatial'] = {
            'train': {'X': activations[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': activations[test_mask], 'y': zone_directions[test_mask]}
        }
        
        # 3. Environmental split (different worlds)
        unique_worlds = list(np.unique(world_ids))
        random.shuffle(unique_worlds)
        
        split_idx = len(unique_worlds) // 2
        train_worlds = set(unique_worlds[:split_idx])
        test_worlds = set(unique_worlds[split_idx:])
        
        train_mask = np.array([w in train_worlds for w in world_ids])
        test_mask = np.array([w in test_worlds for w in world_ids])
        
        splits['environmental'] = {
            'train': {'X': activations[train_mask], 'y': zone_directions[train_mask]},
            'test': {'X': activations[test_mask], 'y': zone_directions[test_mask]}
        }
        
        # 4. GOAL GENERALIZATION (CRITICAL TEST!)
        unique_goals = list(np.unique(goal_types))
        if len(unique_goals) > 1:
            random.shuffle(unique_goals)
            split_idx = len(unique_goals) // 2
            train_goals = set(unique_goals[:split_idx])
            test_goals = set(unique_goals[split_idx:])
            
            train_mask = np.array([g in train_goals for g in goal_types])
            test_mask = np.array([g in test_goals for g in goal_types])
            
            splits['goal'] = {
                'train': {'X': activations[train_mask], 'y': zone_directions[train_mask]},
                'test': {'X': activations[test_mask], 'y': zone_directions[test_mask]}
            }
        
        return splits
    
    def train_and_evaluate_probes(self, splits):
        """Train and evaluate zone direction prediction probes."""
        results = {}
        
        for split_name, split_data in splits.items():
            X_train, y_train = split_data['train']['X'], split_data['train']['y']
            X_test, y_test = split_data['test']['X'], split_data['test']['y']
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"  Skipping {split_name}: insufficient data")
                continue
            
            # Apply PCA if needed
            if self.n_components is not None:
                if self.pca is None:
                    self.pca = PCA(n_components=self.n_components)
                    X_train_pca = self.pca.fit_transform(X_train)
                else:
                    X_train_pca = self.pca.transform(X_train)
                X_test_pca = self.pca.transform(X_test)
            else:
                # Automatic PCA for high-dimensional data
                if X_train.shape[1] > 50:
                    if self.pca is None:
                        self.pca = PCA(n_components=0.95)  # 95% variance
                        X_train_pca = self.pca.fit_transform(X_train)
                    else:
                        X_train_pca = self.pca.transform(X_train)
                    X_test_pca = self.pca.transform(X_test)
                    n_components_used = X_train_pca.shape[1]
                else:
                    X_train_pca = X_train
                    X_test_pca = X_test
                    n_components_used = X_train.shape[1]
            
            # Train probe
            probe = Ridge(alpha=1.0)
            probe.fit(X_train_pca, y_train)
            
            # Evaluate
            y_pred = probe.predict(X_test_pca)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  {split_name}: R²={r2:.3f}, MSE={mse:.3f} (train: {len(X_train)}, test: {len(X_test)}, dims: {X_train.shape[1]}→{n_components_used})")
            
            results[split_name] = {
                'r2_test': r2,
                'mse_test': mse,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'input_dims': X_train.shape[1],
                'pca_dims': n_components_used
            }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Actor Zone Direction Probing')
    parser.add_argument('--probe-type', required=True, choices=['input', 'output'],
                       help='What to probe: input (combined embedding) or output (action logits)')
    parser.add_argument('--goals', nargs='+', default=["FG blue", "FG red", "FG green"],
                       help='LTL goals to test (e.g., "FG blue" "FG red")')
    parser.add_argument('--n-worlds', type=int, default=10,
                       help='Number of worlds to use')
    parser.add_argument('--n-rollouts', type=int, default=5,
                       help='Number of rollouts per world per goal')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum steps per rollout')
    parser.add_argument('--n-components', type=int, default=None,
                       help='Number of PCA components (None for 95% variance)')
    parser.add_argument('--output-dir', type=str, 
                       default='interpretability/probing/corrected_results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=== ACTOR ZONE DIRECTION PROBING ===")
    print(f"🎯 CRITICAL TEST: Can Actor predict direction toward target zones?")
    print(f"Probe Type: {args.probe_type.upper()}")
    print(f"Goals to test: {args.goals}")
    print(f"Worlds per goal: {args.n_worlds}")
    print(f"Rollouts per world: {args.n_rollouts}")
    print(f"Max steps: {args.max_steps}")
    
    if args.probe_type == 'input':
        print("📥 Probing: Combined embeddings (env+ltl) going INTO actor")
        print("💡 This should be TRIVIAL if Actor has spatial goal info!")
    else:
        print("📤 Probing: Action logits/probabilities coming OUT of actor")
        print("💡 This tests if actions encode directional intent!")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, args.seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Create probe
    probe = ActorZoneDirectionProbe(model, args.probe_type, args.goals, args.n_components)
    
    # Collect data
    world_ids = list(range(args.n_worlds))
    data = probe.collect_data(world_ids, args.n_rollouts, args.max_steps)
    
    if data is None:
        print("No data collected. Exiting.")
        return
    
    # Create generalization splits
    print("\nCreating generalization splits...")
    splits = probe.create_generalization_splits(data)
    
    # Train and evaluate probes
    print("\nTraining and evaluating zone direction probes...")
    results = probe.train_and_evaluate_probes(splits)
    
    # Save results
    print("\nSaving results...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save CSV
    csv_data = []
    for split_name, metrics in results.items():
        csv_data.append({
            'timestamp': timestamp,
            'probe_type': f'actor_zone_direction_{args.probe_type}',
            'goals': '|'.join(args.goals),
            'split_type': split_name,
            **metrics
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f"{args.output_dir}/actor_zone_direction_{args.probe_type}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to: {csv_path}")
    
    print(f"\n=== ACTOR ZONE DIRECTION PROBE SUMMARY ===")
    for split_name, metrics in results.items():
        print(f"{split_name}: R² = {metrics['r2_test']:.4f}, MSE = {metrics['mse_test']:.4f}")
    
    print(f"\n🎯 INTERPRETATION:")
    goal_r2 = results.get('goal', {}).get('r2_test', 'N/A')
    temporal_r2 = results.get('temporal', {}).get('r2_test', 'N/A')
    
    if isinstance(goal_r2, float) and goal_r2 > 0.8:
        print("✅ EXCELLENT goal generalization - Actor has spatial goal information!")
    elif isinstance(goal_r2, float) and goal_r2 > 0.3:
        print("🤔 MODERATE goal generalization - Actor has some spatial goal info")
    elif isinstance(goal_r2, float) and goal_r2 < 0.1:
        print("❌ POOR goal generalization - ltl_embedding likely purely symbolic")
    
    if isinstance(temporal_r2, float) and temporal_r2 > 0.5:
        print("✅ Good temporal performance - Actor can predict zone directions")
    elif isinstance(temporal_r2, float) and temporal_r2 < 0.1:
        print("❌ Poor temporal performance - Actor doesn't encode zone directions")

if __name__ == "__main__":
    main() 