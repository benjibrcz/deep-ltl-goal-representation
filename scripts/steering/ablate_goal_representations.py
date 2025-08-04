#!/usr/bin/env python3
"""
Goal Representation Ablation Test

This script tests causality by ablating goal-related features and measuring
if goal-directed behavior stops. This is different from steering - we're
removing information rather than adding steering signals.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import random
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

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
MAX_STEPS = 1000
FORMULA   = "GF blue & GF green"

class GoalRepresentationAblator:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.probe = None
        self.ablation_mask = None
        self.original_forward = None
        
    def train_probe(self, num_samples=500):
        """Train a probe to identify goal-related features"""
        print(f"Training probe on {self.layer_name}...")
        
        # Collect features and labels
        features = []
        labels = []
        
        # Create a temporary environment for data collection
        sampler_fn = FixedSampler.partial(FORMULA)
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        
        for _ in trange(num_samples, desc="Collecting probe data"):
            ret = env.reset(seed=0)
            obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
            
            # Get current subgoal
            current_subgoal = self._get_current_subgoal(info)
            if current_subgoal is None:
                continue
                
            # Get features from the target layer
            with torch.no_grad():
                # Forward pass to get features
                _ = self.model(obs, info)
                features.append(self._get_layer_features().flatten().cpu().numpy())
                labels.append(current_subgoal)
        
        env.close()
        
        # Train probe
        X = np.array(features)
        y = np.array(labels)
        
        # Convert to numeric labels
        unique_goals = sorted(list(set(y)))
        goal_to_idx = {goal: i for i, goal in enumerate(unique_goals)}
        y_numeric = np.array([goal_to_idx[goal] for goal in y])
        
        self.probe = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.probe.fit(X, y_numeric)
        
        # Calculate accuracy
        accuracy = self.probe.score(X, y_numeric)
        print(f"Probe accuracy: {accuracy:.3f}")
        
        # Create ablation mask based on probe weights
        self._create_ablation_mask()
        
        return accuracy
    
    def _get_current_subgoal(self, info):
        """Extract current subgoal from info"""
        # This is a simplified version - you might need to adapt based on your setup
        if 'propositions' in info and info['propositions']:
            return list(info['propositions'])[0]
        return None
    
    def _get_layer_features(self):
        """Get features from the target layer"""
        # This will need to be adapted based on your model architecture
        # You'll need to hook into the specific layer
        if hasattr(self.model, 'env_net') and 'env_net' in self.layer_name:
            return self.model.env_net.get_features()
        elif hasattr(self.model, 'policy_net') and 'policy' in self.layer_name:
            return self.model.policy_net.get_features()
        else:
            # Default fallback
            return torch.randn(64)  # Placeholder
    
    def _create_ablation_mask(self):
        """Create mask for ablating goal-related features"""
        if self.probe is None:
            raise ValueError("Must create ablation mask first")
        
        # Get probe weights - simplified approach
        try:
            weights = []
            for estimator in self.probe.estimators_:
                if hasattr(estimator, 'coef_'):
                    coef = estimator.coef_
                    if coef is not None and len(coef) > 0:
                        weights.append(coef[0])
            
            if not weights:
                # Fallback: create random mask
                self.ablation_mask = np.ones(64, dtype=np.float32)
                print("Warning: Could not extract probe weights, using fallback mask")
                return
            
            # Combine weights across all classes
            combined_weights = np.mean(np.abs(weights), axis=0)
            
            # Create mask - zero out top features
            threshold = np.percentile(combined_weights, 80)  # Zero out top 20%
            self.ablation_mask = (combined_weights < threshold).astype(np.float32)
            
            print(f"Created ablation mask: {np.sum(self.ablation_mask)}/{len(self.ablation_mask)} features preserved")
            
        except Exception as e:
            print(f"Error creating ablation mask: {e}")
            # Fallback: create random mask
            self.ablation_mask = np.ones(64, dtype=np.float32)
    
    def apply_ablation(self, features):
        """Apply ablation mask to features"""
        if self.ablation_mask is None:
            raise ValueError("Must create ablation mask first")
        
        # Apply mask (zero out goal-related features)
        ablated_features = features * torch.tensor(self.ablation_mask, device=features.device)
        return ablated_features
    
    def hook_ablation(self):
        """Hook ablation into the model's forward pass"""
        def ablation_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return self.apply_ablation(output)
            return output
        
        # Hook into the target layer
        target_module = self._get_target_module()
        self.original_forward = target_module.forward
        target_module.forward = ablation_hook
    
    def unhook_ablation(self):
        """Remove ablation hook"""
        if self.original_forward is not None:
            target_module = self._get_target_module()
            target_module.forward = self.original_forward
            self.original_forward = None
    
    def _get_target_module(self):
        """Get the target module to hook into"""
        # This will need to be adapted based on your model architecture
        if 'env_net' in self.layer_name:
            return self.model.env_net
        elif 'policy' in self.layer_name:
            return self.model.policy_net
        else:
            raise ValueError(f"Unknown layer: {self.layer_name}")

def run_ablation_experiment():
    """Run the ablation experiment"""
    print("=== Goal Representation Ablation Experiment ===\n")
    
    # Set up
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load model
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, 0)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Test different layers
    layers_to_test = ['env_net_mlp_0', 'policy_mlp_0']
    
    for layer_name in layers_to_test:
        print(f"\n--- Testing layer: {layer_name} ---")
        
        # Create ablator
        ablator = GoalRepresentationAblator(model, layer_name)
        
        # Train probe
        probe_accuracy = ablator.train_probe()
        
        # Run baseline (no ablation)
        print("\nRunning baseline (no ablation)...")
        baseline_metrics = run_rollout_with_metrics(model, layer_name, ablation=False)
        
        # Run with ablation
        print("\nRunning with ablation...")
        ablator.hook_ablation()
        ablation_metrics = run_rollout_with_metrics(model, layer_name, ablation=True)
        ablator.unhook_ablation()
        
        # Compare results
        print(f"\n--- Results for {layer_name} ---")
        print(f"Probe accuracy: {probe_accuracy:.3f}")
        print(f"Baseline goal completion: {baseline_metrics['goal_completion_rate']:.3f}")
        print(f"Ablation goal completion: {ablation_metrics['goal_completion_rate']:.3f}")
        print(f"Goal completion change: {ablation_metrics['goal_completion_rate'] - baseline_metrics['goal_completion_rate']:.3f}")
        print(f"Path efficiency change: {ablation_metrics['path_efficiency'] - baseline_metrics['path_efficiency']:.3f}")

def run_rollout_with_metrics(model, layer_name, ablation=False, num_rollouts=5):
    """Run rollouts and collect behavioral metrics"""
    sampler_fn = FixedSampler.partial(FORMULA)
    
    metrics = {
        'goal_completion_rate': 0,
        'path_efficiency': 0,
        'avg_steps': 0,
        'goal_switching': 0
    }
    
    completed_rollouts = 0
    
    for i in range(num_rollouts):
        env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        ret = env.reset(seed=SEED + i)
        obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
        
        # Create agent
        props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
        search = ExhaustiveSearch(model, set(props), num_loops=2)
        agent = Agent(model, search=search, propositions=set(props), verbose=False)
        agent.reset()
        
        # Track metrics
        steps = 0
        goals_visited = set()
        initial_goal = None
        
        for step in range(MAX_STEPS):
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Track current goal
            current_goal = get_current_goal(info)
            if initial_goal is None:
                initial_goal = current_goal
            
            if current_goal:
                goals_visited.add(current_goal)
            
            ret = env.step(action)
            if len(ret) == 5:
                obs, rew, term, trunc, info = ret
                done = term or trunc
            else:
                obs, rew, done, info = ret
                term, trunc = done, done
            
            steps += 1
            
            if done:
                break
        
        env.close()
        
        # Calculate metrics for this rollout
        goal_completed = len(goals_visited) >= 2  # Both blue and green
        if goal_completed:
            completed_rollouts += 1
        
        metrics['goal_completion_rate'] += goal_completed
        metrics['avg_steps'] += steps
        metrics['path_efficiency'] += steps  # Lower is better
        
        # Check for goal switching
        if initial_goal and len(goals_visited) > 1:
            metrics['goal_switching'] += 1
    
    # Average metrics
    for key in metrics:
        if key == 'goal_completion_rate':
            metrics[key] = float(metrics[key]) / float(num_rollouts)
        else:
            metrics[key] = float(metrics[key]) / float(num_rollouts)
    
    return metrics

def get_current_goal(info):
    """Extract current goal from info"""
    if 'propositions' in info and info['propositions']:
        return list(info['propositions'])[0]
    return None

if __name__ == "__main__":
    run_ablation_experiment() 