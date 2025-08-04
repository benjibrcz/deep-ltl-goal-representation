#!/usr/bin/env python3
"""
Visualize Grid Probe Methodology

Verify our grid probing approach by visualizing:
1. Actual agent trajectories
2. Ground truth labels (which cells get visited)  
3. Probe predictions for each cell
4. Train vs test set comparison
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import trange
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from sequence.search import ExhaustiveSearch
from model.agent import Agent

ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0

def position_to_grid_cell(pos, grid_size, map_bounds):
    x, y = pos
    x_min, y_min, x_max, y_max = map_bounds
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    grid_x = int(np.clip(x_norm * grid_size, 0, grid_size - 1))
    grid_y = int(np.clip(y_norm * grid_size, 0, grid_size - 1))
    return grid_x, grid_y

def collect_visualization_data(model, n_trajectories=3, max_steps=30, horizon=5):
    """Collect data specifically for visualization."""
    print(f"🎨 COLLECTING VISUALIZATION DATA")
    print(f"Trajectories: {n_trajectories}, Steps: {max_steps}, Horizon: {horizon}")
    
    env = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2), propositions=props, verbose=False)
    
    grid_size = 5
    map_bounds = (-2, -2, 2, 2)
    
    trajectory_data = []
    samples = []
    
    for traj_id in range(n_trajectories):
        print(f"Collecting trajectory {traj_id}...")
        
        done = False
        obs = env.reset(seed=traj_id * 100)
        agent.reset()
        
        traj_positions = []
        traj_features = []
        
        for step_id in range(max_steps):
            if done:
                break
                
            current_pos = env.agent_pos[:2].copy()
            traj_positions.append(current_pos)
            
            # Get features
            obs_features = obs.get('features', np.zeros(80))
            goal_encoding = np.zeros(10)
            goal_encoding[0] = 1.0  # blue
            raw_features = np.concatenate([obs_features, goal_encoding])
            traj_features.append(raw_features)
            
            action = agent.get_action(obs, {}, deterministic=True).flatten()
            obs, _, done, info = env.step(action)
        
        trajectory_data.append({
            'positions': traj_positions,
            'features': traj_features,
            'trajectory_id': traj_id
        })
        
        # Create samples from this trajectory
        if len(traj_positions) >= horizon + 2:
            for i in range(len(traj_positions) - horizon):
                current_pos = traj_positions[i]
                current_features = traj_features[i]
                
                # Get future positions for next horizon steps
                future_positions = traj_positions[i+1:i+1+horizon]
                
                # Create grid visits set
                grid_visits = set()
                for future_pos in future_positions:
                    gx, gy = position_to_grid_cell(future_pos, grid_size, map_bounds)
                    grid_visits.add((gx, gy))
                
                samples.append({
                    'current_pos': current_pos,
                    'features': current_features,
                    'grid_visits': grid_visits,
                    'future_positions': future_positions,
                    'trajectory_id': traj_id,
                    'step_id': i
                })
    
    env.close()
    print(f"Collected {len(trajectory_data)} trajectories, {len(samples)} samples")
    return trajectory_data, samples, grid_size, map_bounds

def train_grid_probes(samples, grid_size):
    """Train binary probes for each grid cell."""
    print(f"🧠 TRAINING GRID PROBES")
    
    # Extract features
    features = np.array([s['features'] for s in samples])
    print(f"Features shape: {features.shape}")
    
    # Apply PCA
    pca = PCA(n_components=min(10, features.shape[1]))
    features_pca = pca.fit_transform(features)
    print(f"PCA: {features.shape[1]} → {features_pca.shape[1]} dimensions")
    
    # Train/test split
    n_train = int(0.7 * len(samples))
    indices = list(range(len(samples)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = features_pca[train_indices]
    X_test = features_pca[test_indices]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train probe for each cell
    probes = {}
    cell_stats = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Create labels for this cell
            y_all = [1 if (i, j) in s['grid_visits'] else 0 for s in samples]
            y_train = [y_all[idx] for idx in train_indices]
            y_test = [y_all[idx] for idx in test_indices]
            
            positive_ratio = sum(y_all) / len(y_all)
            cell_stats[(i, j)] = {
                'positive_ratio': positive_ratio,
                'total_positive': sum(y_all),
                'total_samples': len(y_all)
            }
            
            # Train probe if we have both classes
            if len(set(y_train)) >= 2 and positive_ratio >= 0.02:
                try:
                    probe = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                    probe.fit(X_train, y_train)
                    
                    # Get predictions
                    train_pred = probe.predict(X_train)
                    test_pred = probe.predict(X_test)
                    train_proba = probe.predict_proba(X_train)[:, 1] if len(set(y_train)) > 1 else np.zeros(len(y_train))
                    test_proba = probe.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else np.zeros(len(y_test))
                    
                    probes[(i, j)] = {
                        'model': probe,
                        'train_pred': train_pred,
                        'test_pred': test_pred,
                        'train_proba': train_proba,
                        'test_proba': test_proba,
                        'train_labels': y_train,
                        'test_labels': y_test,
                        'status': 'trained'
                    }
                    
                except Exception as e:
                    probes[(i, j)] = {'status': 'failed', 'error': str(e)}
            else:
                probes[(i, j)] = {'status': 'insufficient_data'}
    
    return probes, cell_stats, train_indices, test_indices

def visualize_methodology(trajectory_data, samples, probes, cell_stats, train_indices, test_indices, grid_size, map_bounds):
    """Create comprehensive visualization of our methodology."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Actual trajectories
    ax1 = plt.subplot(2, 4, 1)
    colors = ['red', 'blue', 'green']
    for i, traj in enumerate(trajectory_data):
        positions = np.array(traj['positions'])
        ax1.plot(positions[:, 0], positions[:, 1], 'o-', color=colors[i % len(colors)], 
                label=f'Trajectory {i}', alpha=0.7, markersize=4)
    
    ax1.set_xlim(map_bounds[0], map_bounds[2])
    ax1.set_ylim(map_bounds[1], map_bounds[3])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Actual Agent Trajectories', fontweight='bold')
    ax1.legend()
    
    # Plot 2: Grid overlay
    ax2 = plt.subplot(2, 4, 2)
    # Draw grid lines
    x_min, y_min, x_max, y_max = map_bounds
    for i in range(grid_size + 1):
        x = x_min + i * (x_max - x_min) / grid_size
        ax2.axvline(x, color='gray', alpha=0.5, linewidth=1)
    for j in range(grid_size + 1):
        y = y_min + j * (y_max - y_min) / grid_size
        ax2.axhline(y, color='gray', alpha=0.5, linewidth=1)
    
    # Plot trajectories with grid
    for i, traj in enumerate(trajectory_data):
        positions = np.array(traj['positions'])
        ax2.plot(positions[:, 0], positions[:, 1], 'o-', color=colors[i % len(colors)], 
                alpha=0.7, markersize=3)
    
    ax2.set_xlim(map_bounds[0], map_bounds[2])
    ax2.set_ylim(map_bounds[1], map_bounds[3])
    ax2.set_title('Trajectories on Grid', fontweight='bold')
    
    # Plot 3: Visit frequency heatmap
    ax3 = plt.subplot(2, 4, 3)
    visit_matrix = np.zeros((grid_size, grid_size))
    for (i, j), stats in cell_stats.items():
        visit_matrix[j, i] = stats['positive_ratio']  # Note: j,i for correct orientation
    
    im3 = ax3.imshow(visit_matrix, cmap='Reds', interpolation='nearest')
    ax3.set_title('Cell Visit Frequency\n(Ground Truth)', fontweight='bold')
    
    # Add text annotations
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax3.text(i, j, f'{visit_matrix[j, i]:.2f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot 4: Example sample with ground truth
    ax4 = plt.subplot(2, 4, 4)
    if len(samples) > 0:
        # Take a sample from train set
        sample_idx = train_indices[len(train_indices)//2] if len(train_indices) > 0 else 0
        sample = samples[sample_idx]
        
        # Draw grid
        for i in range(grid_size + 1):
            x = x_min + i * (x_max - x_min) / grid_size
            ax4.axvline(x, color='gray', alpha=0.5, linewidth=1)
        for j in range(grid_size + 1):
            y = y_min + j * (y_max - y_min) / grid_size
            ax4.axhline(y, color='gray', alpha=0.5, linewidth=1)
        
        # Current position
        ax4.plot(sample['current_pos'][0], sample['current_pos'][1], 
                'ro', markersize=10, label='Current')
        
        # Future positions
        future_pos = np.array(sample['future_positions'])
        ax4.plot(future_pos[:, 0], future_pos[:, 1], 
                'b-', linewidth=2, alpha=0.7, label='Future path')
        ax4.plot(future_pos[:, 0], future_pos[:, 1], 
                'bo', markersize=6)
        
        # Highlight visited cells
        for (i, j) in sample['grid_visits']:
            cell_x = x_min + (i + 0.5) * (x_max - x_min) / grid_size
            cell_y = y_min + (j + 0.5) * (y_max - y_min) / grid_size
            ax4.add_patch(patches.Rectangle((x_min + i * (x_max - x_min) / grid_size,
                                       y_min + j * (y_max - y_min) / grid_size),
                                      (x_max - x_min) / grid_size,
                                      (y_max - y_min) / grid_size,
                                      facecolor='yellow', alpha=0.3))
        
        ax4.set_xlim(map_bounds[0], map_bounds[2])
        ax4.set_ylim(map_bounds[1], map_bounds[3])
        ax4.set_title(f'Example: Ground Truth Labels\n(Sample {sample_idx})', fontweight='bold')
        ax4.legend()
    
    # Plot 5-8: Probe predictions for train and test
    trained_cells = [(i, j) for (i, j) in probes.keys() if probes[(i, j)].get('status') == 'trained']
    
    if len(trained_cells) > 0:
        # Plot 5: Train predictions heatmap
        ax5 = plt.subplot(2, 4, 5)
        train_pred_matrix = np.zeros((grid_size, grid_size))
        for (i, j) in trained_cells:
            probe = probes[(i, j)]
            avg_train_proba = np.mean(probe['train_proba'])
            train_pred_matrix[j, i] = avg_train_proba
        
        im5 = ax5.imshow(train_pred_matrix, cmap='Blues', interpolation='nearest')
        ax5.set_title('Average Train Predictions', fontweight='bold')
        
        # Add text annotations
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in trained_cells:
                    text = ax5.text(i, j, f'{train_pred_matrix[j, i]:.2f}', 
                                   ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        
        # Plot 6: Test predictions heatmap
        ax6 = plt.subplot(2, 4, 6)
        test_pred_matrix = np.zeros((grid_size, grid_size))
        for (i, j) in trained_cells:
            probe = probes[(i, j)]
            avg_test_proba = np.mean(probe['test_proba'])
            test_pred_matrix[j, i] = avg_test_proba
        
        im6 = ax6.imshow(test_pred_matrix, cmap='Greens', interpolation='nearest')
        ax6.set_title('Average Test Predictions', fontweight='bold')
        
        # Add text annotations
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in trained_cells:
                    text = ax6.text(i, j, f'{test_pred_matrix[j, i]:.2f}', 
                                   ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        
        # Plot 7: Performance comparison
        ax7 = plt.subplot(2, 4, 7)
        f1_scores = []
        cell_labels = []
        
        for (i, j) in trained_cells:
            probe = probes[(i, j)]
            try:
                f1 = f1_score(probe['test_labels'], probe['test_pred'], zero_division='0.0')
                f1_scores.append(f1)
                cell_labels.append(f"({i},{j})")
            except:
                f1_scores.append(0.0)
                cell_labels.append(f"({i},{j})")
        
        if len(f1_scores) > 0:
            bars = ax7.bar(range(len(f1_scores)), f1_scores, color='purple', alpha=0.7)
            ax7.set_xticks(range(len(cell_labels)))
            ax7.set_xticklabels(cell_labels, rotation=45)
            ax7.set_ylabel('F1 Score')
            ax7.set_title('F1 Scores by Cell', fontweight='bold')
            ax7.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, f1 in zip(bars, f1_scores):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 8: Methodology summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""
METHODOLOGY VERIFICATION:

✅ Binary Classification per Cell:
   - {grid_size}×{grid_size} = {grid_size*grid_size} separate probes
   - Each predicts: "Will cell be visited?"

✅ Labels:
   - Label=1: Cell visited in next 5 steps
   - Label=0: Cell NOT visited

✅ Class Balance:
   - Avg positive ratio: {np.mean([stats['positive_ratio'] for stats in cell_stats.values()]):.3f}
   - This explains poor F1 scores!

✅ Data Split:
   - Train samples: {len(train_indices)}
   - Test samples: {len(test_indices)}

✅ Probes Trained:
   - Successfully: {len([p for p in probes.values() if p.get('status') == 'trained'])}
   - Failed/Insufficient: {len([p for p in probes.values() if p.get('status') != 'trained'])}
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('interpretability/probing/corrected_results/methodology_verification.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Methodology visualization saved to corrected_results/methodology_verification.png")

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== GRID PROBE METHODOLOGY VERIFICATION ===")
    print("🎨 Visualizing: trajectories, labels, predictions, and performance")
    
    # Load model
    print("\nLoading model...")
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg = model_configs[ENV]
    dummy = make_env(ENV, FixedSampler.partial("FG blue"), sequence=False, render_mode=None)
    model = build_model(dummy, status, cfg).eval()
    dummy.close()
    
    # Collect visualization data
    trajectory_data, samples, grid_size, map_bounds = collect_visualization_data(model)
    
    if len(samples) == 0:
        print("❌ No samples collected!")
        return
    
    # Train probes
    probes, cell_stats, train_indices, test_indices = train_grid_probes(samples, grid_size)
    
    # Create visualization
    visualize_methodology(trajectory_data, samples, probes, cell_stats, 
                         train_indices, test_indices, grid_size, map_bounds)
    
    print(f"\n✅ METHODOLOGY VERIFICATION COMPLETE")
    print(f"Review the visualization to confirm our approach is correct!")

if __name__ == "__main__":
    main() 