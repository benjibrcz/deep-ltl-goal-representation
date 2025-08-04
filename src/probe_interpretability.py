#!/usr/bin/env python3
import os
import sys
import random
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))

from utils.model_store import ModelStore
from model.model       import build_model
from config            import model_configs
from ltl               import FixedSampler
from envs              import make_env
from sequence.search   import ExhaustiveSearch
from model.agent       import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 0
FORMULA = "GF blue & GF green"
MAX_STEPS = 1000

def analyze_probe_weights(clf, feature_names=None, class_names=None):
    """Analyze the learned probe weights"""
    print("=== Probe Weight Analysis ===")
    
    # Get coefficients
    coef = clf.coef_[0]  # For binary classification
    intercept = clf.intercept_[0]
    
    print(f"Intercept: {intercept:.4f}")
    print(f"Number of features: {len(coef)}")
    print(f"Mean absolute weight: {np.mean(np.abs(coef)):.4f}")
    print(f"Max absolute weight: {np.max(np.abs(coef)):.4f}")
    print(f"Weight norm: {np.linalg.norm(coef):.4f}")
    
    # Feature importance analysis
    abs_weights = np.abs(coef)
    sorted_indices = np.argsort(abs_weights)[::-1]
    
    print(f"\nTop 10 most important features:")
    for i, idx in enumerate(sorted_indices[:10]):
        weight = coef[idx]
        abs_weight = abs_weights[idx]
        feature_name = f"Feature_{idx}" if feature_names is None else feature_names[idx]
        print(f"  {i+1:2d}. {feature_name:<15} | weight={weight:8.4f} | abs={abs_weight:8.4f}")
    
    # Weight distribution analysis
    print(f"\nWeight distribution:")
    print(f"  Positive weights: {np.sum(coef > 0)} ({np.sum(coef > 0)/len(coef)*100:.1f}%)")
    print(f"  Negative weights: {np.sum(coef < 0)} ({np.sum(coef < 0)/len(coef)*100:.1f}%)")
    print(f"  Zero weights: {np.sum(coef == 0)} ({np.sum(coef == 0)/len(coef)*100:.1f}%)")
    
    # Sparsity analysis
    threshold = np.percentile(abs_weights, 90)  # Top 10% weights
    sparse_weights = abs_weights > threshold
    print(f"\nSparsity analysis (top 10% threshold = {threshold:.4f}):")
    print(f"  Sparse features: {np.sum(sparse_weights)} ({np.sum(sparse_weights)/len(coef)*100:.1f}%)")
    
    return {
        'coef': coef,
        'intercept': intercept,
        'abs_weights': abs_weights,
        'sorted_indices': sorted_indices,
        'sparse_features': sparse_weights,
        'weight_norm': np.linalg.norm(coef)
    }

def visualize_probe_weights(weight_analysis, save_path=None):
    """Create visualizations of probe weights"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    coef = weight_analysis['coef']
    abs_weights = weight_analysis['abs_weights']
    sorted_indices = weight_analysis['sorted_indices']
    
    # 1. Weight distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(coef, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('Distribution of Probe Weights')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Top weights bar plot
    ax2 = axes[0, 1]
    top_n = 20
    top_weights = coef[sorted_indices[:top_n]]
    top_indices = sorted_indices[:top_n]
    
    colors = ['red' if w < 0 else 'blue' for w in top_weights]
    bars = ax2.bar(range(top_n), top_weights, color=colors, alpha=0.7)
    ax2.set_title(f'Top {top_n} Most Important Features')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Weight Value')
    ax2.set_xticks(range(top_n))
    ax2.set_xticklabels([f'F{i}' for i in top_indices], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative weight importance
    ax3 = axes[1, 0]
    cumulative_importance = np.cumsum(abs_weights[sorted_indices]) / np.sum(abs_weights)
    ax3.plot(range(len(cumulative_importance)), cumulative_importance, 'b-', linewidth=2)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% importance')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% importance')
    ax3.set_title('Cumulative Weight Importance')
    ax3.set_xlabel('Number of Features (sorted by importance)')
    ax3.set_ylabel('Cumulative Importance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Weight magnitude vs. index
    ax4 = axes[1, 1]
    ax4.scatter(range(len(abs_weights)), abs_weights, alpha=0.6, s=20)
    ax4.set_title('Weight Magnitude by Feature Index')
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Absolute Weight Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved weight analysis to {save_path}")
    plt.show()

def analyze_feature_correlations(features, labels, weight_analysis, save_path=None):
    """Analyze correlations between features and labels"""
    print("\n=== Feature-Label Correlation Analysis ===")
    
    # Calculate correlations
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], labels)[0, 1]
        correlations.append(corr)
    
    correlations = np.array(correlations)
    abs_correlations = np.abs(correlations)
    
    # Compare with probe weights
    coef = weight_analysis['coef']
    weight_corr = np.corrcoef(coef, correlations)[0, 1]
    print(f"Correlation between probe weights and feature-label correlations: {weight_corr:.4f}")
    
    # Top correlated features
    sorted_corr_indices = np.argsort(abs_correlations)[::-1]
    print(f"\nTop 10 most correlated features:")
    for i, idx in enumerate(sorted_corr_indices[:10]):
        corr = correlations[idx]
        abs_corr = abs_correlations[idx]
        print(f"  {i+1:2d}. Feature_{idx:<3} | corr={corr:8.4f} | abs={abs_corr:8.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation distribution
    ax1 = axes[0, 0]
    ax1.hist(correlations, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title('Distribution of Feature-Label Correlations')
    ax1.set_xlabel('Correlation Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # 2. Weights vs. correlations
    ax2 = axes[0, 1]
    ax2.scatter(correlations, coef, alpha=0.6, s=20)
    ax2.set_title('Probe Weights vs. Feature-Label Correlations')
    ax2.set_xlabel('Feature-Label Correlation')
    ax2.set_ylabel('Probe Weight')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top correlations bar plot
    ax3 = axes[1, 0]
    top_n = 20
    top_corrs = correlations[sorted_corr_indices[:top_n]]
    top_corr_indices = sorted_corr_indices[:top_n]
    
    colors = ['red' if c < 0 else 'blue' for c in top_corrs]
    bars = ax3.bar(range(top_n), top_corrs, color=colors, alpha=0.7)
    ax3.set_title(f'Top {top_n} Most Correlated Features')
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('Correlation Value')
    ax3.set_xticks(range(top_n))
    ax3.set_xticklabels([f'F{i}' for i in top_corr_indices], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature importance comparison
    ax4 = axes[1, 1]
    top_weight_indices = weight_analysis['sorted_indices'][:20]
    top_corr_indices_20 = sorted_corr_indices[:20]
    
    # Find overlap
    overlap = set(top_weight_indices) & set(top_corr_indices_20)
    print(f"\nOverlap between top 20 weight features and top 20 correlation features: {len(overlap)}")
    
    ax4.scatter(range(20), [i in top_corr_indices_20 for i in top_weight_indices], 
               c=['red' if i in overlap else 'blue' for i in top_weight_indices], 
               s=100, alpha=0.7)
    ax4.set_title('Top 20 Weight Features: Present in Top 20 Correlations?')
    ax4.set_xlabel('Rank in Weight Importance')
    ax4.set_ylabel('Present in Top 20 Correlations')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved correlation analysis to {save_path}")
    plt.show()
    
    return {
        'correlations': correlations,
        'abs_correlations': abs_correlations,
        'weight_correlation': weight_corr,
        'overlap_count': len(overlap)
    }

def analyze_decision_boundary(features, labels, clf, save_path=None):
    """Analyze the decision boundary of the probe"""
    print("\n=== Decision Boundary Analysis ===")
    
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Get predictions and probabilities
    predictions = clf.predict(features)
    probabilities = clf.predict_proba(features)[:, 1]  # Probability of positive class
    
    # Create mesh for decision boundary
    x_min, x_max = features_2d[:, 0].min() - 0.5, features_2d[:, 0].max() + 0.5
    y_min, y_max = features_2d[:, 1].min() - 0.5, features_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Transform mesh back to original space and predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_features = pca.inverse_transform(mesh_points)
    mesh_probs = clf.predict_proba(mesh_features)[:, 1].reshape(xx.shape)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Decision boundary
    ax1 = axes[0]
    contour = ax1.contourf(xx, yy, mesh_probs, levels=20, cmap='RdYlBu', alpha=0.8)
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='viridis', alpha=0.7, s=30)
    ax1.set_title('Decision Boundary (PCA projection)')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    plt.colorbar(contour, ax=ax1)
    
    # 2. Probability distribution
    ax2 = axes[1]
    ax2.hist(probabilities[labels == 0], bins=30, alpha=0.7, label='Class 0', color='blue')
    ax2.hist(probabilities[labels == 1], bins=30, alpha=0.7, label='Class 1', color='red')
    ax2.set_title('Prediction Probability Distribution')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved decision boundary analysis to {save_path}")
    plt.show()
    
    # Analyze separability
    class_0_probs = probabilities[labels == 0]
    class_1_probs = probabilities[labels == 1]
    
    print(f"Class 0 mean probability: {np.mean(class_0_probs):.4f}")
    print(f"Class 1 mean probability: {np.mean(class_1_probs):.4f}")
    print(f"Probability separation: {np.mean(class_1_probs) - np.mean(class_0_probs):.4f}")
    
    return {
        'pca': pca,
        'probabilities': probabilities,
        'class_separation': np.mean(class_1_probs) - np.mean(class_0_probs)
    }

def main():
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    print("=== Probe Interpretability Analysis ===")
    print(f"Environment: {ENV}")
    print(f"Experiment: {EXP}")
    print(f"Formula: {FORMULA}")
    print()
    
    # Load model
    print("Loading model...")
    sampler_fn = FixedSampler.partial(FORMULA)
    build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location="cpu")
    cfg = model_configs[ENV]
    model = build_model(build_env, status, cfg).eval()
    build_env.close()
    
    # Create environment
    env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
    
    # Collect data for probing
    print("Collecting data for probe analysis...")
    
    # Hook into LTL network
    feats = []
    def hook_fn(mod, inp, out):
        h_n = out[1]
        arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
        feats.append(arr)
    
    handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
    
    # Create agent and collect data
    props = set(env.get_propositions())
    search = ExhaustiveSearch(model, props, num_loops=2)
    agent = Agent(model, search=search, propositions=props, verbose=False)
    
    ret = env.reset(seed=SEED)
    obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
    agent.reset()
    
    labels = []
    for step in trange(MAX_STEPS, desc="Collecting data"):
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
        
        ret = env.step(action)
        if len(ret) == 5:
            obs, rew, term, trunc, info = ret
            done = term or trunc
        else:
            obs, rew, done, info = ret
        if done:
            break
    
    handle.remove()
    env.close()
    
    # Process data
    X = np.array(feats)
    y = np.array(labels)
    valid_idxs = (y != -1)
    if len(X) > len(y):
        X = X[:len(y)]
    X, y = X[valid_idxs], y[valid_idxs]
    
    print(f"Collected {len(y)} valid samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train probe
    print("Training probe...")
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"Probe accuracy: {acc:.3f}")
    
    # Analyze probe weights
    weight_analysis = analyze_probe_weights(clf)
    visualize_probe_weights(weight_analysis, 'probe_weight_analysis.png')
    
    # Analyze feature correlations
    corr_analysis = analyze_feature_correlations(X, y, weight_analysis, 'probe_correlation_analysis.png')
    
    # Analyze decision boundary
    boundary_analysis = analyze_decision_boundary(X, y, clf, 'probe_decision_boundary.png')
    
    # Summary
    print("\n=== Summary ===")
    print(f"Probe accuracy: {acc:.3f}")
    print(f"Weight norm: {weight_analysis['weight_norm']:.4f}")
    print(f"Weight-correlation correlation: {corr_analysis['weight_correlation']:.4f}")
    print(f"Feature overlap: {corr_analysis['overlap_count']}/20")
    print(f"Class separation: {boundary_analysis['class_separation']:.4f}")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- probe_weight_analysis.png")
    print("- probe_correlation_analysis.png")
    print("- probe_decision_boundary.png")

if __name__ == '__main__':
    main() 