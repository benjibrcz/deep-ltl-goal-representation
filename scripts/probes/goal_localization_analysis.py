#!/usr/bin/env python3
"""
Comprehensive Goal Representation Analysis - Memory Efficient

This script analyzes goal representations layer by layer to avoid memory issues:
1. Different network layers (LTL RNN, Policy Encoder, Environment Network)
2. Different dimensionality reduction techniques (PCA, t-SNE)
3. Clustering analysis
4. Goal-specific probes
5. Layer-wise comparison
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tqdm import trange, tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from collections import defaultdict
import pandas as pd
import gc

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.model_store import ModelStore
from model.model import build_model
from config import model_configs
from ltl import FixedSampler
from envs import make_env
from envs.flatworld import FlatWorld
from sequence.search import ExhaustiveSearch
from model.agent import Agent

# Configuration
ENV = "PointLtl2-v0"
EXP = "big_test"
SEED = 1
NUM_ROLLOUTS = 50  # More data points for better analysis

def get_formula_color(formula):
    """Get the actual color that matches the formula"""
    color_map = {
        "FG blue": "blue",
        "FG magenta": "magenta", 
        "FG green": "green",
        "FG yellow": "yellow"
    }
    return color_map.get(formula, "gray")

def collect_activations_for_layer(layer_name, formulas):
    """Collect activations for a specific layer across all formulas"""
    all_activations = []
    all_labels = []
    
    def hook_fn(mod, inp, out):
        if layer_name == "ltl_rnn":
            if isinstance(out, tuple):
                h_n = out[1]  # Final hidden state
                arr = h_n.detach().squeeze(0).squeeze(0).cpu().numpy()
            else:
                arr = out.detach().squeeze().cpu().numpy()
        else:
            if hasattr(out, 'detach'):
                arr = out.detach().squeeze().cpu().numpy()
            else:
                arr = out.squeeze().cpu().numpy()
        all_activations.append(arr)

    for formula in formulas:
        print(f"  Processing {formula}...")
        
        sampler_fn = FixedSampler.partial(formula)
        build_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
        store = ModelStore(ENV, EXP, 0)
        store.load_vocab()
        status = store.load_training_status(map_location="cpu")
        cfg = model_configs[ENV]
        model = build_model(build_env, status, cfg).eval()
        props = sorted(list(set(c.color for c in FlatWorld.CIRCLES)))
        search = ExhaustiveSearch(model, set(props), num_loops=2)
        agent = Agent(model, search=search, propositions=set(props), verbose=False)
        build_env.close()

        # Hook into the layer
        handle = None
        if layer_name == "ltl_rnn":
            if hasattr(model.ltl_net, 'rnn') and model.ltl_net.rnn is not None:
                handle = model.ltl_net.rnn.register_forward_hook(hook_fn)
        elif layer_name == "policy_encoder":
            if hasattr(model, 'actor') and hasattr(model.actor, 'enc'):
                handle = model.actor.enc.register_forward_hook(hook_fn)
        elif layer_name == "env_net":
            if hasattr(model, 'env_net'):
                handle = model.env_net.register_forward_hook(hook_fn)
        elif layer_name == "set_network":
            if hasattr(model.ltl_net, 'set_network'):
                handle = model.ltl_net.set_network.register_forward_hook(hook_fn)
        
        if handle is None:
            print(f"    Could not hook into {layer_name}")
            continue
            
        # Collect activations for this formula
        start_idx = len(all_activations)
        for rollout_idx in trange(NUM_ROLLOUTS, desc=f"    Rollouts", leave=False):
            rollout_env = make_env(ENV, sampler_fn, sequence=False, render_mode=None)
            ret = rollout_env.reset(seed=rollout_idx + 1)
            obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})
            agent.reset()
            
            # Take one step and record activation
            action = agent.get_action(obs, info, deterministic=True).flatten()
            
            # Get the FIRST activation from this rollout only
            if len(all_activations) > start_idx:
                # Only take the first activation for this rollout
                first_activation_idx = start_idx
                if first_activation_idx < len(all_activations):
                    all_labels.append(formula)
            
            rollout_env.close()
        
        handle.remove()
        
        # Clean up memory
        del model, agent, search
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Only keep the first activation per rollout
    final_activations = []
    final_labels = []
    for i in range(0, len(all_activations), len(all_activations)//(len(formulas) * NUM_ROLLOUTS)):
        if i < len(all_activations):
            final_activations.append(all_activations[i])
            if i//(len(all_activations)//(len(formulas) * NUM_ROLLOUTS)) < len(all_labels):
                final_labels.append(all_labels[i//(len(all_activations)//(len(formulas) * NUM_ROLLOUTS))])
    
    print(f"    Collected {len(final_activations)} activations, {len(final_labels)} labels")
    return np.array(final_activations), final_labels

def analyze_layer_separation(X, labels, layer_name):
    """Analyze separation for a specific layer"""
    print(f"\n=== Analyzing {layer_name} ===")
    print(f"Data shape: {X.shape}")
    print(f"Unique labels: {len(set(labels))}")
    
    # 1. PCA Analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 2. t-SNE Analysis
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
    X_tsne = tsne.fit_transform(X)
    
    # 3. Clustering Analysis
    print("Running clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # 4. Silhouette Score
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    
    # 5. Goal Classification
    unique_labels = list(set(labels))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_idx[label] for label in labels])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"Goal Classification Accuracy: {acc:.3f}")
    
    # 6. Visualizations
    colors = [get_formula_color(label) for label in labels]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA Plot
    scatter1 = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=50)
    axes[0,0].set_xlabel('PC1')
    axes[0,0].set_ylabel('PC2')
    axes[0,0].set_title(f'{layer_name} - PCA\nAccuracy: {acc:.3f}, Silhouette: {silhouette_avg:.3f}')
    axes[0,0].grid(True, alpha=0.3)
    
    # t-SNE Plot
    scatter2 = axes[0,1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.7, s=50)
    axes[0,1].set_xlabel('t-SNE 1')
    axes[0,1].set_ylabel('t-SNE 2')
    axes[0,1].set_title(f'{layer_name} - t-SNE')
    axes[0,1].grid(True, alpha=0.3)
    
    # Clustering Plot (PCA space)
    scatter3 = axes[1,0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, alpha=0.7, s=50, cmap='tab10')
    axes[1,0].set_xlabel('PC1')
    axes[1,0].set_ylabel('PC2')
    axes[1,0].set_title(f'{layer_name} - K-means Clustering')
    axes[1,0].grid(True, alpha=0.3)
    
    # Feature Importance
    feature_importance = np.abs(clf.coef_).mean(axis=0)
    axes[1,1].bar(range(len(feature_importance)), feature_importance)
    axes[1,1].set_xlabel('Feature Index')
    axes[1,1].set_ylabel('Average |Weight|')
    axes[1,1].set_title(f'{layer_name} - Feature Importance')
    
    # Add legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_formula_color(label), 
                     markersize=10, label=label)
        for label in unique_labels
    ]
    axes[0,0].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'goal_analysis_plots/{layer_name}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': acc,
        'silhouette': silhouette_avg,
        'pca_variance': pca.explained_variance_ratio_,
        'feature_importance': feature_importance
    }

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    FORMULAS = ["FG blue", "FG magenta", "FG green", "FG yellow"]
    
    # Layers to analyze
    layers_to_analyze = ["policy_encoder", "env_net"]
    
    # Store results for each layer
    summary_results = {}
    
    print(f"Analyzing {len(layers_to_analyze)} layers with {len(FORMULAS)} formulas each")
    print(f"Total expected data points: {len(layers_to_analyze) * len(FORMULAS) * NUM_ROLLOUTS}")
    
    for layer_name in layers_to_analyze:
        print(f"\n{'='*60}")
        print(f"ANALYZING LAYER: {layer_name}")
        print(f"{'='*60}")
        
        try:
            # Collect activations for this layer
            X, labels = collect_activations_for_layer(layer_name, FORMULAS)
            
            if len(X) > 0:
                # Analyze this layer
                results = analyze_layer_separation(X, labels, layer_name)
                summary_results[layer_name] = results
                
                # Save raw data
                np.save(f'goal_analysis_plots/{layer_name}_activations.npy', X)
                np.save(f'goal_analysis_plots/{layer_name}_labels.npy', labels)
                
                print(f"✓ Completed analysis for {layer_name}")
            else:
                print(f"✗ No activations collected for {layer_name}")
                
        except Exception as e:
            print(f"✗ Error analyzing {layer_name}: {e}")
            continue
        
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL LAYERS")
    print(f"{'='*60}")
    print(f"{'Layer':<15} {'Accuracy':<10} {'Silhouette':<12} {'PCA Var':<15}")
    print("-" * 60)
    
    for layer_name, results in summary_results.items():
        pca_var = results['pca_variance'].sum()
        print(f"{layer_name:<15} {results['accuracy']:<10.3f} {results['silhouette']:<12.3f} {pca_var:<15.3f}")
    
    # Find best layer
    if summary_results:
        best_layer = max(summary_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest layer for goal separation: {best_layer[0]} (accuracy: {best_layer[1]['accuracy']:.3f})")
    
    # Save summary
    summary_df = pd.DataFrame(summary_results).T
    summary_df.to_csv('goal_analysis_plots/layer_analysis_summary.csv')
    print(f"\nSummary saved to: goal_analysis_plots/layer_analysis_summary.csv")

if __name__ == '__main__':
    main() 