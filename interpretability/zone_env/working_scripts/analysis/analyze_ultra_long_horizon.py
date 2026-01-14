#!/usr/bin/env python3
"""
Analyze velocity prediction performance across ultra-long horizons (1-100 steps).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def create_velocity_targets(npz_path: Path, output_path: Path):
    """Create velocity, direction, and speed targets from position data."""
    print(f"Loading dataset from {npz_path}...")
    npz = np.load(npz_path, allow_pickle=True)
    data = {k: npz[k] for k in npz.keys()}
    
    # Find all position targets
    position_keys = [k for k in npz.keys() if k.startswith('next') and '_agent_pos' in k]
    horizons = []
    
    for key in position_keys:
        horizon_str = key.replace('next', '').replace('_agent_pos', '')
        try:
            horizons.append(int(horizon_str))
        except ValueError:
            continue
    
    horizons = sorted(set(horizons))
    print(f"Found horizons: {horizons}")
    
    current_pos = npz['obs'][:, 0:2]  # Current agent position
    
    print("Creating velocity targets...")
    for horizon in horizons:
        pos_key = f'next{horizon}_agent_pos'
        if pos_key in npz:
            future_pos = npz[pos_key]
            
            # Calculate velocity (position change)
            velocity = future_pos - current_pos
            data[f'next{horizon}_velocity'] = velocity
            
            # Calculate direction (normalized velocity)
            speed = np.linalg.norm(velocity, axis=1, keepdims=True)
            direction = np.divide(velocity, speed, out=np.zeros_like(velocity), where=speed!=0)
            data[f'next{horizon}_direction'] = direction
            
            # Calculate speed (magnitude)
            data[f'next{horizon}_speed'] = speed.flatten()
    
    # Save enhanced dataset
    print(f"Saving enhanced dataset to {output_path}...")
    np.savez_compressed(output_path, **data)
    print(f"Created {len(horizons)} velocity/direction/speed target sets")
    
    return horizons

def analyze_horizon_decay(results_path: Path, output_dir: Path):
    """Analyze and visualize horizon decay patterns."""
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract performance data
    hook = 'hook_env_mlp1'  # Use environment encoder
    hook_results = results['per_hook'][hook]
    
    horizon_data = []
    for target_name, metrics in hook_results.items():
        if 'velocity' in target_name and target_name.startswith('next'):
            horizon_str = target_name.replace('next', '').replace('_velocity', '')
            try:
                horizon = int(horizon_str)
                sensors_action = metrics.get('sensors_plus_action', {})
                velocity_persistence = metrics.get('velocity_persistence', {})
                
                if sensors_action and 'r2' in sensors_action:
                    horizon_data.append({
                        'horizon': horizon,
                        'r2_network': sensors_action['r2'],
                        'mse_network': sensors_action['mse'],
                        'r2_persistence': velocity_persistence.get('r2', np.nan),
                        'mse_persistence': velocity_persistence.get('mse', np.nan)
                    })
            except ValueError:
                continue
    
    # Sort by horizon
    horizon_data.sort(key=lambda x: x['horizon'])
    
    if not horizon_data:
        print("No horizon data found!")
        return
    
    # Extract arrays for analysis
    horizons = [d['horizon'] for d in horizon_data]
    r2_network = [d['r2_network'] for d in horizon_data]
    mse_network = [d['mse_network'] for d in horizon_data]
    r2_persistence = [d['r2_persistence'] for d in horizon_data]
    
    # Print analysis
    print("\n=== ULTRA-LONG HORIZON VELOCITY PREDICTION ===")
    print()
    print("Horizon | Network R² | Network MSE | Persistence R²")
    print("--------|------------|-------------|---------------")
    for i, h in enumerate(horizons):
        pers_r2 = r2_persistence[i] if not np.isnan(r2_persistence[i]) else "N/A"
        print(f"{h:>7} | {r2_network[i]:>10.3f} | {mse_network[i]:>11.1f} | {pers_r2}")
    
    # Decay analysis
    print("\n=== DECAY ANALYSIS ===")
    initial_r2 = r2_network[0]
    final_r2 = r2_network[-1]
    peak_r2 = max(r2_network)
    peak_horizon = horizons[r2_network.index(peak_r2)]
    
    print(f"Initial R² (horizon {horizons[0]}): {initial_r2:.3f}")
    print(f"Peak R² (horizon {peak_horizon}): {peak_r2:.3f}")
    print(f"Final R² (horizon {horizons[-1]}): {final_r2:.3f}")
    print(f"Total decay from peak: {peak_r2 - final_r2:.3f} ({(peak_r2 - final_r2)/peak_r2:.1%})")
    
    # Find meaningful prediction threshold
    meaningful_threshold = 0.1  # R² > 0.1 considered meaningful
    meaningful_horizons = [h for h, r2 in zip(horizons, r2_network) if r2 > meaningful_threshold]
    
    if meaningful_horizons:
        max_meaningful = max(meaningful_horizons)
        print(f"Meaningful prediction maintained up to: {max_meaningful} steps")
    else:
        print("No meaningful long-term prediction found")
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: R² decay
    ax1.plot(horizons, r2_network, 'b-o', markersize=4, linewidth=2, label='Network')
    valid_persistence = [(h, r2) for h, r2 in zip(horizons, r2_persistence) if not np.isnan(r2)]
    if valid_persistence:
        pers_h, pers_r2 = zip(*valid_persistence)
        ax1.plot(pers_h, pers_r2, 'r--s', markersize=3, alpha=0.7, label='Persistence Baseline')
    
    ax1.axhline(y=meaningful_threshold, color='gray', linestyle=':', alpha=0.7, label='Meaningful Threshold')
    ax1.set_xlabel('Prediction Horizon (steps)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Velocity Prediction Performance vs Horizon')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(bottom=-0.1)
    
    # Plot 2: MSE growth
    ax2.plot(horizons, mse_network, 'g-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Prediction Horizon (steps)')
    ax2.set_ylabel('MSE')
    ax2.set_title('Prediction Error vs Horizon')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Log scale for long horizons
    ax3.semilogx(horizons, r2_network, 'b-o', markersize=4, linewidth=2)
    ax3.axhline(y=meaningful_threshold, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Prediction Horizon (steps, log scale)')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Long-term Prediction (Log Scale)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative performance decay
    relative_performance = [r2 / peak_r2 for r2 in r2_network]
    ax4.plot(horizons, relative_performance, 'purple', marker='o', markersize=4, linewidth=2)
    ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='50% of Peak')
    ax4.set_xlabel('Prediction Horizon (steps)')
    ax4.set_ylabel('Relative Performance (fraction of peak)')
    ax4.set_title('Relative Performance Decay')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plot_path = output_dir / "ultra_long_horizon_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comprehensive analysis plot to {plot_path}")
    
    # Save analysis data
    analysis_data = {
        'horizons': horizons,
        'r2_network': r2_network,
        'mse_network': mse_network,
        'r2_persistence': r2_persistence,
        'summary': {
            'initial_r2': initial_r2,
            'peak_r2': peak_r2,
            'peak_horizon': peak_horizon,
            'final_r2': final_r2,
            'max_meaningful_horizon': max(meaningful_horizons) if meaningful_horizons else 0
        }
    }
    
    analysis_path = output_dir / "ultra_long_horizon_data.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Saved analysis data to {analysis_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True, help="Input NPZ with position targets")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--create_targets", action="store_true", help="Create velocity targets from positions")
    ap.add_argument("--run_probe", action="store_true", help="Run probe analysis")
    ap.add_argument("--analyze", action="store_true", help="Analyze results")
    args = ap.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_npz = args.output_dir / "ultra_long_enhanced.npz"
    
    if args.create_targets:
        horizons = create_velocity_targets(args.npz, enhanced_npz)
        print(f"Created velocity targets for horizons: {horizons}")
    
    if args.run_probe:
        print("Running probe analysis...")
        # Import and run probe analysis
        import sys
        sys.path.append(str(Path(__file__).parent))
        from probe_forward_look import main as probe_main
        
        # Create velocity target list
        velocity_targets = []
        if enhanced_npz.exists():
            npz = np.load(enhanced_npz, allow_pickle=True)
            velocity_targets = [k for k in npz.keys() if k.endswith('_velocity') and k.startswith('next')]
            velocity_targets.sort(key=lambda x: int(x.replace('next', '').replace('_velocity', '')))
        
        if velocity_targets:
            old_argv = sys.argv
            sys.argv = [
                "probe_forward_look.py",
                "--npz", str(enhanced_npz),
                "--hooks", "hook_env_mlp1",
                "--targets"] + velocity_targets + [
                "--include_action",
                "--world_level_split",
                "--out_dir", str(args.output_dir / "probe_results")
            ]
            
            try:
                probe_main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
    
    if args.analyze:
        results_path = args.output_dir / "probe_results" / "results.json"
        if results_path.exists():
            analyze_horizon_decay(results_path, args.output_dir)
        else:
            print(f"Results file not found: {results_path}")

if __name__ == "__main__":
    main()
