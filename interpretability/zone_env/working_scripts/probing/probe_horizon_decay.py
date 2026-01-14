#!/usr/bin/env python3
"""
Probe velocity prediction performance across many time horizons to analyze decay.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

def extract_horizon_results(results_dir: Path) -> Dict[int, Dict[str, float]]:
    """Extract r2 and mse for velocity targets across horizons."""
    results_file = results_dir / "results.json"
    if not results_file.exists():
        return {}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    horizon_data = {}
    
    # Extract from first hook (they should be similar for baseline metrics)
    hook_name = list(results["per_hook"].keys())[0]
    hook_results = results["per_hook"][hook_name]
    
    for target_name, metrics in hook_results.items():
        if "velocity" in target_name and target_name.startswith("next"):
            # Extract horizon number from target name (e.g., "next5_velocity" -> 5)
            try:
                horizon_str = target_name.replace("next", "").replace("_velocity", "")
                horizon = int(horizon_str)
                
                # Get sensors+action performance (best baseline)
                sensors_action = metrics.get("sensors_plus_action", {})
                if sensors_action and "r2" in sensors_action:
                    horizon_data[horizon] = {
                        "r2": sensors_action["r2"],
                        "mse": sensors_action["mse"]
                    }
            except ValueError:
                continue
    
    return horizon_data

def plot_horizon_decay(horizon_data: Dict[int, Dict[str, float]], output_path: Path):
    """Plot velocity prediction performance vs horizon."""
    if not horizon_data:
        print("No horizon data to plot")
        return
    
    horizons = sorted(horizon_data.keys())
    r2_values = [horizon_data[h]["r2"] for h in horizons]
    mse_values = [horizon_data[h]["mse"] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² decay
    ax1.plot(horizons, r2_values, 'b-o', markersize=4, linewidth=2)
    ax1.set_xlabel('Prediction Horizon (steps)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Velocity Prediction R² vs Horizon')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # MSE growth
    ax2.plot(horizons, mse_values, 'r-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Prediction Horizon (steps)')
    ax2.set_ylabel('MSE')
    ax2.set_title('Velocity Prediction MSE vs Horizon')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved horizon decay plot to {output_path}")
    
    # Print summary statistics
    print(f"\nHorizon Decay Analysis:")
    print(f"Horizons tested: {min(horizons)} to {max(horizons)} steps")
    print(f"R² at horizon 1: {horizon_data[min(horizons)]['r2']:.3f}")
    print(f"R² at horizon {max(horizons)}: {horizon_data[max(horizons)]['r2']:.3f}")
    print(f"R² decay: {horizon_data[min(horizons)]['r2'] - horizon_data[max(horizons)]['r2']:.3f}")
    
    # Find half-life (horizon where r2 drops to half of initial value)
    initial_r2 = horizon_data[min(horizons)]['r2']
    half_r2 = initial_r2 / 2
    
    for h in horizons:
        if horizon_data[h]['r2'] <= half_r2:
            print(f"R² half-life: ~{h} steps (r2={horizon_data[h]['r2']:.3f})")
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--hooks", nargs="+", default=["hook_env_mlp1"])
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NPZ and find all velocity targets
    data = np.load(args.npz, allow_pickle=True)
    velocity_targets = [k for k in data.keys() if "velocity" in k and k.startswith("next")]
    
    if not velocity_targets:
        print("No velocity targets found in NPZ file")
        return
    
    print(f"Found {len(velocity_targets)} velocity targets: {velocity_targets}")
    
    # Run probe analysis
    from probe_forward_look import main as probe_main
    import sys
    
    # Temporarily modify sys.argv for probe_forward_look
    old_argv = sys.argv
    sys.argv = [
        "probe_forward_look.py",
        "--npz", str(args.npz),
        "--hooks"] + args.hooks + [
        "--targets"] + velocity_targets + [
        "--include_action",
        "--out_dir", str(args.output_dir / "probe_results")
    ]
    
    try:
        probe_main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv
    
    # Extract and plot results
    horizon_data = extract_horizon_results(args.output_dir / "probe_results")
    if horizon_data:
        plot_horizon_decay(horizon_data, args.output_dir / "velocity_horizon_decay.png")
        
        # Save raw data
        with open(args.output_dir / "horizon_data.json", 'w') as f:
            json.dump(horizon_data, f, indent=2)
    else:
        print("No horizon data extracted")

if __name__ == "__main__":
    main()
