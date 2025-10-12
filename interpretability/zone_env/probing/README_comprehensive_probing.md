# Comprehensive Environment Network Probing

This directory contains a comprehensive probing system for analyzing the environment network and various environmental features in the deep-ltl project.

## Overview

The comprehensive probing system allows you to probe:

1. **Environment Network Components:**
   - `env_net.mlp.0` - First MLP layer (close to input)
   - `env_net.mlp.1` - Middle MLP layer 
   - `env_net.mlp.2` - Final MLP layer (close to output)

2. **Environmental Features:**
   - `agent_pos` - Agent position coordinates
   - `zone_lidar` - Zone lidar readings
   - `zone_differences` - Differences between zone lidars
   - `wall_sensor` - Wall sensor readings
   - `wall_lidar` - Wall lidar readings  
   - `agent_sensors` - Agent sensors (accelerometer, velocimeter, gyro)
   - `joint_positions` - Joint positions and velocities

## Files

- `comprehensive_env_net_probe.py` - Main probing script
- `run_comprehensive_probe_examples.py` - Helper script to run multiple experiments
- `README_comprehensive_probing.md` - This documentation

## Usage

### Single Probe Experiment

```bash
python interpretability/probing/comprehensive_env_net_probe.py \
    --layer env_net.mlp.0 \
    --target agent_pos \
    --n-worlds 10 \
    --n-rollouts 10 \
    --max-steps 200
```

### Run Multiple Experiments

```bash
python interpretability/probing/run_comprehensive_probe_examples.py
```

This will run a predefined set of experiments covering different layer-target combinations.

## Command Line Arguments

- `--layer` (required): Layer to probe (e.g., `env_net.mlp.0`, `env_net.mlp.2`)
- `--target` (required): Target feature to predict (see list above)
- `--n-worlds`: Number of worlds to use (default: 10)
- `--n-rollouts`: Number of rollouts per world (default: 10)
- `--max-steps`: Maximum steps per rollout (default: 200)  
- `--n-components`: Number of PCA components (default: None for 95% variance)
- `--output-dir`: Output directory for results (default: `interpretability/probing/comprehensive_results`)
- `--seed`: Random seed for reproducibility (default: 0)

## Output Files

Each experiment generates three files:

1. **CSV File** (`comprehensive_probe_{target}_{layer}_{timestamp}.csv`):
   - Structured data with all metrics
   - Columns: timestamp, split_type, target_feature, layer, r2_train, r2_test, mse_train, mse_test, n_train, n_test, n_components

2. **Visualization** (`comprehensive_probe_{target}_{layer}_{timestamp}.png`):
   - 4-panel plot showing:
     - R² scores by split type
     - MSE scores by split type
     - Sample sizes (train vs test)
     - R² vs MSE scatter plot

3. **Summary Report** (`summary_{target}_{layer}_{timestamp}.txt`):
   - Human-readable summary
   - Performance by split type
   - Generalization analysis

## Generalization Analysis

The system evaluates three types of generalization:

### 1. Temporal Generalization
- **Train**: Early time steps in trajectories
- **Test**: Later time steps in same trajectories
- **Tests**: Consistency of representations over time

### 2. Spatial Generalization  
- **Train**: Some rollouts from each world
- **Test**: Different rollouts from same worlds
- **Tests**: Generalization to new starting positions

### 3. Environmental Generalization
- **Train**: First half of worlds  
- **Test**: Second half of worlds
- **Tests**: Generalization to completely unseen environments

## Interpreting Results

### Good Generalization Indicators
- High R² scores (> 0.7) across all split types
- Small performance gaps between temporal, spatial, and environmental splits
- Consistent performance across different layers

### Poor Generalization Indicators
- Large drops from temporal → spatial → environmental performance
- Negative R² scores (worse than random baseline)
- High variance in results

### Expected Performance Hierarchy
Typically: **Temporal > Spatial > Environmental**

## Example Results Analysis

```
TEMPORAL SPLIT:
  R² (test): 0.8234
  MSE (test): 0.0876
  Train samples: 1247
  Test samples: 1089

SPATIAL SPLIT:
  R² (test): 0.7123  
  MSE (test): 0.1234
  Train samples: 1156
  Test samples: 1180

ENVIRONMENTAL SPLIT:
  R² (test): 0.5234
  MSE (test): 0.2234
  Train samples: 1089
  Test samples: 1247

Temporal-Environmental Gap: 0.3000
Best performing split: temporal (R² = 0.8234)
```

This shows good temporal consistency but moderate environmental generalization.

## Common Layer-Target Combinations

**For Agent Position Analysis:**
```bash
python comprehensive_env_net_probe.py --layer env_net.mlp.0 --target agent_pos
python comprehensive_env_net_probe.py --layer env_net.mlp.2 --target agent_pos
```

**For Zone Analysis:**
```bash  
python comprehensive_env_net_probe.py --layer env_net.mlp.0 --target zone_lidar
python comprehensive_env_net_probe.py --layer env_net.mlp.1 --target zone_differences
```

**For Sensor Analysis:**
```bash
python comprehensive_env_net_probe.py --layer env_net.mlp.0 --target wall_sensor
python comprehensive_env_net_probe.py --layer env_net.mlp.1 --target agent_sensors
```

## Tips

1. **Start Small**: Use fewer worlds/rollouts for initial testing
2. **Layer Comparison**: Compare the same target across different layers
3. **Target Comparison**: Compare different targets on the same layer
4. **PCA Components**: Experiment with different numbers of PCA components
5. **Multiple Runs**: Run with different seeds to assess stability

## Troubleshooting

**No data collected**: Check that the environment and model load correctly
**Low performance**: Try different layers or increase sample sizes
**Memory issues**: Reduce n-worlds, n-rollouts, or max-steps 