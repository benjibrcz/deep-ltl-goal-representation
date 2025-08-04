# Comprehensive Generalization Results

This folder contains the results from comprehensive generalization analysis using `probe_comprehensive_generalization.py`.

## File Naming Convention

All files follow the pattern: `{test_type}_{target_feature}_{layer_name}_{timestamp}.{extension}`

### Components:
- `test_type`: Type of analysis (e.g., "comprehensive_generalization", "summary")
- `target_feature`: Which feature was probed (e.g., "agent_pos", "wall_sensor", "zone_lidar")  
- `layer_name`: Neural network layer name with dots replaced by underscores (e.g., "env_net_mlp_0")
- `timestamp`: When the analysis was run (format: YYYYMMDD_HHMMSS)
- `extension`: File type (.png, .csv, .txt)

### Example Files:
```
comprehensive_generalization_agent_pos_env_net_mlp_0_20241218_143022.png
comprehensive_generalization_agent_pos_env_net_mlp_0_20241218_143022.csv  
summary_agent_pos_env_net_mlp_0_20241218_143022.txt
```

## File Types

### 1. PNG Files (Visualization)
4-panel plots comparing the three generalization types:
- **Panel 1**: R² scores by split type (temporal, spatial, environmental)
- **Panel 2**: MSE scores by split type  
- **Panel 3**: Sample sizes (train vs test) for each split
- **Panel 4**: R² vs MSE scatter plot

### 2. CSV Files (Raw Data)
Structured data with columns:
- `timestamp`: When analysis was run
- `split_type`: temporal/spatial/environmental
- `target_feature`: What was being predicted
- `layer`: Neural network layer analyzed
- `r2_train`, `r2_test`: R² scores for train/test sets
- `mse_train`, `mse_test`: MSE scores for train/test sets  
- `n_train_samples`, `n_test_samples`: Sample sizes
- `feature_dim`, `target_dim`: Dimensionality information

### 3. TXT Files (Summary Reports)
Human-readable summaries containing:
- Experiment metadata (layer, target, parameters)
- Split statistics (sample sizes per split type)
- Performance results (R², MSE for each generalization type)
- Best/worst performing splits
- Generalization gaps (performance differences between split types)

## Generalization Types Explained

### Temporal Split
- **Train**: First half of time steps from each rollout
- **Test**: Second half of time steps from same rollouts
- **Tests**: Consistency of representations over time

### Spatial Split  
- **Train**: Some rollouts from each world
- **Test**: Different rollouts from same worlds
- **Tests**: Generalization to new starting positions in familiar environments

### Environmental Split
- **Train**: First half of worlds (different zone maps)
- **Test**: Second half of worlds (completely new zone maps)
- **Tests**: Generalization to entirely unseen environments

## Interpreting Results

### Expected Performance Hierarchy
Typically: **Temporal > Spatial > Environmental**

### Good Generalization Indicators
- High R² scores (> 0.7) across all split types
- Small gaps between temporal, spatial, and environmental performance
- Consistent performance across different layers/features

### Poor Generalization Indicators  
- Large drops from temporal to spatial to environmental
- Negative R² scores (worse than random baseline)
- High variance in results across different runs 