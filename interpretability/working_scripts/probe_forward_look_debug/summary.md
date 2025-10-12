# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_small_enhanced.npz`  
Hooks: hook_env_mlp1  
Targets: next_obs  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=-218657.611, mse=234.740 | r2=-218724.813, mse=264.131 | r2=-19510673143404.531, mse=2.549 | r2=-19510673143400.246, mse=2.734 |