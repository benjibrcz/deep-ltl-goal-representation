# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_new_fields_fixed.npz`  
Hooks: hook_env_mlp1  
Targets: next_obs  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.197, mse=0.860 | r2=-2060169289.196, mse=0.857 |