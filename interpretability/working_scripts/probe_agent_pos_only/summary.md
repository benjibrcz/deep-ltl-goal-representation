# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_next0_large.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h  
Targets: next0_agent_pos  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.007, mse=27.373 | r2=-0.006, mse=27.371 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.001, mse=26.931 | r2=0.001, mse=26.912 |