# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_multi_horizon_semantic.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_agent_pos_semantic, next1_agent_pos_semantic, next3_agent_pos_semantic, next5_agent_pos_semantic, next10_agent_pos_semantic  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=28.338 | r2=-0.001, mse=28.341 | r2=-0.004, mse=28.378 | r2=-0.002, mse=28.379 |
| next1_agent_pos_semantic | r2=-0.004, mse=28.472 | r2=-0.005, mse=28.482 | r2=-0.011, mse=28.802 | r2=-0.012, mse=28.800 |
| next3_agent_pos_semantic | r2=-0.014, mse=28.562 | r2=-0.014, mse=28.564 | r2=-0.022, mse=28.908 | r2=-0.024, mse=28.954 |
| next5_agent_pos_semantic | r2=-0.016, mse=28.582 | r2=-0.016, mse=28.572 | r2=-0.020, mse=28.749 | r2=-0.019, mse=28.733 |
| next10_agent_pos_semantic | r2=-0.045, mse=29.458 | r2=-0.046, mse=29.472 | r2=-0.041, mse=29.230 | r2=-0.042, mse=29.252 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=28.338 | r2=-0.001, mse=28.341 | r2=0.000, mse=28.096 | r2=0.003, mse=28.057 |
| next1_agent_pos_semantic | r2=-0.004, mse=28.472 | r2=-0.005, mse=28.482 | r2=0.000, mse=28.164 | r2=0.001, mse=28.152 |
| next3_agent_pos_semantic | r2=-0.014, mse=28.562 | r2=-0.014, mse=28.564 | r2=-0.000, mse=28.096 | r2=0.001, mse=28.084 |
| next5_agent_pos_semantic | r2=-0.016, mse=28.582 | r2=-0.016, mse=28.572 | r2=-0.001, mse=28.064 | r2=-0.001, mse=28.060 |
| next10_agent_pos_semantic | r2=-0.045, mse=29.458 | r2=-0.046, mse=29.472 | r2=-0.001, mse=28.013 | r2=-0.002, mse=28.015 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=28.338 | r2=-0.001, mse=28.341 | r2=-0.004, mse=28.316 | r2=-0.004, mse=28.338 |
| next1_agent_pos_semantic | r2=-0.004, mse=28.472 | r2=-0.005, mse=28.482 | r2=-0.008, mse=28.456 | r2=-0.008, mse=28.448 |
| next3_agent_pos_semantic | r2=-0.014, mse=28.562 | r2=-0.014, mse=28.564 | r2=-0.010, mse=28.360 | r2=-0.009, mse=28.332 |
| next5_agent_pos_semantic | r2=-0.016, mse=28.582 | r2=-0.016, mse=28.572 | r2=-0.008, mse=28.305 | r2=-0.007, mse=28.296 |
| next10_agent_pos_semantic | r2=-0.045, mse=29.458 | r2=-0.046, mse=29.472 | r2=-0.015, mse=28.480 | r2=-0.015, mse=28.508 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=28.338 | r2=-0.001, mse=28.341 | r2=0.004, mse=28.253 | r2=0.001, mse=28.280 |
| next1_agent_pos_semantic | r2=-0.004, mse=28.472 | r2=-0.005, mse=28.482 | r2=-0.001, mse=28.332 | r2=-0.002, mse=28.336 |
| next3_agent_pos_semantic | r2=-0.014, mse=28.562 | r2=-0.014, mse=28.564 | r2=-0.010, mse=28.531 | r2=-0.011, mse=28.575 |
| next5_agent_pos_semantic | r2=-0.016, mse=28.582 | r2=-0.016, mse=28.572 | r2=-0.008, mse=28.267 | r2=-0.008, mse=28.271 |
| next10_agent_pos_semantic | r2=-0.045, mse=29.458 | r2=-0.046, mse=29.472 | r2=-0.017, mse=28.279 | r2=-0.019, mse=28.332 |