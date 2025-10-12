# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_velocity_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_direction, next1_direction, next3_direction  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_direction | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.432, mse=0.298 | r2=0.432, mse=0.298 |
| next1_direction | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.412, mse=0.310 | r2=0.413, mse=0.310 |
| next3_direction | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.366, mse=0.325 | r2=0.365, mse=0.325 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_direction | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=-0.000, mse=0.495 | r2=0.022, mse=0.482 |
| next1_direction | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=-0.000, mse=0.495 | r2=0.016, mse=0.486 |
| next3_direction | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=-0.000, mse=0.495 | r2=0.014, mse=0.486 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_direction | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.417, mse=0.304 | r2=0.416, mse=0.305 |
| next1_direction | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.400, mse=0.313 | r2=0.400, mse=0.313 |
| next3_direction | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.361, mse=0.325 | r2=0.361, mse=0.326 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_direction | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.435, mse=0.297 | r2=0.434, mse=0.297 |
| next1_direction | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.416, mse=0.307 | r2=0.416, mse=0.307 |
| next3_direction | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.374, mse=0.321 | r2=0.374, mse=0.321 |