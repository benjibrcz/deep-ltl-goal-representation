# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_velocity_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_direction, next1_direction, next3_direction, next5_direction, next10_direction  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_direction | r2=-1.776, mse=1.360 | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.432, mse=0.298 | r2=0.432, mse=0.298 |
| next1_direction | r2=-1.750, mse=1.345 | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.412, mse=0.310 | r2=0.413, mse=0.310 |
| next3_direction | r2=-1.808, mse=1.377 | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.366, mse=0.325 | r2=0.365, mse=0.325 |
| next5_direction | r2=-1.723, mse=1.354 | r2=0.380, mse=0.317 | r2=0.381, mse=0.317 | r2=0.397, mse=0.310 | r2=0.398, mse=0.310 |
| next10_direction | r2=-1.741, mse=1.363 | r2=0.357, mse=0.330 | r2=0.358, mse=0.329 | r2=0.370, mse=0.325 | r2=0.370, mse=0.325 |

## Hook: `hook_ltl_rnn_h`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_direction | r2=-1.776, mse=1.360 | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=-0.000, mse=0.495 | r2=0.022, mse=0.482 |
| next1_direction | r2=-1.750, mse=1.345 | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=-0.000, mse=0.495 | r2=0.016, mse=0.486 |
| next3_direction | r2=-1.808, mse=1.377 | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=-0.000, mse=0.495 | r2=0.014, mse=0.486 |
| next5_direction | r2=-1.723, mse=1.354 | r2=0.380, mse=0.317 | r2=0.381, mse=0.317 | r2=-0.000, mse=0.495 | r2=0.018, mse=0.485 |
| next10_direction | r2=-1.741, mse=1.363 | r2=0.357, mse=0.330 | r2=0.358, mse=0.329 | r2=0.000, mse=0.495 | r2=0.016, mse=0.487 |

## Hook: `hook_actor_h5`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_direction | r2=-1.776, mse=1.360 | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.417, mse=0.304 | r2=0.416, mse=0.305 |
| next1_direction | r2=-1.750, mse=1.345 | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.400, mse=0.313 | r2=0.400, mse=0.313 |
| next3_direction | r2=-1.808, mse=1.377 | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.361, mse=0.325 | r2=0.361, mse=0.326 |
| next5_direction | r2=-1.723, mse=1.354 | r2=0.380, mse=0.317 | r2=0.381, mse=0.317 | r2=0.388, mse=0.314 | r2=0.388, mse=0.314 |
| next10_direction | r2=-1.741, mse=1.363 | r2=0.357, mse=0.330 | r2=0.358, mse=0.329 | r2=0.371, mse=0.324 | r2=0.371, mse=0.323 |

## Hook: `hook_critic_mlp0`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_direction | r2=-1.776, mse=1.360 | r2=0.412, mse=0.305 | r2=0.415, mse=0.304 | r2=0.435, mse=0.297 | r2=0.434, mse=0.297 |
| next1_direction | r2=-1.750, mse=1.345 | r2=0.391, mse=0.318 | r2=0.392, mse=0.318 | r2=0.416, mse=0.307 | r2=0.416, mse=0.307 |
| next3_direction | r2=-1.808, mse=1.377 | r2=0.354, mse=0.330 | r2=0.356, mse=0.329 | r2=0.374, mse=0.321 | r2=0.374, mse=0.321 |
| next5_direction | r2=-1.723, mse=1.354 | r2=0.380, mse=0.317 | r2=0.381, mse=0.317 | r2=0.401, mse=0.307 | r2=0.401, mse=0.308 |
| next10_direction | r2=-1.741, mse=1.363 | r2=0.357, mse=0.330 | r2=0.358, mse=0.329 | r2=0.385, mse=0.316 | r2=0.384, mse=0.316 |