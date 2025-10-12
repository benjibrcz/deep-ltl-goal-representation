# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_velocity_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_velocity, next1_velocity, next3_velocity, next5_velocity, next10_velocity  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_velocity | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=0.498, mse=28.367 | r2=0.498, mse=28.369 |
| next1_velocity | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=0.486, mse=28.770 | r2=0.486, mse=28.768 |
| next3_velocity | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=0.461, mse=28.883 | r2=0.460, mse=28.929 |
| next5_velocity | r2=0.476, mse=28.582 | r2=0.476, mse=28.572 | r2=0.474, mse=28.715 | r2=0.475, mse=28.698 |
| next10_velocity | r2=0.477, mse=29.458 | r2=0.477, mse=29.472 | r2=0.480, mse=29.181 | r2=0.479, mse=29.204 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_velocity | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=-0.000, mse=56.163 | r2=0.020, mse=54.473 |
| next1_velocity | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=-0.000, mse=54.409 | r2=0.020, mse=52.918 |
| next3_velocity | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=-0.000, mse=53.825 | r2=0.019, mse=52.373 |
| next5_velocity | r2=0.476, mse=28.582 | r2=0.476, mse=28.572 | r2=-0.000, mse=52.846 | r2=0.020, mse=51.454 |
| next10_velocity | r2=0.477, mse=29.458 | r2=0.477, mse=29.472 | r2=-0.000, mse=54.948 | r2=0.019, mse=53.688 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_velocity | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=0.480, mse=28.960 | r2=0.480, mse=28.979 |
| next1_velocity | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=0.469, mse=29.273 | r2=0.470, mse=29.248 |
| next3_velocity | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=0.451, mse=28.955 | r2=0.451, mse=28.954 |
| next5_velocity | r2=0.476, mse=28.582 | r2=0.476, mse=28.572 | r2=0.462, mse=28.885 | r2=0.463, mse=28.869 |
| next10_velocity | r2=0.477, mse=29.458 | r2=0.477, mse=29.472 | r2=0.475, mse=29.352 | r2=0.475, mse=29.362 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_velocity | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=0.501, mse=28.228 | r2=0.499, mse=28.253 |
| next1_velocity | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=0.491, mse=28.323 | r2=0.490, mse=28.328 |
| next3_velocity | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=0.469, mse=28.460 | r2=0.468, mse=28.514 |
| next5_velocity | r2=0.476, mse=28.582 | r2=0.476, mse=28.572 | r2=0.479, mse=28.261 | r2=0.479, mse=28.266 |
| next10_velocity | r2=0.477, mse=29.458 | r2=0.477, mse=29.472 | r2=0.491, mse=28.288 | r2=0.490, mse=28.349 |