# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_velocity_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5  
Targets: next0_velocity, next1_velocity, next3_velocity  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_velocity | r2=-2.017, mse=171.591 | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=0.498, mse=28.367 | r2=0.498, mse=28.369 |
| next1_velocity | r2=-2.011, mse=165.663 | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=0.486, mse=28.770 | r2=0.486, mse=28.768 |
| next3_velocity | r2=-2.091, mse=166.564 | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=0.461, mse=28.883 | r2=0.460, mse=28.929 |

## Hook: `hook_ltl_rnn_h`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_velocity | r2=-2.017, mse=171.591 | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=-0.000, mse=56.163 | r2=0.020, mse=54.473 |
| next1_velocity | r2=-2.011, mse=165.663 | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=-0.000, mse=54.409 | r2=0.020, mse=52.918 |
| next3_velocity | r2=-2.091, mse=166.564 | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=-0.000, mse=53.825 | r2=0.019, mse=52.373 |

## Hook: `hook_actor_h5`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next0_velocity | r2=-2.017, mse=171.591 | r2=0.498, mse=28.338 | r2=0.499, mse=28.341 | r2=0.480, mse=28.960 | r2=0.480, mse=28.979 |
| next1_velocity | r2=-2.011, mse=165.663 | r2=0.489, mse=28.471 | r2=0.489, mse=28.482 | r2=0.469, mse=29.273 | r2=0.470, mse=29.248 |
| next3_velocity | r2=-2.091, mse=166.564 | r2=0.465, mse=28.562 | r2=0.465, mse=28.564 | r2=0.451, mse=28.955 | r2=0.451, mse=28.954 |