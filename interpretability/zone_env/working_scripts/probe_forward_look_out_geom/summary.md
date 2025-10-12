# Forward-look Probe (v0)

NPZ: `/Users/benji.berczi/Documents/deep-ltl/interpretability/working_scripts/rollouts_stateful.npz`  
Hooks: hook_env_mlp1, hook_env_mlp3, hook_ltl_rnn_h, hook_actor_h5  
Targets: next_ap, next_delta_xy, next_direction_xy, next_dist_to_goal, next_obs, next_positives, next_speed, next_unit_vec_to_goal  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | geom_only | sensors+action | hook | hook+action |
|---|---|---|---|---|---|---|
| next_ap | — | r2=-4.314, mse=0.185 | r2=-0.557, mse=0.077 | r2=-4.831, mse=0.173 | r2=-5.776, mse=0.163 | r2=-5.858, mse=0.158 |
| next_delta_xy | r2=-2.063, mse=110.542 | r2=-2.481, mse=27.715 | r2=0.252, mse=19.518 | r2=0.518, mse=22.577 | r2=-1.206, mse=25.252 | r2=0.521, mse=22.393 |
| next_direction_xy | r2=-0.455, mse=0.943 | r2=0.084, mse=0.476 | r2=0.153, mse=0.404 | r2=-0.036, mse=0.481 | r2=-0.178, mse=0.577 | r2=-0.327, mse=0.587 |
| next_dist_to_goal | — | r2=-0.413, mse=23.557 | r2=-0.008, mse=16.808 | r2=-0.232, mse=20.553 | r2=-0.537, mse=25.625 | r2=-0.488, mse=24.817 |
| next_obs | — | r2=0.526, mse=4.949 | r2=-0.032, mse=3.765 | r2=0.730, mse=4.136 | r2=0.559, mse=4.231 | r2=0.660, mse=3.768 |
| next_positives | — | r2=-0.408, mse=0.484 | r2=-0.234, mse=0.224 | r2=-0.509, mse=0.493 | r2=-0.439, mse=0.421 | r2=-0.528, mse=0.438 |
| next_speed | r2=0.148, mse=36.096 | r2=-1.104, mse=88.492 | r2=-0.053, mse=44.277 | r2=-0.818, mse=76.463 | r2=-0.755, mse=73.823 | r2=-0.626, mse=68.397 |
| next_unit_vec_to_goal | — | r2=-1.371, mse=1.156 | r2=0.089, mse=0.448 | r2=-0.673, mse=0.834 | r2=-0.582, mse=0.775 | r2=-0.445, mse=0.710 |

## Hook: `hook_env_mlp3`

| target | velocity_persistence | baseline | geom_only | sensors+action | hook | hook+action |
|---|---|---|---|---|---|---|
| next_ap | — | r2=-4.314, mse=0.185 | r2=-0.557, mse=0.077 | r2=-4.831, mse=0.173 | r2=-3.170, mse=0.101 | r2=-3.458, mse=0.102 |
| next_delta_xy | r2=-2.063, mse=110.542 | r2=-2.481, mse=27.715 | r2=0.252, mse=19.518 | r2=0.518, mse=22.577 | r2=-0.013, mse=22.183 | r2=0.310, mse=21.425 |
| next_direction_xy | r2=-0.455, mse=0.943 | r2=0.084, mse=0.476 | r2=0.153, mse=0.404 | r2=-0.036, mse=0.481 | r2=0.029, mse=0.463 | r2=-0.074, mse=0.494 |
| next_dist_to_goal | — | r2=-0.413, mse=23.557 | r2=-0.008, mse=16.808 | r2=-0.232, mse=20.553 | r2=-0.391, mse=23.199 | r2=-0.315, mse=21.921 |
| next_obs | — | r2=0.526, mse=4.949 | r2=-0.032, mse=3.765 | r2=0.730, mse=4.136 | r2=0.081, mse=4.021 | r2=0.161, mse=3.717 |
| next_positives | — | r2=-0.408, mse=0.484 | r2=-0.234, mse=0.224 | r2=-0.509, mse=0.493 | r2=-0.385, mse=0.374 | r2=-0.458, mse=0.390 |
| next_speed | r2=0.148, mse=36.096 | r2=-1.104, mse=88.492 | r2=-0.053, mse=44.277 | r2=-0.818, mse=76.463 | r2=-0.750, mse=73.620 | r2=-0.720, mse=72.344 |
| next_unit_vec_to_goal | — | r2=-1.371, mse=1.156 | r2=0.089, mse=0.448 | r2=-0.673, mse=0.834 | r2=-0.233, mse=0.601 | r2=-0.154, mse=0.565 |

## Hook: `hook_ltl_rnn_h`

| target | velocity_persistence | baseline | geom_only | sensors+action | hook | hook+action |
|---|---|---|---|---|---|---|
| next_ap | — | r2=-4.314, mse=0.185 | r2=-0.557, mse=0.077 | r2=-4.831, mse=0.173 | r2=-1.540, mse=0.122 | r2=-1.531, mse=0.122 |
| next_delta_xy | r2=-2.063, mse=110.542 | r2=-2.481, mse=27.715 | r2=0.252, mse=19.518 | r2=0.518, mse=22.577 | r2=-0.021, mse=35.312 | r2=-0.016, mse=35.267 |
| next_direction_xy | r2=-0.455, mse=0.943 | r2=0.084, mse=0.476 | r2=0.153, mse=0.404 | r2=-0.036, mse=0.481 | r2=-0.143, mse=0.525 | r2=-0.142, mse=0.521 |
| next_dist_to_goal | — | r2=-0.413, mse=23.557 | r2=-0.008, mse=16.808 | r2=-0.232, mse=20.553 | r2=-0.055, mse=17.594 | r2=-0.035, mse=17.259 |
| next_obs | — | r2=0.526, mse=4.949 | r2=-0.032, mse=3.765 | r2=0.730, mse=4.136 | r2=-0.146, mse=3.945 | r2=-0.089, mse=3.681 |
| next_positives | — | r2=-0.408, mse=0.484 | r2=-0.234, mse=0.224 | r2=-0.509, mse=0.493 | r2=0.500, mse=0.000 | r2=0.500, mse=0.000 |
| next_speed | r2=0.148, mse=36.096 | r2=-1.104, mse=88.492 | r2=-0.053, mse=44.277 | r2=-0.818, mse=76.463 | r2=0.001, mse=42.008 | r2=0.022, mse=41.135 |
| next_unit_vec_to_goal | — | r2=-1.371, mse=1.156 | r2=0.089, mse=0.448 | r2=-0.673, mse=0.834 | r2=-0.146, mse=0.556 | r2=0.095, mse=0.444 |

## Hook: `hook_actor_h5`

| target | velocity_persistence | baseline | geom_only | sensors+action | hook | hook+action |
|---|---|---|---|---|---|---|
| next_ap | — | r2=-4.314, mse=0.185 | r2=-0.557, mse=0.077 | r2=-4.831, mse=0.173 | r2=-5.285, mse=0.151 | r2=-5.218, mse=0.144 |
| next_delta_xy | r2=-2.063, mse=110.542 | r2=-2.481, mse=27.715 | r2=0.252, mse=19.518 | r2=0.518, mse=22.577 | r2=-0.579, mse=62.575 | r2=-0.768, mse=62.898 |
| next_direction_xy | r2=-0.455, mse=0.943 | r2=0.084, mse=0.476 | r2=0.153, mse=0.404 | r2=-0.036, mse=0.481 | r2=-0.498, mse=0.827 | r2=-0.508, mse=0.823 |
| next_dist_to_goal | — | r2=-0.413, mse=23.557 | r2=-0.008, mse=16.808 | r2=-0.232, mse=20.553 | r2=-0.581, mse=26.369 | r2=-0.610, mse=26.849 |
| next_obs | — | r2=0.526, mse=4.949 | r2=-0.032, mse=3.765 | r2=0.730, mse=4.136 | r2=-0.323, mse=4.345 | r2=-0.280, mse=4.123 |
| next_positives | — | r2=-0.408, mse=0.484 | r2=-0.234, mse=0.224 | r2=-0.509, mse=0.493 | r2=0.240, mse=0.094 | r2=0.240, mse=0.094 |
| next_speed | r2=0.148, mse=36.096 | r2=-1.104, mse=88.492 | r2=-0.053, mse=44.277 | r2=-0.818, mse=76.463 | r2=-0.023, mse=43.026 | r2=-0.083, mse=45.550 |
| next_unit_vec_to_goal | — | r2=-1.371, mse=1.156 | r2=0.089, mse=0.448 | r2=-0.673, mse=0.834 | r2=-0.231, mse=0.605 | r2=-0.214, mse=0.600 |