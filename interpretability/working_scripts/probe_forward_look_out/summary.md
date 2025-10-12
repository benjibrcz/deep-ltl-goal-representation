# Forward-look Probe (v0)

NPZ: `/Users/benji.berczi/Documents/deep-ltl/interpretability/working_scripts/rollouts_stateful.npz`  
Hooks: hook_env_mlp1, hook_env_mlp3, hook_ltl_rnn_h, hook_actor_h5  
Targets: next_ap, next_delta_xy, next_direction_xy, next_dist_to_goal, next_obs, next_positives, next_speed, next_unit_vec_to_goal  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_ap | r2=0.338, mse=0.042 | r2=0.339, mse=0.042 | r2=0.397, mse=0.038 | r2=0.398, mse=0.038 |
| next_delta_xy | r2=0.324, mse=23.522 | r2=0.639, mse=22.766 | r2=0.351, mse=23.449 | r2=0.638, mse=22.762 |
| next_direction_xy | r2=0.304, mse=0.364 | r2=0.335, mse=0.358 | r2=0.340, mse=0.350 | r2=0.369, mse=0.344 |
| next_dist_to_goal | — | — | — | — |
| next_obs | r2=0.734, mse=4.704 | r2=0.776, mse=4.553 | r2=0.711, mse=4.688 | r2=0.750, mse=4.548 |
| next_positives | r2=0.423, mse=0.107 | r2=0.431, mse=0.105 | r2=0.540, mse=0.085 | r2=0.546, mse=0.084 |
| next_speed | r2=0.072, mse=39.145 | r2=0.088, mse=38.471 | r2=0.245, mse=31.866 | r2=0.248, mse=31.742 |
| next_unit_vec_to_goal | — | — | — | — |

## Hook: `hook_env_mlp3`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_ap | r2=0.338, mse=0.042 | r2=0.339, mse=0.042 | r2=0.317, mse=0.043 | r2=0.320, mse=0.043 |
| next_delta_xy | r2=0.324, mse=23.522 | r2=0.639, mse=22.766 | r2=0.304, mse=24.324 | r2=0.572, mse=23.680 |
| next_direction_xy | r2=0.304, mse=0.364 | r2=0.335, mse=0.358 | r2=0.301, mse=0.362 | r2=0.321, mse=0.358 |
| next_dist_to_goal | — | — | — | — |
| next_obs | r2=0.734, mse=4.704 | r2=0.776, mse=4.553 | r2=0.447, mse=4.686 | r2=0.489, mse=4.537 |
| next_positives | r2=0.423, mse=0.107 | r2=0.431, mse=0.105 | r2=0.535, mse=0.087 | r2=0.544, mse=0.085 |
| next_speed | r2=0.072, mse=39.145 | r2=0.088, mse=38.471 | r2=0.138, mse=36.372 | r2=0.145, mse=36.062 |
| next_unit_vec_to_goal | — | — | — | — |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_ap | r2=0.338, mse=0.042 | r2=0.339, mse=0.042 | r2=0.036, mse=0.062 | r2=0.039, mse=0.062 |
| next_delta_xy | r2=0.324, mse=23.522 | r2=0.639, mse=22.766 | r2=-0.001, mse=41.725 | r2=0.022, mse=41.092 |
| next_direction_xy | r2=0.304, mse=0.364 | r2=0.335, mse=0.358 | r2=0.003, mse=0.499 | r2=0.013, mse=0.497 |
| next_dist_to_goal | — | — | — | — |
| next_obs | r2=0.734, mse=4.704 | r2=0.776, mse=4.553 | r2=0.014, mse=4.904 | r2=0.070, mse=4.721 |
| next_positives | r2=0.423, mse=0.107 | r2=0.431, mse=0.105 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |
| next_speed | r2=0.072, mse=39.145 | r2=0.088, mse=38.471 | r2=0.015, mse=41.564 | r2=0.035, mse=40.688 |
| next_unit_vec_to_goal | — | — | — | — |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_ap | r2=0.338, mse=0.042 | r2=0.339, mse=0.042 | r2=0.264, mse=0.047 | r2=0.267, mse=0.046 |
| next_delta_xy | r2=0.324, mse=23.522 | r2=0.639, mse=22.766 | r2=0.206, mse=30.809 | r2=0.265, mse=30.605 |
| next_direction_xy | r2=0.304, mse=0.364 | r2=0.335, mse=0.358 | r2=0.199, mse=0.406 | r2=0.199, mse=0.406 |
| next_dist_to_goal | — | — | — | — |
| next_obs | r2=0.734, mse=4.704 | r2=0.776, mse=4.553 | r2=0.243, mse=4.735 | r2=0.270, mse=4.634 |
| next_positives | r2=0.423, mse=0.107 | r2=0.431, mse=0.105 | r2=0.910, mse=0.016 | r2=0.911, mse=0.016 |
| next_speed | r2=0.072, mse=39.145 | r2=0.088, mse=38.471 | r2=0.168, mse=35.097 | r2=0.171, mse=34.977 |
| next_unit_vec_to_goal | — | — | — | — |