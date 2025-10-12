# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_velocity_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_speed, next1_speed, next3_speed, next5_speed, next10_speed  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_speed | r2=-0.029, mse=40.453 | r2=-0.016, mse=39.904 | r2=0.255, mse=29.271 | r2=0.255, mse=29.280 |
| next1_speed | r2=-0.031, mse=37.727 | r2=-0.022, mse=37.376 | r2=0.243, mse=27.688 | r2=0.242, mse=27.723 |
| next3_speed | r2=-0.025, mse=38.329 | r2=-0.015, mse=37.951 | r2=0.227, mse=28.907 | r2=0.228, mse=28.891 |
| next5_speed | r2=-0.007, mse=35.157 | r2=-0.000, mse=34.914 | r2=0.218, mse=27.288 | r2=0.218, mse=27.283 |
| next10_speed | r2=-0.018, mse=36.739 | r2=-0.009, mse=36.425 | r2=0.251, mse=27.026 | r2=0.249, mse=27.103 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_speed | r2=-0.029, mse=40.453 | r2=-0.016, mse=39.904 | r2=-0.002, mse=39.362 | r2=-0.002, mse=39.373 |
| next1_speed | r2=-0.031, mse=37.727 | r2=-0.022, mse=37.376 | r2=-0.001, mse=36.619 | r2=-0.002, mse=36.663 |
| next3_speed | r2=-0.025, mse=38.329 | r2=-0.015, mse=37.951 | r2=-0.001, mse=37.429 | r2=0.000, mse=37.399 |
| next5_speed | r2=-0.007, mse=35.157 | r2=-0.000, mse=34.914 | r2=0.000, mse=34.885 | r2=-0.001, mse=34.921 |
| next10_speed | r2=-0.018, mse=36.739 | r2=-0.009, mse=36.425 | r2=-0.000, mse=36.112 | r2=-0.003, mse=36.209 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_speed | r2=-0.029, mse=40.453 | r2=-0.016, mse=39.904 | r2=0.262, mse=29.018 | r2=0.259, mse=29.100 |
| next1_speed | r2=-0.031, mse=37.727 | r2=-0.022, mse=37.376 | r2=0.231, mse=28.150 | r2=0.232, mse=28.117 |
| next3_speed | r2=-0.025, mse=38.329 | r2=-0.015, mse=37.951 | r2=0.233, mse=28.677 | r2=0.234, mse=28.660 |
| next5_speed | r2=-0.007, mse=35.157 | r2=-0.000, mse=34.914 | r2=0.219, mse=27.265 | r2=0.219, mse=27.269 |
| next10_speed | r2=-0.018, mse=36.739 | r2=-0.009, mse=36.425 | r2=0.256, mse=26.855 | r2=0.256, mse=26.854 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_speed | r2=-0.029, mse=40.453 | r2=-0.016, mse=39.904 | r2=0.265, mse=28.887 | r2=0.265, mse=28.887 |
| next1_speed | r2=-0.031, mse=37.727 | r2=-0.022, mse=37.376 | r2=0.247, mse=27.553 | r2=0.246, mse=27.581 |
| next3_speed | r2=-0.025, mse=38.329 | r2=-0.015, mse=37.951 | r2=0.235, mse=28.620 | r2=0.235, mse=28.622 |
| next5_speed | r2=-0.007, mse=35.157 | r2=-0.000, mse=34.914 | r2=0.226, mse=27.009 | r2=0.226, mse=27.013 |
| next10_speed | r2=-0.018, mse=36.739 | r2=-0.009, mse=36.425 | r2=0.254, mse=26.926 | r2=0.253, mse=26.972 |