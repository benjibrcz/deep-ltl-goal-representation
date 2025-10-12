# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_semantic_targets.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_agent_pos_semantic, next0_wall_lidar_semantic, next0_zone_lidar_semantic, next0_orientation_semantic, next0_contacts_semantic  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=27.143 | r2=0.001, mse=27.120 | r2=-0.009, mse=27.611 | r2=-0.008, mse=27.606 |
| next0_wall_lidar_semantic | r2=0.862, mse=0.000 | r2=0.862, mse=0.000 | r2=-2520388457648472.500, mse=0.002 | r2=-2520388457648472.500, mse=0.002 |
| next0_zone_lidar_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.871, mse=0.002 | r2=0.871, mse=0.002 |
| next0_orientation_semantic | r2=0.871, mse=0.614 | r2=0.873, mse=0.609 | r2=0.845, mse=0.609 | r2=0.845, mse=0.609 |
| next0_contacts_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.887, mse=0.002 | r2=0.888, mse=0.002 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=27.143 | r2=0.001, mse=27.120 | r2=-0.001, mse=27.095 | r2=0.001, mse=27.080 |
| next0_wall_lidar_semantic | r2=0.862, mse=0.000 | r2=0.862, mse=0.000 | r2=-2520388457648473.000, mse=0.019 | r2=-2520388457648473.000, mse=0.017 |
| next0_zone_lidar_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.008, mse=0.017 | r2=0.081, mse=0.016 |
| next0_orientation_semantic | r2=0.871, mse=0.614 | r2=0.873, mse=0.609 | r2=0.231, mse=0.667 | r2=0.317, mse=0.620 |
| next0_contacts_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=-0.016, mse=0.049 | r2=0.082, mse=0.045 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=27.143 | r2=0.001, mse=27.120 | r2=-0.003, mse=27.298 | r2=-0.004, mse=27.327 |
| next0_wall_lidar_semantic | r2=0.862, mse=0.000 | r2=0.862, mse=0.000 | r2=-2520388457648473.000, mse=0.015 | r2=-2520388457648473.000, mse=0.015 |
| next0_zone_lidar_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.235, mse=0.012 | r2=0.232, mse=0.012 |
| next0_orientation_semantic | r2=0.871, mse=0.614 | r2=0.873, mse=0.609 | r2=0.395, mse=0.621 | r2=0.394, mse=0.621 |
| next0_contacts_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.159, mse=0.029 | r2=0.164, mse=0.029 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_agent_pos_semantic | r2=-0.002, mse=27.143 | r2=0.001, mse=27.120 | r2=0.007, mse=27.193 | r2=0.006, mse=27.196 |
| next0_wall_lidar_semantic | r2=0.862, mse=0.000 | r2=0.862, mse=0.000 | r2=-2520388457648472.500, mse=0.007 | r2=-2520388457648472.500, mse=0.007 |
| next0_zone_lidar_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.550, mse=0.007 | r2=0.558, mse=0.007 |
| next0_orientation_semantic | r2=0.871, mse=0.614 | r2=0.873, mse=0.609 | r2=0.725, mse=0.609 | r2=0.725, mse=0.609 |
| next0_contacts_semantic | r2=0.987, mse=0.000 | r2=0.987, mse=0.000 | r2=0.596, mse=0.009 | r2=0.603, mse=0.008 |