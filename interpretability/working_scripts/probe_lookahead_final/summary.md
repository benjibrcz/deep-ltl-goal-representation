# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_lookahead_fixed.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next1_obs, next3_obs, next5_obs, next1_agent_pos, next3_agent_pos, next5_agent_pos  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | r2=-24.437, mse=130.294 | r2=-19.860, mse=69.844 | r2=-820608829865322.750, mse=2.515 | r2=-820608829865322.625, mse=2.209 |
| next3_obs | r2=-23.756, mse=53.407 | r2=-23.007, mse=36.729 | r2=-871896880993707.750, mse=1.725 | r2=-871896880993707.750, mse=2.217 |
| next5_obs | r2=-27.362, mse=15.440 | r2=-25.682, mse=26.885 | r2=-930023338939212.250, mse=1.166 | r2=-930023338939212.250, mse=1.269 |
| next1_agent_pos | r2=-109.450, mse=4763.527 | r2=-64.674, mse=2405.650 | r2=-2.053, mse=94.479 | r2=-1.865, mse=82.944 |
| next3_agent_pos | r2=-28.044, mse=1900.106 | r2=-22.745, mse=1224.273 | r2=-0.943, mse=69.078 | r2=-1.011, mse=88.902 |
| next5_agent_pos | r2=-41.711, mse=591.321 | r2=-48.373, mse=988.131 | r2=-0.638, mse=48.529 | r2=-0.615, mse=48.034 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | r2=-24.437, mse=130.294 | r2=-19.860, mse=69.844 | r2=-820608829865325.625, mse=0.905 | r2=-820608829865326.750, mse=1.752 |
| next3_obs | r2=-23.756, mse=53.407 | r2=-23.007, mse=36.729 | r2=-871896880993710.750, mse=0.931 | r2=-871896880993711.625, mse=2.644 |
| next5_obs | r2=-27.362, mse=15.440 | r2=-25.682, mse=26.885 | r2=-930023338939215.000, mse=0.890 | r2=-930023338939215.250, mse=1.035 |
| next1_agent_pos | r2=-109.450, mse=4763.527 | r2=-64.674, mse=2405.650 | r2=-0.042, mse=36.353 | r2=-0.762, mse=70.669 |
| next3_agent_pos | r2=-28.044, mse=1900.106 | r2=-22.745, mse=1224.273 | r2=-0.088, mse=46.897 | r2=-1.071, mse=112.269 |
| next5_agent_pos | r2=-41.711, mse=591.321 | r2=-48.373, mse=988.131 | r2=-0.212, mse=56.156 | r2=-0.164, mse=54.328 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | r2=-24.437, mse=130.294 | r2=-19.860, mse=69.844 | r2=-820608829865326.375, mse=1.756 | r2=-820608829865326.375, mse=1.738 |
| next3_obs | r2=-23.756, mse=53.407 | r2=-23.007, mse=36.729 | r2=-871896880993710.875, mse=1.624 | r2=-871896880993710.875, mse=1.672 |
| next5_obs | r2=-27.362, mse=15.440 | r2=-25.682, mse=26.885 | r2=-930023338939214.750, mse=2.284 | r2=-930023338939214.750, mse=2.299 |
| next1_agent_pos | r2=-109.450, mse=4763.527 | r2=-64.674, mse=2405.650 | r2=-2.177, mse=66.604 | r2=-2.149, mse=65.841 |
| next3_agent_pos | r2=-28.044, mse=1900.106 | r2=-22.745, mse=1224.273 | r2=-1.313, mse=74.060 | r2=-1.366, mse=77.036 |
| next5_agent_pos | r2=-41.711, mse=591.321 | r2=-48.373, mse=988.131 | r2=-0.762, mse=74.739 | r2=-0.730, mse=74.845 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | r2=-24.437, mse=130.294 | r2=-19.860, mse=69.844 | r2=-820608829865326.625, mse=3.278 | r2=-820608829865326.500, mse=3.902 |
| next3_obs | r2=-23.756, mse=53.407 | r2=-23.007, mse=36.729 | r2=-871896880993711.625, mse=1.121 | r2=-871896880993711.500, mse=1.337 |
| next5_obs | r2=-27.362, mse=15.440 | r2=-25.682, mse=26.885 | r2=-930023338939215.750, mse=1.483 | r2=-930023338939215.625, mse=1.414 |
| next1_agent_pos | r2=-109.450, mse=4763.527 | r2=-64.674, mse=2405.650 | r2=-3.178, mse=136.684 | r2=-3.551, mse=163.783 |
| next3_agent_pos | r2=-28.044, mse=1900.106 | r2=-22.745, mse=1224.273 | r2=-1.580, mse=53.984 | r2=-1.781, mse=63.987 |
| next5_agent_pos | r2=-41.711, mse=591.321 | r2=-48.373, mse=988.131 | r2=-1.847, mse=78.726 | r2=-1.851, mse=77.743 |