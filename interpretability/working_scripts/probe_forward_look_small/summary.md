# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_small_enhanced.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next_obs, next_ap, next_agent_pos, next_reward, next_done  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=-218657.611, mse=234.740 | r2=-218724.813, mse=264.131 | r2=-19510673143404.531, mse=2.549 | r2=-19510673143400.246, mse=2.734 |
| next_ap | ERROR: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0 |  |  |  |
| next_agent_pos | r2=-387.761, mse=8942.605 | r2=-409.165, mse=10285.750 | r2=-3.093, mse=88.160 | r2=-4.112, mse=97.301 |
| next_reward | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |
| next_done | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=-218657.611, mse=234.740 | r2=-218724.813, mse=264.131 | r2=-19510673193089.730, mse=0.815 | r2=-19510673189918.094, mse=0.962 |
| next_ap | ERROR: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0 |  |  |  |
| next_agent_pos | r2=-387.761, mse=8942.605 | r2=-409.165, mse=10285.750 | r2=-0.005, mse=28.234 | r2=-0.166, mse=32.912 |
| next_reward | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |
| next_done | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=-218657.611, mse=234.740 | r2=-218724.813, mse=264.131 | r2=-19510673173594.777, mse=1.763 | r2=-19510673176110.059, mse=1.778 |
| next_ap | ERROR: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0 |  |  |  |
| next_agent_pos | r2=-387.761, mse=8942.605 | r2=-409.165, mse=10285.750 | r2=-2.088, mse=60.331 | r2=-2.179, mse=61.275 |
| next_reward | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |
| next_done | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=-218657.611, mse=234.740 | r2=-218724.813, mse=264.131 | r2=-19510673196466.910, mse=3.747 | r2=-19510673196695.586, mse=3.358 |
| next_ap | ERROR: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0 |  |  |  |
| next_agent_pos | r2=-387.761, mse=8942.605 | r2=-409.165, mse=10285.750 | r2=-3.342, mse=141.519 | r2=-3.052, mse=126.476 |
| next_reward | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |
| next_done | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 | r2=1.000, mse=0.000 |