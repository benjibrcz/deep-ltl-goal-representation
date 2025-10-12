# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_next0_large.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_obs, next0_agent_pos  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.944, mse=0.740 | r2=0.944, mse=0.739 | r2=-386137115178.452, mse=0.753 | r2=-386137115178.452, mse=0.753 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.007, mse=27.373 | r2=-0.006, mse=27.371 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.944, mse=0.740 | r2=0.944, mse=0.739 | r2=-386137115179.297, mse=0.764 | r2=-386137115179.220, mse=0.758 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.001, mse=26.931 | r2=0.001, mse=26.912 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.944, mse=0.740 | r2=0.944, mse=0.739 | r2=-386137115179.108, mse=0.759 | r2=-386137115179.107, mse=0.760 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.002, mse=27.100 | r2=-0.004, mse=27.127 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.944, mse=0.740 | r2=0.944, mse=0.739 | r2=-386137115178.734, mse=0.747 | r2=-386137115178.731, mse=0.747 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=0.008, mse=27.016 | r2=0.007, mse=27.030 |