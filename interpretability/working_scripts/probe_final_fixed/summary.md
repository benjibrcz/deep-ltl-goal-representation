# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_next0_large.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next0_obs, next0_agent_pos  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.689, mse=5.428 | r2=0.690, mse=5.424 | r2=-2016310766118778.000, mse=5.507 | r2=-2016310766118778.000, mse=5.506 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.007, mse=27.373 | r2=-0.006, mse=27.371 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.689, mse=5.428 | r2=0.690, mse=5.424 | r2=-2016310766118778.500, mse=5.434 | r2=-2016310766118778.500, mse=5.430 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.001, mse=26.931 | r2=0.001, mse=26.912 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.689, mse=5.428 | r2=0.690, mse=5.424 | r2=-2016310766118778.500, mse=5.468 | r2=-2016310766118778.500, mse=5.473 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=-0.002, mse=27.100 | r2=-0.004, mse=27.127 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next0_obs | r2=0.689, mse=5.428 | r2=0.690, mse=5.424 | r2=-2016310766118778.000, mse=5.431 | r2=-2016310766118778.500, mse=5.433 |
| next0_agent_pos | r2=-0.001, mse=26.982 | r2=0.002, mse=26.960 | r2=0.008, mse=27.016 | r2=0.007, mse=27.030 |