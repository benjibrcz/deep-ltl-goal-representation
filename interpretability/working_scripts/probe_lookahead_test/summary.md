# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_lookahead_small.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next1_obs, next3_obs, next5_obs, next1_agent_pos, next3_agent_pos, next5_agent_pos  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_obs | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_obs | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |
| next1_agent_pos | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_agent_pos | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_agent_pos | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_obs | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_obs | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |
| next1_agent_pos | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_agent_pos | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_agent_pos | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_obs | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_obs | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |
| next1_agent_pos | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_agent_pos | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_agent_pos | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next1_obs | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_obs | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_obs | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |
| next1_agent_pos | ERROR: index 76 is out of bounds for axis 0 with size 76 |  |  |  |
| next3_agent_pos | ERROR: index 68 is out of bounds for axis 0 with size 68 |  |  |  |
| next5_agent_pos | ERROR: index 60 is out of bounds for axis 0 with size 60 |  |  |  |