# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_new_fields_patched.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next_obs, next_ap  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.197, mse=0.860 | r2=-2060169289.196, mse=0.857 |
| next_ap | ERROR: float() argument must be a string or a real number, not 'set' |  |  |  |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169290.046, mse=1.694 | r2=-2060169289.990, mse=1.418 |
| next_ap | ERROR: float() argument must be a string or a real number, not 'set' |  |  |  |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.802, mse=0.944 | r2=-2060169289.796, mse=0.943 |
| next_ap | ERROR: float() argument must be a string or a real number, not 'set' |  |  |  |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.436, mse=0.902 | r2=-2060169289.433, mse=0.900 |
| next_ap | ERROR: float() argument must be a string or a real number, not 'set' |  |  |  |