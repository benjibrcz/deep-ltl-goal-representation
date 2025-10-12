# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_new_fields_fixed.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next_obs, next_ap  
Include action: True  


## Hook: `hook_env_mlp1`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.197, mse=0.860 | r2=-2060169289.196, mse=0.857 |
| next_ap | acc=0.689, f1_macro=0.584, log_loss=0.585 | acc=0.687, f1_macro=0.581, log_loss=0.584 | acc=0.695, f1_macro=0.595, log_loss=0.579 | acc=0.701, f1_macro=0.603, log_loss=0.570 |

## Hook: `hook_ltl_rnn_h`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169290.046, mse=1.694 | r2=-2060169289.990, mse=1.418 |
| next_ap | acc=0.689, f1_macro=0.584, log_loss=0.585 | acc=0.687, f1_macro=0.581, log_loss=0.584 | acc=0.667, f1_macro=0.400, log_loss=0.652 | acc=0.667, f1_macro=0.400, log_loss=0.642 |

## Hook: `hook_actor_h5`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.802, mse=0.944 | r2=-2060169289.796, mse=0.943 |
| next_ap | acc=0.689, f1_macro=0.584, log_loss=0.585 | acc=0.687, f1_macro=0.581, log_loss=0.584 | acc=0.680, f1_macro=0.498, log_loss=0.597 | acc=0.685, f1_macro=0.508, log_loss=0.596 |

## Hook: `hook_critic_mlp0`

| target | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|
| next_obs | r2=0.973, mse=0.979 | r2=0.974, mse=0.967 | r2=-2060169289.436, mse=0.902 | r2=-2060169289.433, mse=0.900 |
| next_ap | acc=0.689, f1_macro=0.584, log_loss=0.585 | acc=0.687, f1_macro=0.581, log_loss=0.584 | acc=0.678, f1_macro=0.524, log_loss=0.602 | acc=0.679, f1_macro=0.530, log_loss=0.601 |