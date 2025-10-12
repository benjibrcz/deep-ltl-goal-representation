# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/ultra_long_analysis/ultra_long_enhanced.npz`  
Hooks: hook_env_mlp1, hook_ltl_rnn_h, hook_actor_h5, hook_critic_mlp0  
Targets: next1_velocity, next5_velocity, next15_velocity, next30_velocity, next50_velocity, next100_velocity  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.001, mse=148.762 | r2=0.476, mse=26.004 | r2=0.475, mse=26.015 | r2=0.476, mse=26.242 | r2=0.475, mse=26.232 |
| next5_velocity | r2=-1.935, mse=148.779 | r2=0.486, mse=26.367 | r2=0.485, mse=26.379 | r2=0.478, mse=26.602 | r2=0.478, mse=26.598 |
| next15_velocity | r2=-1.957, mse=147.122 | r2=0.464, mse=27.874 | r2=0.464, mse=27.910 | r2=0.469, mse=27.536 | r2=0.467, mse=27.614 |
| next30_velocity | r2=-1.936, mse=148.184 | r2=0.452, mse=29.299 | r2=0.451, mse=29.383 | r2=0.464, mse=28.349 | r2=0.464, mse=28.427 |
| next50_velocity | r2=-1.956, mse=147.928 | r2=0.428, mse=30.495 | r2=0.427, mse=30.498 | r2=0.436, mse=29.787 | r2=0.436, mse=29.745 |
| next100_velocity | r2=-1.921, mse=149.875 | r2=0.379, mse=34.977 | r2=0.378, mse=35.115 | r2=0.380, mse=34.980 | r2=0.374, mse=35.568 |

## Hook: `hook_ltl_rnn_h`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.001, mse=148.762 | r2=0.476, mse=26.004 | r2=0.475, mse=26.015 | r2=-0.000, mse=49.659 | r2=0.032, mse=47.178 |
| next5_velocity | r2=-1.935, mse=148.779 | r2=0.486, mse=26.367 | r2=0.485, mse=26.379 | r2=0.000, mse=50.756 | r2=0.028, mse=48.485 |
| next15_velocity | r2=-1.957, mse=147.122 | r2=0.464, mse=27.874 | r2=0.464, mse=27.910 | r2=0.000, mse=50.327 | r2=0.024, mse=48.222 |
| next30_velocity | r2=-1.936, mse=148.184 | r2=0.452, mse=29.299 | r2=0.451, mse=29.383 | r2=0.001, mse=51.048 | r2=0.019, mse=49.178 |
| next50_velocity | r2=-1.956, mse=147.928 | r2=0.428, mse=30.495 | r2=0.427, mse=30.498 | r2=0.002, mse=50.530 | r2=0.022, mse=48.371 |
| next100_velocity | r2=-1.921, mse=149.875 | r2=0.379, mse=34.977 | r2=0.378, mse=35.115 | r2=0.008, mse=52.353 | r2=0.029, mse=49.625 |

## Hook: `hook_actor_h5`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.001, mse=148.762 | r2=0.476, mse=26.004 | r2=0.475, mse=26.015 | r2=0.470, mse=26.134 | r2=0.471, mse=26.151 |
| next5_velocity | r2=-1.935, mse=148.779 | r2=0.486, mse=26.367 | r2=0.485, mse=26.379 | r2=0.479, mse=26.098 | r2=0.478, mse=26.122 |
| next15_velocity | r2=-1.957, mse=147.122 | r2=0.464, mse=27.874 | r2=0.464, mse=27.910 | r2=0.475, mse=26.638 | r2=0.475, mse=26.687 |
| next30_velocity | r2=-1.936, mse=148.184 | r2=0.452, mse=29.299 | r2=0.451, mse=29.383 | r2=0.468, mse=27.457 | r2=0.471, mse=27.290 |
| next50_velocity | r2=-1.956, mse=147.928 | r2=0.428, mse=30.495 | r2=0.427, mse=30.498 | r2=0.450, mse=28.291 | r2=0.453, mse=28.062 |
| next100_velocity | r2=-1.921, mse=149.875 | r2=0.379, mse=34.977 | r2=0.378, mse=35.115 | r2=0.426, mse=31.555 | r2=0.426, mse=31.423 |

## Hook: `hook_critic_mlp0`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.001, mse=148.762 | r2=0.476, mse=26.004 | r2=0.475, mse=26.015 | r2=0.481, mse=25.798 | r2=0.481, mse=25.770 |
| next5_velocity | r2=-1.935, mse=148.779 | r2=0.486, mse=26.367 | r2=0.485, mse=26.379 | r2=0.484, mse=26.178 | r2=0.484, mse=26.191 |
| next15_velocity | r2=-1.957, mse=147.122 | r2=0.464, mse=27.874 | r2=0.464, mse=27.910 | r2=0.479, mse=26.842 | r2=0.478, mse=26.847 |
| next30_velocity | r2=-1.936, mse=148.184 | r2=0.452, mse=29.299 | r2=0.451, mse=29.383 | r2=0.482, mse=27.187 | r2=0.481, mse=27.238 |
| next50_velocity | r2=-1.956, mse=147.928 | r2=0.428, mse=30.495 | r2=0.427, mse=30.498 | r2=0.448, mse=28.841 | r2=0.448, mse=28.821 |
| next100_velocity | r2=-1.921, mse=149.875 | r2=0.379, mse=34.977 | r2=0.378, mse=35.115 | r2=0.364, mse=36.115 | r2=0.361, mse=36.353 |