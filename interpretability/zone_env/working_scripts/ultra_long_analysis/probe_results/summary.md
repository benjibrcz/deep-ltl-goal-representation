# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/ultra_long_analysis/ultra_long_enhanced.npz`  
Hooks: hook_env_mlp1  
Targets: next1_velocity, next2_velocity, next3_velocity, next4_velocity, next5_velocity, next7_velocity, next10_velocity, next15_velocity, next20_velocity, next25_velocity, next30_velocity, next35_velocity, next40_velocity, next45_velocity, next50_velocity, next60_velocity, next70_velocity, next80_velocity, next90_velocity, next100_velocity  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.001, mse=148.762 | r2=0.476, mse=26.004 | r2=0.475, mse=26.015 | r2=0.476, mse=26.242 | r2=0.475, mse=26.232 |
| next2_velocity | r2=-1.967, mse=147.141 | r2=0.482, mse=26.068 | r2=0.483, mse=26.045 | r2=0.479, mse=26.206 | r2=0.478, mse=26.247 |
| next3_velocity | r2=-1.942, mse=149.395 | r2=0.495, mse=26.196 | r2=0.494, mse=26.195 | r2=0.493, mse=26.156 | r2=0.493, mse=26.154 |
| next4_velocity | r2=-1.957, mse=149.836 | r2=0.494, mse=26.307 | r2=0.494, mse=26.290 | r2=0.495, mse=26.165 | r2=0.495, mse=26.179 |
| next5_velocity | r2=-1.935, mse=148.779 | r2=0.486, mse=26.367 | r2=0.485, mse=26.379 | r2=0.478, mse=26.602 | r2=0.478, mse=26.598 |
| next7_velocity | r2=-1.918, mse=147.599 | r2=0.483, mse=26.997 | r2=0.482, mse=26.999 | r2=0.484, mse=26.522 | r2=0.483, mse=26.533 |
| next10_velocity | r2=-1.955, mse=151.231 | r2=0.483, mse=26.964 | r2=0.483, mse=26.989 | r2=0.481, mse=27.051 | r2=0.481, mse=27.017 |
| next15_velocity | r2=-1.957, mse=147.122 | r2=0.464, mse=27.874 | r2=0.464, mse=27.910 | r2=0.469, mse=27.536 | r2=0.467, mse=27.614 |
| next20_velocity | r2=-1.934, mse=152.306 | r2=0.460, mse=28.416 | r2=0.459, mse=28.418 | r2=0.471, mse=27.594 | r2=0.469, mse=27.589 |
| next25_velocity | r2=-1.950, mse=148.885 | r2=0.445, mse=28.715 | r2=0.445, mse=28.739 | r2=0.453, mse=28.521 | r2=0.450, mse=28.700 |
| next30_velocity | r2=-1.936, mse=148.184 | r2=0.452, mse=29.299 | r2=0.451, mse=29.383 | r2=0.464, mse=28.349 | r2=0.464, mse=28.427 |
| next35_velocity | r2=-1.965, mse=153.037 | r2=0.458, mse=28.790 | r2=0.458, mse=28.771 | r2=0.462, mse=28.227 | r2=0.462, mse=28.204 |
| next40_velocity | r2=-1.964, mse=149.135 | r2=0.451, mse=29.159 | r2=0.452, mse=29.161 | r2=0.448, mse=28.800 | r2=0.450, mse=28.758 |
| next45_velocity | r2=-1.946, mse=148.368 | r2=0.447, mse=30.062 | r2=0.446, mse=30.101 | r2=0.448, mse=29.799 | r2=0.450, mse=29.732 |
| next50_velocity | r2=-1.956, mse=147.928 | r2=0.428, mse=30.495 | r2=0.427, mse=30.498 | r2=0.436, mse=29.787 | r2=0.436, mse=29.745 |
| next60_velocity | r2=-1.972, mse=149.710 | r2=0.433, mse=30.434 | r2=0.433, mse=30.443 | r2=0.436, mse=30.300 | r2=0.435, mse=30.334 |
| next70_velocity | r2=-1.995, mse=151.581 | r2=0.389, mse=33.222 | r2=0.389, mse=33.247 | r2=0.426, mse=30.990 | r2=0.426, mse=31.036 |
| next80_velocity | r2=-1.965, mse=153.240 | r2=0.413, mse=32.876 | r2=0.411, mse=32.981 | r2=0.422, mse=32.621 | r2=0.422, mse=32.732 |
| next90_velocity | r2=-1.914, mse=154.230 | r2=0.429, mse=33.213 | r2=0.429, mse=33.296 | r2=0.440, mse=32.692 | r2=0.439, mse=32.883 |
| next100_velocity | r2=-1.921, mse=149.875 | r2=0.379, mse=34.977 | r2=0.378, mse=35.115 | r2=0.380, mse=34.980 | r2=0.374, mse=35.568 |