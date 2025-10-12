# Forward-look Probe (v0)

NPZ: `interpretability/working_scripts/test_long_horizon_velocity_enhanced.npz`  
Hooks: hook_env_mlp1  
Targets: next1_velocity, next2_velocity, next3_velocity, next4_velocity, next5_velocity, next7_velocity, next10_velocity, next15_velocity, next20_velocity, next25_velocity, next30_velocity, next40_velocity, next50_velocity  
Include action: True  


## Hook: `hook_env_mlp1`

| target | velocity_persistence | baseline | sensors+action | hook | hook+action |
|---|---:|---:|---:|---:|---:|
| next1_velocity | r2=-2.034, mse=146.804 | r2=0.408, mse=28.530 | r2=0.400, mse=28.611 | r2=0.449, mse=27.203 | r2=0.439, mse=27.418 |
| next2_velocity | r2=-1.997, mse=156.468 | r2=0.424, mse=29.053 | r2=0.421, mse=29.197 | r2=0.462, mse=27.553 | r2=0.454, mse=27.653 |
| next3_velocity | r2=-1.984, mse=157.575 | r2=0.442, mse=28.696 | r2=0.438, mse=28.628 | r2=0.475, mse=26.821 | r2=0.472, mse=26.880 |
| next4_velocity | r2=-2.003, mse=152.265 | r2=0.447, mse=28.975 | r2=0.446, mse=28.801 | r2=0.470, mse=27.379 | r2=0.462, mse=27.574 |
| next5_velocity | r2=-2.017, mse=153.951 | r2=0.454, mse=28.412 | r2=0.450, mse=28.635 | r2=0.457, mse=28.037 | r2=0.450, mse=28.068 |
| next7_velocity | r2=-1.974, mse=154.962 | r2=0.464, mse=28.646 | r2=0.466, mse=28.514 | r2=0.475, mse=27.648 | r2=0.465, mse=27.781 |
| next10_velocity | r2=-2.008, mse=153.910 | r2=0.445, mse=29.467 | r2=0.447, mse=29.170 | r2=0.465, mse=28.026 | r2=0.464, mse=28.096 |
| next15_velocity | r2=-1.993, mse=159.527 | r2=0.479, mse=28.004 | r2=0.481, mse=27.867 | r2=0.493, mse=27.163 | r2=0.492, mse=27.281 |
| next20_velocity | r2=-1.947, mse=153.067 | r2=0.449, mse=28.338 | r2=0.449, mse=28.391 | r2=0.458, mse=27.940 | r2=0.454, mse=28.266 |
| next25_velocity | r2=-1.970, mse=156.517 | r2=0.438, mse=29.019 | r2=0.440, mse=28.903 | r2=0.456, mse=28.151 | r2=0.455, mse=28.080 |
| next30_velocity | r2=-1.972, mse=161.401 | r2=0.432, mse=30.085 | r2=0.434, mse=29.951 | r2=0.454, mse=28.643 | r2=0.452, mse=28.784 |
| next40_velocity | r2=-1.938, mse=157.644 | r2=0.403, mse=31.902 | r2=0.403, mse=31.721 | r2=0.419, mse=31.129 | r2=0.417, mse=31.236 |
| next50_velocity | r2=-1.942, mse=161.295 | r2=0.357, mse=36.671 | r2=0.355, mse=36.454 | r2=0.425, mse=31.499 | r2=0.423, mse=31.452 |