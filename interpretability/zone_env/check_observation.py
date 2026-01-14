#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from envs import make_env
from ltl import FixedSampler

env = make_env('PointLtl2-v0', FixedSampler.partial('FG blue'), sequence=False, render_mode=None)
ret = env.reset()
obs, info = ret if isinstance(ret, tuple) and len(ret) == 2 else (ret, {})

print('All observation keys:', list(obs.keys()))
for key in obs.keys():
    if hasattr(obs[key], 'shape'):
        print(f'{key}: {type(obs[key])} - {obs[key].shape}')
    elif hasattr(obs[key], '__len__'):
        print(f'{key}: {type(obs[key])} - {len(obs[key])}')
    else:
        print(f'{key}: {type(obs[key])} - no length')

# Check if agent_pos is in the observation
if 'agent_pos' in obs:
    print(f"\nagent_pos found in observation: {obs['agent_pos']}")
else:
    print("\nagent_pos NOT found in observation")

# Check the features vector breakdown
features = obs['features']
print(f"\nFeatures breakdown:")
print(f"  Total length: {len(features)}")
print(f"  First 3 (agent sensors): {features[0:3]}")
print(f"  Next 16 (zone lidar): {features[3:19]}")
print(f"  Next 16 (wall lidar): {features[19:35]}")
print(f"  Next 4 (wall sensor): {features[35:39]}")
print(f"  Remaining: {features[39:]}")

env.close() 