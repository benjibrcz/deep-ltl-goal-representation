#!/usr/bin/env python3
import os, sys
import torch
import numpy as np

# 1) make sure we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 2) imports
from utils.model_store   import ModelStore
from model.model         import build_model
from config              import model_configs
from ltl                 import FixedSampler
from envs                import make_env

# 3) configuration – tweak these
ENV_NAME = "PointLtl2-v0"
EXP      = "big_test"               # folder under experiments/ppo/PointLtl2-v0/
SEED     = 0
FORMULA  = "F blue & F green"       # your LTL goal
NUM_EP   = 50

CKPT_PATH = os.path.join(
    "experiments","ppo",ENV_NAME,EXP,str(SEED),"checkpoint.pt"
)

# 4) load the LTL vocab & checkpoint
store  = ModelStore(ENV_NAME, EXP, SEED)
store.load_vocab()
status = store.load_training_status(map_location="cpu")

# 5) rebuild exactly the same Model you trained
cfg       = model_configs[ENV_NAME]
# we need a “dummy” env so build_model knows the obs‐space and prop set
dummy_env = make_env(ENV_NAME, FixedSampler.partial(FORMULA))
model     = build_model(dummy_env, status, cfg)
model.eval()

# 6) now create the real eval env with your fixed formula
env = make_env(ENV_NAME, FixedSampler.partial(FORMULA))

# 7) run deterministic rollouts and count successes
successes = 0
for _ in range(NUM_EP):
    obs = env.reset()   # env.reset() returns obs
    done = False
    while not done:
        x     = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist, _ = model(x)  # `model` returns (action_distribution, value)
        action = dist.probs.argmax(dim=-1).item()
        obs, _, done, info = env.step(action)
    if info.get("success", False):
        successes += 1

print(f"Success rate over {NUM_EP} episodes: {successes}/{NUM_EP} = {successes/NUM_EP:.2%}")
