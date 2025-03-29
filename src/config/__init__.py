from .experiment_config import *
from .ppo_config import *
from .model_config import zones, letter, flatworld, ModelConfig, SetNetConfig

model_configs = {
    'PointLtl2-v0': zones,
    'LetterEnv-v0': letter,
    'FlatWorld-v0': flatworld,
    'FlatWorld-big-8-v0': flatworld,
    'FlatWorld-big-12-v0': flatworld,
    'FlatWorld-big-16-v0': flatworld,
    'FlatWorld-big-20-v0': flatworld,
}

__all__ = ['ExperimentConfig', 'PPOConfig', 'ModelConfig', 'SetNetConfig', 'model_configs']
