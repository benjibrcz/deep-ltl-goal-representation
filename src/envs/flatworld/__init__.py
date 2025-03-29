from .flatworld import FlatWorld

from gymnasium.envs.registration import register

register(
    id='FlatWorld-v0',
    entry_point='envs.flatworld.flatworld:FlatWorld',
    kwargs=dict(
        continuous_actions=False
    )
)

for n in [8,12,16,20]:
    register(
        id=f'FlatWorld-big-{n}-v0',
        entry_point='envs.flatworld.flatworld_big:FlatWorldBig',
        kwargs=dict(
            num_colors=n,
            continuous_actions=False
        )
    )