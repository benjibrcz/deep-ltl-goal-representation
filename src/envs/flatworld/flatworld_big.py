# Code adapted from https://github.com/clvoloshin/RL-LTL/blob/main/envs/base_envs/flatworld.py
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces

import matplotlib.pyplot as plt
import seaborn as sns
from gymnasium.core import ActType, ObsType

from ltl.logic import Assignment


@dataclass
class Circle:
    center: np.ndarray
    radius: float
    color: str


class FlatWorldBig(gym.Env):
    ALL_COLORS = ['red', 'magenta', 'yellow', 'orange', 'blue', 'green', 'aqua', 'lavender', 'pink', 'brown',
              'black', 'gray', 'lime', 'teal', 'olive', 'navy', 'maroon', 'silver', 'fuchsia', 'salmon']

    def __init__(self, num_colors: int, continuous_actions=True):
        self.rng = np.random.default_rng()
        self.continuous_actions = continuous_actions
        self.delta_t = 0.08

        self.observation_space = spaces.Box(low=-2., high=2., shape=(2,), dtype=np.float64)
        if continuous_actions:
            self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(9)

        self.agent_pos = np.array([-1, -1])  # will be updated in reset
        self.colors = self.ALL_COLORS[:num_colors]
        self.circles = self.init_circles()

    def init_circles(self):
        circles = []
        for i, color in enumerate(self.colors):
            x = i % 5
            y = i // 5
            y_start = -2 + (4 - (math.ceil(len(self.colors) / 5))) * .5
            radius = .3
            x_spacing = .25
            y_spacing = .5
            center = np.array(
                [-2 + (2 * radius + x_spacing) * x + radius, y_start + (2 * radius + y_spacing) * y + radius])
            circles.append(Circle(center=center, radius=radius, color=color))
        return circles

    def get_active_propositions(self) -> set[str]:
        props = set()
        for circle in self.circles:
            if np.linalg.norm(self.agent_pos - circle.center) < circle.radius:
                props.add(circle.color)
        assert len(props) <= 1
        return props

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = self.rng.uniform(low=-2., high=2., size=(2,))
        while len(self.get_active_propositions()) > 0:
            self.agent_pos = self.rng.uniform(low=-2., high=2., size=(2,))
        return self.agent_pos.copy(), {'propositions': set()}

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        if not self.continuous_actions:
            if action == 0:
                action = np.array([0, 1])
            elif action == 1:
                action = np.array([1, 0])
            elif action == 2:
                action = np.array([0, -1])
            elif action == 3:
                action = np.array([-1, 0])
            elif action == 4:
                action = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
            elif action == 5:
                action = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
            elif action == 6:
                action = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])
            elif action == 7:
                action = np.array([-1 / np.sqrt(2), -1 / np.sqrt(2)])
            elif action == 8:
                action = np.array([0, 0])
            else:
                raise ValueError(f"Invalid action: {action}")

        action = np.clip(action, -1, 1)
        self.agent_pos = self.agent_pos + action.flatten() * self.delta_t
        terminated = False
        reward = 0.0
        if (self.agent_pos < -2).any() or (self.agent_pos > 2).any():
            terminated = True
            reward = -1.0
        return self.agent_pos.copy(), reward, terminated, False, {'propositions': self.get_active_propositions()}

    def get_propositions(self):
        return sorted(list({circle.color for circle in self.circles}))

    def get_possible_assignments(self) -> list[Assignment]:
        props = set(self.get_propositions())
        return [
            Assignment.where(color, propositions=props)
            for color in self.colors
        ] + [Assignment.zero_propositions(props)]

    def render(self, trajectory: list[np.ndarray] = None, ax=None):
        if trajectory is None:
            trajectory = []
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for circle in self.circles:
            xy = (float(circle.center[0]), float(circle.center[1]))
            patch = plt.Circle(xy, circle.radius, color=circle.color, fill=True, alpha=.2)
            ax.add_patch(patch)

        if len(trajectory) > 0:
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='green', marker='o',
                    linestyle='dashed',
                    linewidth=2, markersize=1)
            ax.scatter([trajectory[0, 0]], [trajectory[0, 1]], s=100, marker='o', c="orange")
            ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], s=100, marker='o', c="g")
        ax.axis('square')
        hide_ticks(ax.xaxis)
        hide_ticks(ax.yaxis)
        ax.set_xlim([-2.1, 2.1])
        ax.set_ylim([-2.1, 2.1])


def hide_ticks(axis):
    for tick in axis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)


if __name__ == '__main__':
    env = FlatWorldBig(12, continuous_actions=False)
    obs, _ = env.reset()
    trajectory = [obs]
    colors = []
    for i in range(300):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        if len(info['propositions']) > 0:
            colors.append(info['propositions'])
        trajectory.append(obs)
        if term or trunc:
            break
    print(colors)
    env.render(trajectory)
    plt.show()
