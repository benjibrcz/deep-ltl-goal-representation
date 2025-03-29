import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Callable, Optional

import numpy as np
import torch

from ltl.automata import LDBASequence
from sequence.samplers.flatworld_sequence_samplers import flatworld_all_reach_tasks, \
    flatworld_sample_reach_avoid, flatworld_sample_reach_stay, flatworld_sample_reach
from sequence.samplers.sequence_samplers import sample_reach_avoid, all_reach_avoid_tasks, all_reach_tasks, \
    all_reach_stay_tasks, sample_reach_stay


@dataclass
class CurriculumStage(ABC):
    threshold: float | None
    threshold_type: Literal['mean', 'min'] | None

    @abstractmethod
    def sample(self, propositions: list[str]) -> LDBASequence:
        pass

    @abstractmethod
    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


@dataclass
class ExplicitCurriculumStage(CurriculumStage):
    """A curriculum stage in which all tasks are explicitly listed, and sampled from according to previous success."""
    task_fn: Optional[Callable[[list[str]], list[LDBASequence]]]
    eps_task_fn: Optional[Callable[[list[str]], list[LDBASequence]]] = None
    temperature: float = 0.5
    _tasks: list[LDBASequence] | None = None
    _task_success: dict[LDBASequence, float] | None = None

    def sample(self, propositions: list[str]) -> LDBASequence:
        if self._tasks is None:
            self._tasks = []
            if self.task_fn is not None:
                self._tasks += self.task_fn(propositions)
            if self.eps_task_fn is not None:
                self._tasks += self.eps_task_fn(propositions)
        if self._task_success is None:
            return random.choice(self._tasks)
        probs = self.compute_sampling_prob()
        index = np.random.choice(np.arange(len(self._tasks)), p=probs).item()
        return self._tasks[index]

    def compute_sampling_prob(self) -> np.ndarray:
        if len(self._task_success) != len(self._tasks):
            raise ValueError('Task success must be available for all tasks')
        success = torch.tensor([self._task_success[t] for t in self._tasks])
        probs = torch.nn.functional.softmax(-success / self.temperature, dim=0)
        return probs.numpy()

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        if self._task_success is None:
            self._task_success = {k: v for k, v in task_success.items() if k in self._tasks}
            for t in self._tasks:
                if t not in self._task_success:
                    self._task_success[t] = 0.0
        else:
            self._task_success.update(task_success)


@dataclass
class RandomCurriculumStage(CurriculumStage):
    """A curriculum stage in which tasks are sampled randomly."""
    sampler: Callable[[list[str]], LDBASequence]

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.sampler(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


@dataclass
class MultiRandomStage(CurriculumStage):
    """A combination of multiple RandomCurriculumStages with associated sampling probabilities."""
    stages: list[RandomCurriculumStage]
    probs: list[float]

    def sample(self, propositions: list[str]) -> LDBASequence:
        stage = np.random.choice(self.stages, p=self.probs)
        return stage.sample(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float]) -> None:
        pass


class Curriculum:
    def __init__(self, stages: list[CurriculumStage]):
        self.stages = stages
        self.stage_index = 0
        self.num_updates = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def finished(self) -> bool:
        return self.stage_index >= len(self.stages)

    def sample(self, propositions: list[str]) -> LDBASequence:
        return self.current_stage.sample(propositions)

    def update_task_success(self, task_success: dict[LDBASequence, float], verbose=False) -> None:
        if self.current_stage.threshold is None:
            return
        self.num_updates += 1
        self.num_updates %= 100
        self.current_stage.update_task_success(task_success)
        aggr = np.mean if self.current_stage.threshold_type == 'mean' else np.min
        if aggr(list(task_success.values())) >= self.current_stage.threshold:
            if verbose:
                print('=' * 80)
                print(f"Stage {self.stage_index} completed.")
                print('=' * 80)
            self.stage_index += 1
        else:
            if verbose and self.num_updates % 100 == 0:
                print(f"Stage {self.stage_index} not completed.")
                print(f'MEAN: {np.mean(list(task_success.values()))}, THRESHOLD: {self.current_stage.threshold}')


LETTER_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(
        task_fn=all_reach_avoid_tasks(1),
        temperature=0.1,
        threshold=0.95,
        threshold_type='mean',
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
        threshold=0.95,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
        threshold=None,
        threshold_type=None
    ),
])

ZONES_CURRICULUM = Curriculum([
    ExplicitCurriculumStage(  # 0
        task_fn=all_reach_tasks(1),
        temperature=0.5,
        threshold=0.8,
        threshold_type='min',
    ),
    ExplicitCurriculumStage(  # 1
        task_fn=all_reach_tasks(2),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 2
        task_fn=all_reach_avoid_tasks(1),
        threshold=0.95,
        threshold_type='mean'
    ),
    ExplicitCurriculumStage(  # 3
        task_fn=all_reach_avoid_tasks(2),
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 4
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(1, (1, 2), (0, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(30, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.4, 0.6],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 5
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(2, (1, 2), (1, 2)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 1)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=0.9,
        threshold_type='mean'
    ),
    MultiRandomStage(  # 6
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid(3, (1, 2), (0, 3)),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_stay(60, (0, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.8, 0.2],
        threshold=None,
        threshold_type=None
    ),
])

FLATWORLD_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=flatworld_sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=flatworld_sample_reach((1, 2)),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=flatworld_sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])

FLATWORLD_BIG_CURRICULUM = Curriculum([
    MultiRandomStage(  # 0
        stages=[
            RandomCurriculumStage(
                sampler=sample_reach_avoid((1, 2), 1, 1),
                threshold=None,
                threshold_type=None
            ),
            RandomCurriculumStage(
                sampler=sample_reach_avoid((1, 2), 1, 0),
                threshold=None,
                threshold_type=None
            ),
        ],
        probs=[0.6, 0.4],
        threshold=0.8,
        threshold_type='mean'
    ),
    RandomCurriculumStage(
        sampler=sample_reach_avoid((1, 2), (1, 2), (0, 2)),
        threshold=None,
        threshold_type=None
    ),
])
