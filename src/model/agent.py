import numpy as np
import torch

import preprocessing
from model.model import Model
from sequence.search import SequenceSearch


class Agent:
    def __init__(self, model: Model, search: SequenceSearch, propositions: set[str], verbose=False):
        self.model = model
        self.search = search
        self.propositions = propositions
        self.verbose = verbose
        self.sequence = None

    def reset(self):
        self.sequence = None

    def get_action(self, obs, info, deterministic=False) -> np.ndarray:
        if self.sequence is not None and len(self.sequence) > 0:
            current_goal_assignment_set, _ = self.sequence[0]
            
            # The goal is a frozenset of FrozenAssignment objects. We need to check if ANY of them are satisfied.
            # In our case, it's usually just one.
            goal_satisfied = False
            if isinstance(current_goal_assignment_set, (set, frozenset)):
                for frozen_assignment in current_goal_assignment_set:
                    # Extract the set of true propositions from the FrozenAssignment
                    if hasattr(frozen_assignment, 'assignment'):
                        current_goal_props = {prop for prop, val in frozen_assignment.assignment if val}
                        if current_goal_props.issubset(obs['propositions']):
                            goal_satisfied = True
                            break
            
            if goal_satisfied:
                # Debug info
                prop_name = None
                zone_center = None
                zone_radius = None
                if isinstance(current_goal_assignment_set, (set, frozenset)):
                    for frozen_assignment in current_goal_assignment_set:
                        if hasattr(frozen_assignment, 'assignment'):
                            current_goal_props = {prop for prop, val in frozen_assignment.assignment if val}
                            if len(current_goal_props) == 1:
                                prop_name = next(iter(current_goal_props))
                                # Find zone info
                                from envs.flatworld import FlatWorld
                                for c in FlatWorld.CIRCLES:
                                    if c.color == prop_name:
                                        zone_center = c.center
                                        zone_radius = c.radius
                                        break
                agent_pos = obs['features'][:2] if 'features' in obs else None
                dist = np.linalg.norm(agent_pos - zone_center) if agent_pos is not None and zone_center is not None else None
                print(f"Goal satisfied, advancing sequence! Subgoal: {prop_name}, Agent pos: {agent_pos}, Zone center: {zone_center}, Dist: {dist}, Radius: {zone_radius}")
                self.sequence = self.sequence[1:]

        if 'ldba_state_changed' in info or self.sequence is None or len(self.sequence) == 0:
            self.sequence = self.search(obs['ldba'], obs['ldba_state'], obs)
            if self.verbose:
                print(f'Selected sequence: {self.sequence}')
        assert self.sequence is not None
        obs['goal'] = self.sequence
        return self.forward(obs, deterministic)

    def forward(self, obs, deterministic=False) -> np.ndarray:
        obs_list = [obs] if not isinstance(obs, list) else obs
        preprocessed = preprocessing.preprocess_obss(obs_list, self.propositions)
        with torch.no_grad():
            dist, value = self.model(preprocessed)
            action = dist.mode if deterministic else dist.sample()
        return action.detach().numpy()
