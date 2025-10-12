import functools
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import WrapperObsType, WrapperActType

from envs import get_env_attr
from ltl.automata import ltl2ldba, LDBA, LDBASequence


class LDBAWrapper(gymnasium.Wrapper):
    """
    Wrapper that keeps track of LTL goal satisfaction using an LDBA, which is added to the observation space.
    """

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gymnasium.spaces.Dict):
            raise ValueError('LDBA wrapper requires dict observations')
        if 'goal' not in env.observation_space.spaces:
            raise ValueError('LDBA wrapper requires goal in observation space')
        self.terminate_on_acceptance = False
        self.ldba = None
        self.ldba_state = None
        self.num_accepting_visits = 0
        self.obs = None
        self.info = None

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Support both array-like and scalar actions when checking for EPSILON
        eq_eps = (action == LDBASequence.EPSILON)
        try:
            is_all_eps = eq_eps.all()
        except AttributeError:
            # Scalar bool
            is_all_eps = bool(eq_eps)
        if is_all_eps:
            obs, reward, terminated, truncated, info = self.obs, 0.0, False, False, self.info
            take_epsilon = True
        else:
            try:
                assert not (eq_eps.any())
            except AttributeError:
                # Scalar path: already known to be False
                pass
            obs, reward, terminated, truncated, info = super().step(action)
            take_epsilon = False
            self.obs = obs
            self.info = info
        new_ldba_state, accepting = self.ldba.get_next_state(self.ldba_state, info['propositions'], take_epsilon)
        if new_ldba_state != self.ldba_state:
            self.ldba_state = new_ldba_state
            info['ldba_state_changed'] = True
        self.complete_observation(obs, info)
        if self.terminate_on_acceptance and accepting:
            terminated = True
            info['success'] = True
        if accepting:
            self.num_accepting_visits += 1
        scc = self.ldba.state_to_scc[self.ldba_state]
        if scc.bottom and not scc.accepting:
            terminated = True
            info['violation'] = True
        info['accepting'] = accepting
        info['num_accepting_visits'] = self.num_accepting_visits
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        self.info= info
        self.ldba = self.construct_ldba(obs['goal'])
        self.terminate_on_acceptance = self.ldba.is_finite_specification()
        self.ldba_state = self.ldba.initial_state
        self.num_accepting_visits = 0
        self.complete_observation(obs, info)
        info['ldba_state_changed'] = True
        return obs, info

    def complete_observation(self, obs: WrapperObsType, info: dict[str, Any] = None):
        obs['ldba'] = self.ldba
        obs['ldba_state'] = self.ldba_state
        obs['propositions'] = info['propositions']
    
    def set_goal(self, formula: str, *, reset_ldba_state: bool = True, reset_visit_counter: bool = True) -> None:
        """
        Switch the LDBA to a new goal *without* resetting the underlying physics env.

        - Rebuilds the LDBA for `formula`
        - Resets the LDBA state to its initial state (configurable)
        - Optionally clears the accepting-visit counter
        - Updates the observation view so planners immediately see the new LDBA

        Note: we leave the underlying env state intact. On the next `step`, the LDBA
        will advance based on the current propositions of the ongoing world.
        """
        # Rebuild LDBA for the new goal
        self.ldba = self.construct_ldba(formula)
        self.terminate_on_acceptance = self.ldba.is_finite_specification()

        if reset_ldba_state or (self.ldba_state is None):
            self.ldba_state = self.ldba.initial_state

        if reset_visit_counter:
            self.num_accepting_visits = 0

        # Update the observation view with the new goal text (if we have one cached)
        if self.obs is not None and isinstance(self.obs, dict):
            try:
                self.obs['goal'] = formula
            except Exception:
                pass

        # Refresh the LDBA-related fields in the observation and mark change
        if self.obs is not None:
            # Ensure we have an info dict to complete observation with propositions
            info = self.info if isinstance(self.info, dict) else {}
            self.complete_observation(self.obs, info)
            if isinstance(info, dict):
                info['ldba_state_changed'] = True
                # conservative reset for acceptance flag; next step will recompute
                info['accepting'] = False
                info['num_accepting_visits'] = self.num_accepting_visits


    @functools.cache
    def construct_ldba(self, formula: str) -> LDBA:
        propositions = get_env_attr(self.env, 'get_propositions')()
        ldba = ltl2ldba(formula, propositions, simplify_labels=False)
        possible_assignments = get_env_attr(self.env, 'get_possible_assignments')()
        ldba.prune(possible_assignments)
        ldba.complete_sink_state()
        ldba.compute_sccs()
        initial_scc = ldba.state_to_scc[ldba.initial_state]
        if initial_scc.bottom and not initial_scc.accepting:
            raise ValueError(f'The language of the LDBA for {formula} is empty.')
        return ldba
