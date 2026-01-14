#!/usr/bin/env python3
import numpy as np
from collections import deque
from typing import Tuple, Set, Optional, List, Dict


def detect_agent_channel(obs_hw_c: np.ndarray) -> int:
    """Heuristic: agent channel is sparsest across grid."""
    H, W, C = obs_hw_c.shape
    ch_sums = obs_hw_c.reshape(H * W, C).sum(axis=0)
    return int(np.argmin(ch_sums))


def letter_cells(obs_hw_c: np.ndarray, letter_ch: int) -> Set[Tuple[int, int]]:
    """Return set of (i,j) where the given letter channel is active."""
    mask = obs_hw_c[..., letter_ch] > 0.5
    return set(zip(*np.where(mask)))


def bfs_shortest(
    H: int,
    W: int,
    start: Tuple[int, int],
    targets: Set[Tuple[int, int]],
    blocked: Set[Tuple[int, int]],
    Hmax: int,
    wrap: bool = True,
) -> Tuple[Optional[int], Optional[int]]:
    """
    4-neighbor BFS on a toroidal grid (wrap by default).
    Returns (dist, first_action), where actions are 0:Right,1:Left,2:Down,3:Up.
    If no path within Hmax, returns (None, None).
    """
    if start in blocked:
        return None, None
    if start in targets:
        return 0, None
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    # Track first action taken from start to reach each visited cell
    visited: Dict[Tuple[int, int], Tuple[int, int]] = {}
    q = deque()
    q.append((start[0], start[1], 0, None))  # (i,j,dist,first_action)
    visited[start] = (0, None)
    while q:
        i, j, d, fa = q.popleft()
        if d >= Hmax:
            continue
        for a, (di, dj) in enumerate(neighbors):
            ni = (i + di) % H if wrap else i + di
            nj = (j + dj) % W if wrap else j + dj
            if not wrap and (ni < 0 or ni >= H or nj < 0 or nj >= W):
                continue
            if (ni, nj) in blocked:
                continue
            if (ni, nj) in visited:
                continue
            next_fa = a if fa is None else fa
            nd = d + 1
            visited[(ni, nj)] = (nd, next_fa)
            if (ni, nj) in targets:
                return nd, next_fa
            q.append((ni, nj, nd, next_fa))
    return None, None


def extract_targets_and_blocked(
    obs_hw_c: np.ndarray,
    target_ch: int,
    avoid_ch: Optional[int],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    tset = letter_cells(obs_hw_c, target_ch)
    bset: Set[Tuple[int, int]] = set()
    if avoid_ch is not None and avoid_ch >= 0:
        bset = letter_cells(obs_hw_c, avoid_ch)
    return tset, bset


def first_letter_channel_at_pos(obs_hw_c: np.ndarray, agent_pos: Tuple[int, int], agent_ch: Optional[int] = None) -> Optional[int]:
    """Return the channel index (excluding agent channel) that is active at the agent's position, or None."""
    i, j = int(agent_pos[0]), int(agent_pos[1])
    C = obs_hw_c.shape[-1]
    a_ch = agent_ch if agent_ch is not None else detect_agent_channel(obs_hw_c)
    for ch in range(C):
        if ch == a_ch:
            continue
        if obs_hw_c[i, j, ch] > 0.5:
            return ch
    return None


