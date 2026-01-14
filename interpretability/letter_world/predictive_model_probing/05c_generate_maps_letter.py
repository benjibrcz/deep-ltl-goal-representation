#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import random
import imageio.v2 as imageio
from typing import Dict, Tuple, List

try:
    from envs.letter_world.letter_env import LetterEnv
except Exception:
    from src.envs.letter_world.letter_env import LetterEnv


def place_surrounded_target(grid_size: int, target: str, avoid: str, surround_k: int = 4, center: Tuple[int,int] = None) -> Dict[Tuple[int,int], str]:
    """Place one target letter and surround it by up to 4 avoid letters (von Neumann neighborhood)."""
    H = grid_size
    if center is None:
        center = (H // 2, H // 2)
    m: Dict[Tuple[int,int], str] = {}
    ci, cj = center
    m[(ci, cj)] = target
    nbrs = [((ci, (cj+1)%H)),
            ((ci, (cj-1+H)%H)),
            (((ci+1)%H, cj)),
            (((ci-1+H)%H, cj))]
    for (ni, nj) in nbrs[:max(0, min(surround_k, 4))]:
        m[(ni, nj)] = avoid
    return m


def scatter_letters(m: Dict[Tuple[int,int], str], grid_size: int, letter: str, count: int, rng: random.Random):
    """Scatter additional occurrences of a letter into empty cells."""
    H = grid_size
    empties = [(i, j) for i in range(H) for j in range(H) if (i, j) not in m and (i, j) != (0, 0)]
    rng.shuffle(empties)
    for pos in empties[:max(0, count)]:
        m[pos] = letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid_size', type=int, default=7)
    ap.add_argument('--target', type=str, default='a')
    ap.add_argument('--avoid', type=str, default='c')
    ap.add_argument('--other', type=str, default='b', help='Second reach letter for safety scenarios')
    ap.add_argument('--surround_k', type=int, default=4, help='How many avoid letters to place around target (0-4)')
    ap.add_argument('--center_i', type=int, default=None, help='Optional row for target center')
    ap.add_argument('--center_j', type=int, default=None, help='Optional col for target center')
    ap.add_argument('--extra_target', type=int, default=2, help='Additional scattered target letters')
    ap.add_argument('--target2_i', type=int, default=None, help='Optional explicit placement row for second TARGET')
    ap.add_argument('--target2_j', type=int, default=None, help='Optional explicit placement col for second TARGET')
    ap.add_argument('--extra_avoid', type=int, default=4, help='Additional scattered avoid letters')
    ap.add_argument('--extra_other', type=int, default=2, help='Additional scattered other letters')
    ap.add_argument('--other_i', type=int, default=None, help='Optional explicit placement row for one OTHER letter')
    ap.add_argument('--other_j', type=int, default=None, help='Optional explicit placement col for one OTHER letter')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_png', type=str, required=True)
    ap.add_argument('--out_map', type=str, default=None, help='Optional path to save pickled map via env.save_world_info')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Build base map
    center = (args.center_i, args.center_j) if (args.center_i is not None and args.center_j is not None) else None
    m = place_surrounded_target(args.grid_size, args.target, args.avoid, surround_k=args.surround_k, center=center)
    # Optionally place ONE explicit second TARGET
    placed_explicit_target2 = False
    if args.target2_i is not None and args.target2_j is not None:
        ti, tj = int(args.target2_i), int(args.target2_j)
        if (ti, tj) != (0, 0) and (ti, tj) not in m:
            m[(ti, tj)] = args.target
            placed_explicit_target2 = True
    # Scatter more targets (minus explicit one if provided)
    scatter_letters(m, args.grid_size, args.target,
                    max(0, args.extra_target - (1 if placed_explicit_target2 else 0)), rng)
    scatter_letters(m, args.grid_size, args.avoid,  args.extra_avoid,  rng)
    # Optionally place ONE explicit other letter
    if args.other_i is not None and args.other_j is not None:
        oi, oj = int(args.other_i), int(args.other_j)
        if (oi, oj) != (0, 0):
            m[(oi, oj)] = args.other
    # Then scatter remaining OTHER letters (avoid overwriting)
    scatter_letters(m, args.grid_size, args.other,  max(0, args.extra_other - (1 if args.other_i is not None and args.other_j is not None else 0)),  rng)
    # Build letters string for renderer from actual map contents (unique letters present)
    letters_str = ''.join(sorted(set(m.values())))
    env = LetterEnv(grid_size=args.grid_size, letters=letters_str, use_fixed_map=True,
                    use_agent_centric_view=False, render_mode='rgb_array', map=m)
    env.agent = (0, 0)
    rgb = env.render()
    outp = Path(args.out_png); outp.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(outp, rgb)
    if args.out_map:
        Path(args.out_map).parent.mkdir(parents=True, exist_ok=True)
        env.save_world_info(args.out_map)
    print(f"[make_maps] wrote {outp}  (map has {len(m)} letters)")


if __name__ == '__main__':
    main()


