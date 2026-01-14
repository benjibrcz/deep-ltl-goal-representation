#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import numpy as np
import imageio.v2 as imageio
from typing import Dict, Tuple, List

try:
    from envs.letter_world.letter_env import LetterEnv
except Exception:
    from src.envs.letter_world.letter_env import LetterEnv


def torus_l1(i: int, j: int, H: int, W: int) -> int:
    return min(i, H - i) + min(j, W - j)


def place_three_surround(H: int, center: Tuple[int, int], avoid: str) -> Dict[Tuple[int, int], str]:
    ci, cj = center
    nbrs = [((ci - 1) % H, cj), ((ci + 1) % H, cj), (ci, (cj - 1) % H), (ci, (cj + 1) % H)]
    # choose any 3 of the 4 neighbors
    return {pos: avoid for pos in nbrs[:-0]}  # will trim later


def build_map_two_letters(grid_size: int,
                          reach: str,
                          avoid: str,
                          rng: random.Random,
                          center_candidates: List[Tuple[int, int]],
                          far_threshold: int,
                          surround_k: int = 3) -> Dict[Tuple[int, int], str]:
    H = grid_size
    W = grid_size
    ci, cj = H // 2, W // 2
    def make_once() -> Dict[Tuple[int, int], str]:
        axis = rng.choice(['h', 'v'])
        max_d = max(1, min(ci, cj))
        # enforce at least one-cell gap from center to reach; if single-block avoid, ensure avoid not adjacent to center
        min_d = 2
        if surround_k == 1:
            min_d = 3
        min_d = min(min_d, max_d)
        d_choices = list(range(min_d, max_d + 1)) if max_d >= min_d else [max_d]
        d = rng.choice(d_choices)
        if axis == 'h':
            p1 = (ci, (cj + d) % W)
            p2 = (ci, (cj - d) % W)
        else:
            p1 = ((ci + d) % H, cj)
            p2 = ((ci - d) % H, cj)
        m: Dict[Tuple[int, int], str] = {}
        m[p1] = reach
        m[p2] = reach
        k = max(0, min(int(surround_k), 4))
        if k == 1:
            if axis == 'h':
                step = 1 if p1[1] == (cj + d) % W else -1
                # place avoid two cells away from center to keep a gap between agent and avoid
                mid = (ci, (cj + 2 * step) % W)
            else:
                step = 1 if p1[0] == (ci + d) % H else -1
                mid = ((ci + 2 * step) % H, cj)
            if mid != p2 and mid != (ci, cj) and mid != p1:
                m[mid] = avoid
        else:
            nbrs = [((p1[0] - 1) % H, p1[1]), ((p1[0] + 1) % H, p1[1]), (p1[0], (p1[1] - 1) % W), (p1[0], (p1[1] + 1) % W)]
            rng.shuffle(nbrs)
            placed = 0
            for pos in nbrs:
                if placed >= k:
                    break
                if pos == p2 or pos == (ci, cj):
                    continue
                # avoid placing avoid next to center
                if pos in {((ci - 1) % H, cj), ((ci + 1) % H, cj), (ci, (cj - 1) % W), (ci, (cj + 1) % W)}:
                    continue
                m[pos] = avoid
                placed += 1
        return m
    def valid(m: Dict[Tuple[int,int],str]) -> bool:
        keys = list(m.keys())
        # exactly two reach and at least one avoid when surround_k>0
        reaches = [pos for pos in keys if m[pos] == reach]
        avoids = [pos for pos in keys if m[pos] == avoid]
        if len(reaches) != 2:
            return False
        if surround_k >= 1 and len(avoids) < 1:
            return False
        # mirrored check
        (r1i,r1j),(r2i,r2j) = reaches
        # same row with center or same column with center
        if not ((r1i == ci == r2i) or (r1j == cj == r2j)):
            return False
        # equal torus distance from center
        if torus_l1(r1i,r1j,H,W) != torus_l1(ci,cj,H,W) or torus_l1(r2i,r2j,H,W) != torus_l1(ci,cj,H,W):
            pass  # distance from (0,0) is irrelevant; skip
        # ensure center not occupied
        if (ci,cj) in m:
            return False
        return True
    for _ in range(50):
        m = make_once()
        if valid(m):
            return m
    # fallback
    return make_once()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10, help='Number of maps to generate')
    ap.add_argument('--grid_size', type=int, default=7)
    ap.add_argument('--out_dir', type=str, default='interpretability/letter_world/results/two_letter_sets')
    ap.add_argument('--letters', type=str, default='abcdefghijklmnopqrstuvwxyz', help='Pool to sample letters from')
    ap.add_argument('--seed', type=int, default=0)
    # layout control
    ap.add_argument('--center_candidates', type=str, default='(1,1),(1,5),(5,1),(5,5),(2,4),(4,2)',
                    help='Comma list of (i,j) for the surrounded reach; default picks corners-ish')
    ap.add_argument('--far_threshold', type=int, default=4, help='Min torus L1 distance from agent for the second reach')
    ap.add_argument('--surround_k', type=int, default=3, help='Number of avoid neighbors around the surrounded reach (0..4)')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # parse centers
    centers: List[Tuple[int, int]] = []
    for tok in args.center_candidates.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ii, jj = tok.strip('()').split(',')
            centers.append((int(ii), int(jj)))
        except Exception:
            continue
    if not centers:
        centers = [(1, 1)]

    letters = [c for c in args.letters if c.isalpha()]
    for k in range(1, args.N + 1):
        # pick reach/avoid distinct
        reach = rng.choice(letters)
        avoid = rng.choice([c for c in letters if c != reach])
        m = build_map_two_letters(args.grid_size, reach, avoid, rng, centers, args.far_threshold, surround_k=args.surround_k)
        # letters present exactly those used
        letters_str = ''.join(sorted(set(m.values())))
        env = LetterEnv(grid_size=args.grid_size, letters=letters_str, use_fixed_map=True,
                        use_agent_centric_view=False, render_mode='rgb_array', map=m)
        env.agent = (0, 0)
        rgb = env.render()
        png = out / f"map_{k:02d}_{reach}_{avoid}.png"
        pkl = out / f"map_{k:02d}_{reach}_{avoid}.pkl"
        imageio.imwrite(png, rgb)
        env.save_world_info(str(pkl))
    print(f"[two_letter_dataset] wrote {args.N} maps to {out}")


if __name__ == '__main__':
    main()


