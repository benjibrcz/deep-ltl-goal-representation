#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

try:
    from interpretability.letter_world.predictive_model_probing.bfs_oracle import letter_cells
except Exception:
    try:
        from predictive_model_probing.bfs_oracle import letter_cells
    except Exception:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.append(str(_Path(__file__).resolve().parent))
        from bfs_oracle import letter_cells

# Import LetterEnv for renderer usage
try:
    from envs.letter_world.letter_env import LetterEnv
except Exception:
    try:
        from src.envs.letter_world.letter_env import LetterEnv
    except Exception:
        LetterEnv = None

def draw_overlay(ax, H, W, cells: set, color: Tuple[float,float,float], alpha: float, label: Optional[str] = None):
    if not cells:
        return
    M = np.zeros((H, W, 4), dtype=float)
    for (i, j) in cells:
        M[i, j, :3] = np.array(color)
        M[i, j, 3]  = alpha
    ax.imshow(M, origin='upper', interpolation='none')
    if label:
        # add a legend proxy by plotting a single invisible point
        ax.plot([], [], color=color, alpha=alpha, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='Sequential NPZ with obs_raw, agent_pos')
    ap.add_argument('--candidates', type=str, required=True, help='NPZ from 05_mine_scenarios_letter.py')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--A_ch', type=int, default=0)
    ap.add_argument('--B_ch', type=int, default=1)
    ap.add_argument('--C_ch', type=int, default=2)
    ap.add_argument('--X_ch', type=int, default=0)
    ap.add_argument('--Y_ch', type=int, default=1)
    ap.add_argument('--max_plots', type=int, default=50)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--use_env_render', action='store_true', help='Render using LetterEnv renderer (rgb_array) instead of matplotlib overlays')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    C = np.load(args.candidates, allow_pickle=True)
    for k in ['obs_raw', 'agent_pos']:
        if k not in D.files:
            raise SystemExit("Sequential dataset missing 'obs_raw' or 'agent_pos'. Re-run 03c with --save_obs.")
    OBS = np.asarray(D['obs_raw'])
    POS = np.asarray(D['agent_pos'])

    idxs = list(np.asarray(C['indices']).tolist())
    kinds = np.asarray(C['kind']).tolist()
    oracle = np.asarray(C['oracle']).tolist()
    meta = np.asarray(C['meta'], dtype=object).tolist()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def channel_to_letters(obs_hw_c: np.ndarray):
        flat_ch = obs_hw_c.reshape(-1, obs_hw_c.shape[-1]).sum(axis=0)
        agent_ch = int(np.argmin(flat_ch))
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        ch_to_char = {}
        next_idx = 0
        for ch in range(obs_hw_c.shape[-1]):
            if ch == agent_ch:
                continue
            ch_to_char[ch] = alphabet[next_idx]
            next_idx += 1
        return ch_to_char, agent_ch

    def scenario_formula(kind: str, args, ch_to_char: dict) -> str:
        def name(ch):
            return ch_to_char.get(int(ch), f"ch{int(ch)}")
        if kind == 'safety_detour':
            return f"(F {name(args.A_ch)} OR F {name(args.B_ch)}) & G ! {name(args.C_ch)}"
        if kind == 'lookahead':
            return f"F {name(args.X_ch)} THEN F {name(args.Y_ch)}"
        return kind

    n_plots = 0
    for t, (i, k, oc, md) in enumerate(zip(idxs, kinds, oracle, meta)):
        if t % max(1, int(args.stride)) != 0:
            continue
        if n_plots >= int(args.max_plots):
            break
        obs = OBS[i]
        pos = tuple(int(v) for v in POS[i])
        H, W, _ = obs.shape

        ch_to_char, agent_ch = channel_to_letters(obs)
        formula_str = scenario_formula(k, args, ch_to_char)
        safe_tag = formula_str.replace(" ", "").replace("(", "").replace(")", "").replace("|", "OR").replace("&", "AND").replace("!", "NOT")
        out_path = out_dir / f"scenario_{t:04d}_idx{i}_{k}_{safe_tag}.png"
        if args.use_env_render and LetterEnv is not None:
            # Reconstruct a LetterEnv from obs: map letters per-channel ignoring agent channel.
            # Map channels -> letters 'a','b','c',... in index order.
            # Detect agent channel as the sparsest across grid (one-hot agent).
            # Build mapping for letter channels to characters
            letters = list(ch_to_char.values())
            letters_str = "".join(letters) if letters else "a"
            # Build map dict
            m = {}
            for ii in range(H):
                for jj in range(W):
                    # choose the letter channel that is active
                    best_ch = None
                    best_val = 0.0
                    for ch in range(obs.shape[-1]):
                        if ch == agent_ch:
                            continue
                        v = float(obs[ii, jj, ch])
                        if v > best_val:
                            best_val = v; best_ch = ch
                    if best_ch is not None and best_val > 0.5:
                        m[(ii, jj)] = ch_to_char[best_ch]
            # Instantiate env with fixed map and agent
            env = LetterEnv(grid_size=H, letters=letters_str, use_fixed_map=True,
                            use_agent_centric_view=False, render_mode='rgb_array', map=m)
            env.agent = (int(pos[0]), int(pos[1]))
            rgb = env.render()
            import imageio.v2 as imageio
            imageio.imwrite(out_path, rgb)
        else:
            # Matplotlib overlay fallback (no dependency on env)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.zeros((H, W)), cmap='gray', vmin=0, vmax=1, interpolation='none', origin='upper')
            ax.set_xticks(range(W)); ax.set_yticks(range(H)); ax.grid(True, color='lightgray', linewidth=0.5)
            ax.set_xlim(-0.5, W-0.5); ax.set_ylim(H-0.5, -0.5)
            Aset = letter_cells(obs, args.A_ch)
            Bset = letter_cells(obs, args.B_ch)
            Cset = letter_cells(obs, args.C_ch)
            Xset = letter_cells(obs, args.X_ch)
            Yset = letter_cells(obs, args.Y_ch)
            draw_overlay(ax, H, W, Cset, (1.0, 0.2, 0.2), 0.35, label='Avoid C')
            if k == 'safety_detour':
                draw_overlay(ax, H, W, Aset, (0.2, 0.8, 0.2), 0.45, label='A')
                draw_overlay(ax, H, W, Bset, (0.2, 0.4, 0.9), 0.45, label='B')
            elif k == 'lookahead':
                draw_overlay(ax, H, W, Xset, (0.95, 0.85, 0.2), 0.50, label='X')
                draw_overlay(ax, H, W, Yset, (0.2, 0.95, 0.95), 0.35, label='Y')
            ax.scatter([pos[1]], [pos[0]], c='k', s=60, marker='o', edgecolors='white', linewidths=0.8, zorder=5, label='Agent')
            title = f"{k} | idx={i} | oracle={oc}"
            if isinstance(md, dict):
                if k == 'safety_detour':
                    da = md.get('dA_safe', None); db = md.get('dB_safe', None)
                    if da is not None and db is not None:
                        title += f" | dA_safe={da}, dB_safe={db}"
                elif k == 'lookahead':
                    jb = md.get('J_best', None); js = md.get('J_second', None)
                    if jb is not None and js is not None:
                        title += f" | J*={jb}, J2={js}"
            ax.set_title(title, fontsize=9)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
        n_plots += 1

    print(f"[viz_mined] wrote {n_plots} figures to {out_dir}")


if __name__ == '__main__':
    main()


