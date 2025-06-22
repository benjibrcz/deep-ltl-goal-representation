#!/usr/bin/env python3
import os, sys, random, argparse
import numpy as np
import torch
from tqdm import trange
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# point at your src/ directory
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from visualize.zones      import draw_trajectories

# ─── defaults ─────────────────────────────────────────────────────────────────
ENV        = "PointLtl2-v0"
EXP        = "big_test"
SEED       = 0
N_WORLDS   = 50
WORLD_DIR  = f"eval_datasets/{ENV}/worlds"
# ───────────────────────────────────────────────────────────────────────────────

CMAP       = {"blue":0, "green":1, "yellow":2, "pink":3}
CMAP_RGB   = {
    0: "#1f77b4",   # blue
    1: "#2ca02c",   # green
    2: "#ff7f0e",   # yellow
    3: "#d62728",   # red
    4: "#9467bd",   # purple
    5: "#8c564b",   # brown
    6: "#e377c2",   # pink
}
COLOR_NAMES= ["blue","green","yellow","red", "purple", "brown", "pink"]
# ───────────────────────────────────────────────────────────────────────────────

def anchor_to_boundary(x, y, grid_limits=(-2.5, 2.5)):
    """Anchor predicted locations to grid boundary if they fall outside"""
    min_coord, max_coord = grid_limits
    
    # Check if point is outside the grid
    if x < min_coord or x > max_coord or y < min_coord or y > max_coord:
        # Calculate direction from origin to the point
        if x == 0 and y == 0:
            return x, y  # Avoid division by zero
        
        # Normalize the direction vector
        length = np.sqrt(x**2 + y**2)
        if length == 0:
            return x, y
        
        dir_x, dir_y = x / length, y / length
        
        # Find intersection with grid boundary
        # Try each boundary line
        candidates = []
        
        # Top boundary (y = max_coord)
        if dir_y != 0:
            t_top = (max_coord - 0) / dir_y
            if t_top > 0:
                x_top = 0 + t_top * dir_x
                if min_coord <= x_top <= max_coord:
                    candidates.append((x_top, max_coord))
        
        # Bottom boundary (y = min_coord)
        if dir_y != 0:
            t_bottom = (min_coord - 0) / dir_y
            if t_bottom > 0:
                x_bottom = 0 + t_bottom * dir_x
                if min_coord <= x_bottom <= max_coord:
                    candidates.append((x_bottom, min_coord))
        
        # Right boundary (x = max_coord)
        if dir_x != 0:
            t_right = (max_coord - 0) / dir_x
            if t_right > 0:
                y_right = 0 + t_right * dir_y
                if min_coord <= y_right <= max_coord:
                    candidates.append((max_coord, y_right))
        
        # Left boundary (x = min_coord)
        if dir_x != 0:
            t_left = (min_coord - 0) / dir_x
            if t_left > 0:
                y_left = 0 + t_left * dir_y
                if min_coord <= y_left <= max_coord:
                    candidates.append((min_coord, y_left))
        
        if candidates:
            # Choose the closest boundary point
            distances = [np.sqrt((cx - x)**2 + (cy - y)**2) for cx, cy in candidates]
            closest_idx = np.argmin(distances)
            return candidates[closest_idx]
    
    return x, y

def fig_to_array(fig):
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[...,:3]

def read_first_task(env_name):
    task_file = f"eval_datasets/{env_name}/tasks.txt"
    if not os.path.exists(task_file):
        # Handle case for LetterEnv-v0
        if env_name == "LetterEnv-v0":
            return "F b" # Default formula for LetterEnv
        else:
            print(f"Task file not found: {task_file}")
            # Try to find a default formula or exit
            return "GF blue & GF green"
    return open(task_file).readline().strip()


def collect_data_for_layer(model, layer_name, sampler):
    X, Ypos, Ycol = [], [], []
    env   = make_env(ENV, sampler, sequence=False, render_mode=None)
    props = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)
    module = dict(model.named_modules())[layer_name]

    world_dir_path = f"{WORLD_DIR}"
    if not os.path.exists(world_dir_path):
        print(f"World directory not found: {world_dir_path}, skipping data collection.")
        env.close()
        return None, None, None

    for i in trange(N_WORLDS, desc="Collect worlds"):
        world_file = f"{world_dir_path}/world_info_{i}.pkl"
        if not os.path.exists(world_file):
            continue
        env.load_world_info(world_file)
        obs = env.reset(seed=SEED+i)
        agent.reset()

        feats = []
        def grab(m, inp, out):
            x = out[1] if isinstance(out, tuple) else out
            feats.append(x.detach().cpu().numpy().ravel())
        h = module.register_forward_hook(grab)
        try: agent.get_action(obs, {}, deterministic=True)
        except: pass
        h.remove()

        X.append(feats[0])

        layout = env.task.world_info.layout
        zs, cs, keys = [], [], []
        for k,v in layout.items():
            if 'zone' in k:
                for name,idx in CMAP.items():
                    if name in k:
                        zs.append(np.asarray(v, float).tolist())
                        cs.append(idx); keys.append(k)
                        break
        order = sorted(range(len(zs)), key=lambda j: keys[j])
        if not order: # Handle case with no zones
            Ypos.append(np.array([]))
            Ycol.append([])
        else:
            Ypos.append(np.array([zs[j] for j in order]).flatten())
            Ycol.append([cs[j] for j in order])

    env.close()

    if not X:
        return None, None, None

    return np.stack(X), np.stack(Ypos), np.stack(Ycol)

def decode_true_layout(env):
    layout = env.task.world_info.layout
    zs, cs, keys = [], [], []
    for k,v in layout.items():
        if 'zone' in k:
            for name,idx in CMAP.items():
                if name in k:
                    zs.append(np.asarray(v, float).tolist())
                    cs.append(idx); keys.append(k)
                    break
    order = sorted(range(len(zs)), key=lambda j: keys[j])
    return np.array([zs[j] for j in order]), [cs[j] for j in order]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--formula',     type=str, default="GF blue & GF green")
    p.add_argument('--layer',       required=True)
    p.add_argument('--snaps',       nargs='+', type=int,
                   default=[0, 100, 200, 500, 700])
    p.add_argument('--warmup',      type=int, default=0)
    p.add_argument('--world-idx',   type=int, default=0)
    p.add_argument('--show-colors', nargs='+', choices=COLOR_NAMES,
                   default=COLOR_NAMES)
    p.add_argument('--gif',         action='store_true')
    p.add_argument('--fps',         type=int, default=5)
    p.add_argument('--gif-dt',      type=int, default=5)
    p.add_argument('--out',         type=str, default='probe_dynamics_anchored.png')
    args = p.parse_args()

    # seeds & sampler
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    formula = args.formula or read_first_task(ENV)
    sampler = FixedSampler.partial(formula)

    # ── load model ───────────────────────────────────────────────────────────────
    store = ModelStore(ENV, EXP, SEED)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[ENV]
    dummy  = make_env(ENV, sampler, sequence=False, render_mode=None)
    model  = build_model(dummy, status, cfg).eval()
    dummy.close()

    # ── train probes ─────────────────────────────────────────────────────────────
    X, Ypos, Ycol = collect_data_for_layer(model, args.layer, sampler)
    if X is None:
        print("Could not collect data for probe training. Exiting.")
        return

    ridge = Ridge().fit(X, Ypos)
    clfs  = []
    if Ycol is not None and Ycol.shape[1] > 0:
        for z in range(Ycol.shape[1]):
            y = Ycol[:,z]
            clf = None
            if len(np.unique(y))>1:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(multi_class="multinomial", max_iter=500)
                clf.fit(X, y)
            clfs.append(clf)

    # ── rollout the selected world ───────────────────────────────────────────────
    env = make_env(ENV, sampler, sequence=False, render_mode=None)
    env.load_world_info(f"{WORLD_DIR}/world_info_{args.world_idx}.pkl")
    obs = env.reset(seed=SEED+args.world_idx)
    true_zpos, true_zcol = decode_true_layout(env)

    props  = set(env.get_propositions())
    agent = Agent(model, ExhaustiveSearch(model, props, num_loops=2),
                  propositions=props, verbose=False)

    feats, traj = [], []
    module = dict(model.named_modules())[args.layer]
    def grab(m, inp, out):
        x = out[1] if isinstance(out, tuple) else out
        feats.append(x.detach().cpu().numpy().ravel())
    h = module.register_forward_hook(grab)

    agent.reset()
    traj.append(env.agent_pos[:2].copy())
    done=False
    max_steps = 1000  # Ensure a long rollout
    for step in range(max_steps):
        try:
            # Continue stepping even if agent thinks it's done
            if done:
                a = env.action_space.sample()
            else:
                a = agent.get_action(obs, {}, deterministic=True).flatten()
        except ValueError as e:
            print(f"Error getting action at step {step}: {e}. Using random action.")
            a = env.action_space.sample()

        obs, _, done, _ = env.step(a)
        traj.append(env.agent_pos[:2].copy())

        # Stop if we have enough data for snapshots and the agent is done
        if done and step > max(args.snaps, default=0):
            break
            
    h.remove()
    env.close()

    feats = np.stack(feats)
    traj  = np.stack(traj)

    # ── figure: snapshots ────────────────────────────────────────────────────────
    snaps = [s for s in args.snaps if args.warmup <= s < len(feats)]
    if not snaps:
        print("No valid snapshots to display. Rollout may have been too short or agent failed.")
        if args.gif:
            print("Skipping GIF generation as well.")
        return

    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4.5), squeeze=False)
    fig.suptitle(f"Formula: {formula} - Anchored Predictions", fontsize=16)

    selected_colors = [c for i,c in enumerate(true_zcol) if COLOR_NAMES[c] in args.show_colors]
    
    for i, s in enumerate(snaps):
        ax = axes[0,i]
        ax.set_title(f"t = {s}")
        ax.set_aspect("equal")
        ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5)
        ax.grid(True, ls="--", color="lightgray", alpha=0.5)

        # true zones (solid fill)
        for (zx,zy), c_idx in zip(true_zpos, true_zcol):
            if COLOR_NAMES[c_idx] not in args.show_colors: continue
            ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor=CMAP_RGB[c_idx], edgecolor='black', alpha=0.8, linewidth=1))

        # predictions for selected zones
        x = feats[s:s+1]
        ppos = ridge.predict(x).reshape(-1,2)

        for z_idx, ((zx,zy), c_idx) in enumerate(zip(true_zpos, true_zcol)):
            if COLOR_NAMES[c_idx] not in args.show_colors: continue
            
            pred_zx, pred_zy = ppos[z_idx]
            
            # Anchor to boundary if outside grid
            anchored_x, anchored_y = anchor_to_boundary(pred_zx, pred_zy)
            
            pred_c_idx = (clfs[z_idx].predict(x)[0]
                          if clfs[z_idx]
                          else int(np.unique(Ycol[:,z_idx])[0]))

            # Draw the predicted circle (anchored if needed)
            ax.add_patch(patches.Circle((anchored_x, anchored_y), 0.3,
                facecolor="none",
                edgecolor=CMAP_RGB[pred_c_idx],
                linestyle='--', linewidth=2))
            
            # If anchored, draw a line from agent to show direction
            if (anchored_x != pred_zx) or (anchored_y != pred_zy):
                agent_pos = traj[s]
                ax.plot([agent_pos[0], anchored_x], [agent_pos[1], anchored_y], 
                       ':', color=CMAP_RGB[pred_c_idx], linewidth=1, alpha=0.7)

        seg = traj[:s+1]
        ax.plot(seg[:,0], seg[:,1], "-o", color="k", markersize=2, linewidth=2)

    handles = [patches.Patch(facecolor=CMAP_RGB[v],
                             edgecolor='k',
                             label=k)
               for k,v in CMAP.items() if k in args.show_colors]
    
    fig.legend(handles=handles, title="zone color",
               loc="center right",
               bbox_to_anchor=(1.05, 0.5))

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(args.out.replace('.gif','.png'), dpi=150, bbox_inches="tight")
    print(f"Saved snapshot figure to {args.out.replace('.gif','.png')}")
    
    # ── make GIF if requested ────────────────────────────────────────────────────
    if args.gif:
        anim_fig, anim_ax = plt.subplots(figsize=(4,4), dpi=100)
        anim_ax.set_xlim(-2.5,2.5); anim_ax.set_ylim(-2.5,2.5)
        anim_ax.set_aspect("equal")
        anim_ax.grid(True, ls="--", color="lightgray", alpha=0.5)

        # draw true zones once
        for (zx,zy),c in zip(true_zpos,true_zcol):
            if COLOR_NAMES[c] not in args.show_colors: continue
            anim_ax.add_patch(patches.Circle((zx,zy),0.3,
                facecolor=CMAP_RGB[c], edgecolor='black', alpha=0.8, linewidth=1))

        traj_line, = anim_ax.plot([],[], '-o', color='k', markersize=2, linewidth=2)
        
        pred_patches = []
        direction_lines = []
        for z_idx, c_idx in enumerate(true_zcol):
             if COLOR_NAMES[c_idx] not in args.show_colors:
                 pred_patches.append(None)
                 direction_lines.append(None)
                 continue
             patch = patches.Circle((0,0), 0.3, facecolor='none', edgecolor=CMAP_RGB[c_idx], linestyle='--', linewidth=2)
             pred_patches.append(patch)
             anim_ax.add_patch(patch)
             
             # Add direction line for anchored predictions
             line, = anim_ax.plot([], [], ':', color=CMAP_RGB[c_idx], linewidth=1, alpha=0.7)
             direction_lines.append(line)

        def init():
            traj_line.set_data([], [])
            for patch in pred_patches:
                if patch: patch.set_center((0,0))
            for line in direction_lines:
                if line: line.set_data([], [])
            return [traj_line] + pred_patches + direction_lines

        def update(i):
            t = i * args.gif_dt
            if t >= len(feats): return [traj_line] + pred_patches + direction_lines
            
            # Update trajectory
            seg = traj[:t+1]
            traj_line.set_data(seg[:,0], seg[:,1])
            
            # Update predictions
            x = feats[t:t+1]
            ppos = ridge.predict(x).reshape(-1,2)
            
            for z_idx, (patch, line) in enumerate(zip(pred_patches, direction_lines)):
                if patch:
                    pred_c_idx = (clfs[z_idx].predict(x)[0]
                                  if clfs[z_idx]
                                  else int(np.unique(Ycol[:,z_idx])[0]))
                    patch.set_edgecolor(CMAP_RGB[pred_c_idx])
                    
                    pred_zx, pred_zy = ppos[z_idx]
                    anchored_x, anchored_y = anchor_to_boundary(pred_zx, pred_zy)
                    patch.set_center((anchored_x, anchored_y))
                    
                    # Update direction line if anchored
                    if line:
                        if (anchored_x != pred_zx) or (anchored_y != pred_zy):
                            agent_pos = traj[t]
                            line.set_data([agent_pos[0], anchored_x], [agent_pos[1], anchored_y])
                            line.set_color(CMAP_RGB[pred_c_idx])
                        else:
                            line.set_data([], [])

            return [traj_line] + pred_patches + direction_lines

        frames = (max(args.snaps) + 50) // args.gif_dt
        anim = FuncAnimation(anim_fig, update, init_func=init,
                             frames=frames, interval=1000/args.fps, blit=True)
        
        output_path = args.out if args.out.endswith('.gif') else f"{os.path.splitext(args.out)[0]}.gif"
        anim.save(output_path, writer=PillowWriter(fps=args.fps))
        print(f"Saved GIF to {output_path}")

if __name__ == '__main__':
    main() 