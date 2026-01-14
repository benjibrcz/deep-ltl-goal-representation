#!/usr/bin/env python3
"""
LetterWorld representation plots:
- Goal manifold: hidden state for the goal module, colored by (Reach/Avoid, Letter)
- Grid manifold: environment/actor features, colored by current grid letter

Assumes your rollout NPZ contains (some subset of) keys like:
  - 'hook_ltl_rnn_h', 'hook_env_mlp3', 'hook_actor_h5'  (activations; shape [T, D] or [N, T, D])
  - 'letters' or 'grid_letters' (int indices 0..25 or chars 'A'..'Z' per time-step)
  - 'reach_goal_letter', 'avoid_goal_letter' (per time-step, char or int; None/-1 if not active)
  - 'episode_id' (optional), 't' (optional)

If your fields differ, tweak the `label_extractors` near the bottom (very small changes).
"""

import argparse
import os
import re
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------

def ensure_2d(x):
    """Accept [T, D], [N, T, D], or list-of-arrays -> return [M, D]."""
    if isinstance(x, list):
        x = np.concatenate(x, axis=0)
    x = np.asarray(x)
    if x.ndim == 3:
        N, T, D = x.shape
        x = x.reshape(N * T, D)
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D or 3D array, got shape {x.shape}")
    return x

def to_letters(array_like):
    """Convert ints or bytes/strings to uppercase letter labels."""
    arr = np.array(array_like)
    # If already strings/chars:
    if arr.dtype.kind in ("U", "S", "O"):
        return np.array([str(a).upper() if a is not None else None for a in arr], dtype=object)
    # If ints (0-based or 1-based)
    out = []
    for a in arr:
        if a is None:
            out.append(None)
        elif a < 0:
            out.append(None)
        else:
            # Try map 0->'A'
            out.append(chr(ord('A') + int(a)))
    return np.array(out, dtype=object)

def take_class_balanced_subset(X, labels, max_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(labels))
    grouped = defaultdict(list)
    for i, y in enumerate(labels):
        grouped[y].append(i)
    kept = []
    for y, group in grouped.items():
        group = np.array(group)
        if len(group) > max_per_class:
            group = rng.choice(group, size=max_per_class, replace=False)
        kept.append(group)
    kept = np.concatenate(kept) if kept else np.array([], dtype=int)
    return X[kept], labels[kept], kept

def pca_project(X, n=2, whiten=False):
    Xs = StandardScaler(with_mean=True, with_std=whiten).fit_transform(X)
    pcs = PCA(n_components=n, random_state=0).fit_transform(Xs)
    return pcs

def plot_scatter(
    XY, labels, title, out_png, class_order=None, annotate_centroids=True, alpha=0.6, s=10
):
    plt.figure(figsize=(7, 6), dpi=150)
    # Deterministic color order
    uniq = list(dict.fromkeys(labels))  # preserve first-seen order
    if class_order:
        # keep only ones in data, in given order; append any extras
        uniq = [c for c in class_order if c in set(labels)] + [c for c in uniq if c not in class_order]
    # Simple matplotlib cycle is fine; colors will repeat if many classes
    xs, ys = XY[:, 0], XY[:, 1]
    for cls in uniq:
        mask = (labels == cls)
        plt.scatter(xs[mask], ys[mask], s=s, alpha=alpha, label=str(cls))
    # Centroids
    if annotate_centroids:
        for cls in uniq:
            mask = (labels == cls)
            if mask.sum() == 0: continue
            cx, cy = xs[mask].mean(), ys[mask].mean()
            plt.scatter([cx], [cy], s=90, marker="X", edgecolors="k", linewidths=0.7, zorder=5)
            plt.text(cx, cy, f" {cls}", fontsize=8, weight='bold')
    plt.title(title)
    plt.legend(markerscale=1.6, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def linear_separability_score(XY, labels):
    """Tiny sanity check: one-vs-rest macro F1-ish accuracy using LR CV split."""
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(0)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]
    y = labels
    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    clf.fit(XY[tr], y[tr])
    acc = accuracy_score(y[te], clf.predict(XY[te]))
    return acc

# -----------------------------
# Label extractors (customize here if your NPZ fields differ)
# -----------------------------

def build_goal_labels(d):
    """
    Compose goal labels per time-step as:
      - 'Reach:X' if reach goal active for letter X
      - 'Avoid:Y' if avoid goal active for letter Y
    If both present, prefer Reach first then Avoid; or create combined tag 'Reach:X|Avoid:Y'
    """
    # Try a few common field names
    reach_raw = None
    avoid_raw = None

    for k in ['reach_goal_letter', 'goal_reach', 'goal_reach_letter', 'reach_letter']:
        if k in d.files:
            reach_raw = d[k]
            break
    for k in ['avoid_goal_letter', 'goal_avoid', 'goal_avoid_letter', 'avoid_letter']:
        if k in d.files:
            avoid_raw = d[k]
            break

    if reach_raw is None and avoid_raw is None:
        # Try to parse from a per-step 'task' string like "Reach:A, Avoid:B"
        if 'task_str' in d.files:
            task = np.array(d['task_str'])
            # naive parse to per-step labels
            reach = []
            avoid = []
            for s in task:
                s = str(s)
                m1 = re.search(r'(Reach|F)\s*:?\s*([A-Za-z])', s)
                m2 = re.search(r'(Avoid|G!|G¬)\s*:?\s*([A-Za-z])', s)
                reach.append(m1.group(2) if m1 else None)
                avoid.append(m2.group(2) if m2 else None)
            reach_raw, avoid_raw = np.array(reach), np.array(avoid)
        else:
            raise KeyError("Could not find reach/avoid goal labels. Please add them or tweak build_goal_labels().")

    reach = to_letters(ensure_2d(reach_raw)[:, 0]) if np.asarray(reach_raw).ndim >= 2 else to_letters(reach_raw)
    avoid = to_letters(ensure_2d(avoid_raw)[:, 0]) if (avoid_raw is not None and np.asarray(avoid_raw).ndim >= 2) else (to_letters(avoid_raw) if avoid_raw is not None else None)

    labels = []
    for i in range(len(reach) if reach is not None else len(avoid)):
        r = reach[i] if reach is not None and i < len(reach) else None
        a = avoid[i] if avoid is not None and i < len(avoid) else None
        tag = None
        if r and a:
            tag = f"Reach:{r}|Avoid:{a}"
        elif r:
            tag = f"Reach:{r}"
        elif a:
            tag = f"Avoid:{a}"
        else:
            tag = "NoGoal"
        labels.append(tag)
    return np.array(labels, dtype=object)

def build_grid_labels(d):
    """
    Label by the current grid letter at the agent's position (or focal tile).
    """
    letter_key = None
    for k in ['grid_letters', 'letters', 'current_letter', 'tile_letter', 'letter_id']:
        if k in d.files:
            letter_key = k
            break
    if letter_key is None:
        raise KeyError("Could not find grid letter labels. Add e.g. 'grid_letters' or tweak build_grid_labels().")
    raw = d[letter_key]
    raw_arr = np.asarray(raw)
    # Accept [T] directly; else coerce via ensure_2d
    if raw_arr.ndim == 1:
        letters = to_letters(raw_arr)
    else:
        arr = ensure_2d(raw_arr)
        # Take first column if 2D with multiple columns
        if arr.ndim == 2 and arr.shape[1] > 1:
            arr = arr[:, 0]
        letters = to_letters(arr)
    return letters

# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=str, required=True, help="Path to rollout NPZ with activations + labels")
    p.add_argument("--goal_hook", type=str, default="hook_ltl_rnn_h", help="NPZ key for goal-representation activations")
    p.add_argument("--grid_hook", type=str, default="hook_env_mlp3", help="NPZ key for env/obs activations (env manifold)")
    p.add_argument("--actor_hook", type=str, default=None, help="Optional NPZ key for actor activations (actor manifold)")
    p.add_argument("--max_per_class", type=int, default=400, help="Subsample cap per class for clean plots")
    p.add_argument("--whiten", action="store_true", help="Use StandardScaler std=1 before PCA")
    p.add_argument("--out_dir", type=str, default="rep_plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.npz, allow_pickle=True)

    # ---- Goal manifold ----
    if args.goal_hook not in d.files:
        print(f"[warn] '{args.goal_hook}' not in NPZ; skipping goal manifold.")
    else:
        goal_X = ensure_2d(d[args.goal_hook])
        goal_labels = build_goal_labels(d)

        # Filter out 'NoGoal' if you want cleaner plots
        keep = goal_labels != "NoGoal"
        if keep.sum() >= 20:
            goal_X = goal_X[keep]
            goal_labels = goal_labels[keep]

        goal_X, goal_labels, kept_idx = take_class_balanced_subset(goal_X, goal_labels, max_per_class=args.max_per_class, seed=0)
        goal_xy = pca_project(goal_X, n=2, whiten=args.whiten)

        # Optional: compute a quick linear separability score
        try:
            acc = linear_separability_score(goal_xy, goal_labels)
            print(f"[goal] LR separability (holdout acc): {acc:.3f}")
        except Exception as e:
            print(f"[goal] separability skip: {e}")

        # Save CSV for later iterations
        df_goal = pd.DataFrame({
            "x": goal_xy[:, 0], "y": goal_xy[:, 1],
            "label": goal_labels
        })
        df_goal.to_csv(os.path.join(args.out_dir, "goal_manifold_pca.csv"), index=False)

        # Plot
        plot_scatter(
            goal_xy, goal_labels,
            title="Goal manifold (PCA on goal module)",
            out_png=os.path.join(args.out_dir, "goal_manifold_pca.png"),
            class_order=None,
            annotate_centroids=True,
            alpha=0.6, s=8
        )
        print(f"[goal] Saved: {os.path.join(args.out_dir, 'goal_manifold_pca.png')}")

    # ---- Grid/Actor manifolds (separate) ----
    def plot_grid_for_hook(hook_key: str, tag: str):
        if hook_key not in d.files:
            print(f"[warn] '{hook_key}' not in NPZ; skipping {tag} manifold.")
            return
        X = ensure_2d(d[hook_key])
        labels = build_grid_labels(d)
        if len(X) != len(labels):
            m = min(len(X), len(labels))
            print(f"[warn] length mismatch {tag}_X({len(X)}) vs labels({len(labels)}); truncating to {m}")
            X = X[:m]
            labels = labels[:m]
        # Drop None labels
        nn_mask = np.array([lbl is not None for lbl in labels])
        X = X[nn_mask]
        labels = labels[nn_mask]
        # Keep only sufficiently frequent classes
        counts = Counter(labels)
        valid = {k for k, v in counts.items() if v >= 20}
        mask = np.array([lbl in valid for lbl in labels])
        X = X[mask]
        labels = labels[mask]
        X, labels, _ = take_class_balanced_subset(X, labels, max_per_class=args.max_per_class, seed=1)
        XY = pca_project(X, n=2, whiten=args.whiten)
        try:
            acc = linear_separability_score(XY, labels)
            print(f"[{tag}] LR separability (holdout acc): {acc:.3f}")
        except Exception as e:
            print(f"[{tag}] separability skip: {e}")
        df = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1], "letter": labels})
        csv_path = os.path.join(args.out_dir, f"{tag}_manifold_pca.csv")
        png_path = os.path.join(args.out_dir, f"{tag}_manifold_pca.png")
        df.to_csv(csv_path, index=False)
        plot_scatter(
            XY, labels,
            title=f"{tag.capitalize()} letter manifold (PCA)",
            out_png=png_path,
            class_order=[chr(ord('A') + i) for i in range(12)],
            annotate_centroids=True,
            alpha=0.6, s=8
        )
        print(f"[{tag}] Saved: {png_path}")

    # Env manifold
    plot_grid_for_hook(args.grid_hook, tag="env")
    # Actor manifold (optional)
    if args.actor_hook is not None:
        plot_grid_for_hook(args.actor_hook, tag="actor")

    print("[done]")

if __name__ == "__main__":
    main()
