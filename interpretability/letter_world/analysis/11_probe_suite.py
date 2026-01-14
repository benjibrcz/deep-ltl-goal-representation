#!/usr/bin/env python3
import argparse
from pathlib import Path
import csv

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier


TARGETS = {
    # regression
    'next_pos': dict(type='reg'),
    'delta': dict(type='reg'),  # signed (-1,0,+1)
    # classification
    'on_letter': dict(type='clf'),
    'action': dict(type='clf'),
    'delta_cls': dict(type='clf'),  # 4-class delta as classification
}

FEATURES = ['actor_mid', 'actor_in', 'obs_emb', 'e_sigma', 'obs_raw_flat']


def load_data(path):
    d = np.load(path, allow_pickle=True)
    return d


def is_arr(x):
    return isinstance(x, np.ndarray)


def build_on_letter_labels(dataset):
    feats = dataset['obs_raw']
    y = []
    for f in feats:
        if not is_arr(f):
            y.append(None)
            continue
        H, W, C = f.shape
        flat = f.reshape(-1, C)
        ch_sums = flat.sum(axis=0)
        if (ch_sums == 1).any():
            agent_ch = int(np.where(ch_sums == 1)[0][0])
        else:
            agent_ch = int(np.argmin(ch_sums))
        agent_mask = f[..., agent_ch] > 0.5
        # sum all other channels as letters
        others = np.delete(f, agent_ch, axis=-1)
        letter_here = (others.sum(axis=-1) > 0)
        y.append(int((agent_mask & letter_here).any()))
    return np.array(y, dtype=object)


def _signed_step(step, grid):
    if step == 0:
        return 0
    return 1 if step == 1 else -1


def build_delta_and_action(dataset):
    pos, nxt = dataset['pos'], dataset['next_pos']
    y_delta, y_action = [], []
    f0 = next((f for f in dataset['obs_raw'] if is_arr(f)), None)
    grid = f0.shape[0] if f0 is not None else 7

    logged_action = dataset['action'] if 'action' in dataset.files else None

    for k, (p, n) in enumerate(zip(pos, nxt)):
        if not (is_arr(p) and is_arr(n)):
            y_delta.append(None)
            y_action.append(None)
            continue
        dxw = (int(n[0]) - int(p[0])) % grid
        dyw = (int(n[1]) - int(p[1])) % grid
        dx, dy = _signed_step(dxw, grid), _signed_step(dyw, grid)
        y_delta.append([dx, dy])

        if logged_action is not None and (isinstance(logged_action[k], (int, np.integer)) or is_arr(logged_action[k])):
            try:
                y_action.append(int(logged_action[k]))
                continue
            except Exception:
                pass
        # Map signed deltas to 4 actions; adjust if your convention differs
        if   dx ==  0 and dy ==  1:
            act = 0  # RIGHT
        elif dx ==  0 and dy == -1:
            act = 1  # LEFT
        elif dx ==  1 and dy ==  0:
            act = 2  # DOWN
        elif dx == -1 and dy ==  0:
            act = 3  # UP
        else:
            act = None
        y_action.append(act)

    return np.array(y_delta, dtype=object), np.array(y_action, dtype=object)


def to_delta_cls(dd):
    if not is_arr(dd):
        return None
    dx, dy = int(dd[0]), int(dd[1])
    if dx == 0 and dy == 1:
        return 0
    if dx == 0 and dy == -1:
        return 1
    if dx == 1 and dy == 0:
        return 2
    if dx == -1 and dy == 0:
        return 3
    return None


def run_regression(X, Y, split=0.8, groups=None):
    n = len(X)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=0)
        tr, te = next(gss.split(np.arange(n), groups=groups))
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        s = int(split * n)
        tr, te = idx[:s], idx[s:]
    pipe = make_pipeline(StandardScaler(with_mean=True), LinearRegression())
    pipe.fit(X[tr], Y[tr])
    pred = pipe.predict(X[te])
    return r2_score(Y[te], pred, multioutput='variance_weighted'), mean_absolute_error(Y[te], pred)


def run_classification(X, y, split=0.8, groups=None):
    n = len(X)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=0)
        tr, te = next(gss.split(np.arange(n), groups=groups))
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        s = int(split * n)
        tr, te = idx[:s], idx[s:]
    pipe = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    return accuracy_score(y[te], pred)


def run_classification_mlp(X, y, split=0.8, groups=None):
    n = len(X)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=0)
        tr, te = next(gss.split(np.arange(n), groups=groups))
    else:
        idx = np.arange(n)
        np.random.shuffle(idx)
        s = int(split * n)
        tr, te = idx[:s], idx[s:]
    pipe = make_pipeline(StandardScaler(with_mean=True), MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True))
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    return accuracy_score(y[te], pred)


def majority_baseline(y):
    vals, cnts = np.unique(y, return_counts=True)
    return cnts.max() / cnts.sum()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--features', type=str, default=','.join(FEATURES))
    p.add_argument('--targets', type=str, default=','.join(TARGETS.keys()))
    p.add_argument('--split', type=float, default=0.8)
    p.add_argument('--use_mlp', action='store_true', help='Use a tiny MLP for classification probes')
    p.add_argument('--confusion', action='store_true', help='Print confusion matrices for classification tasks')
    p.add_argument('--out_csv', type=str, default='interpretability/letter_world/results/probe_same_step.csv')
    return p.parse_args()


def main():
    args = parse_args()
    d = load_data(args.data)
    feats = [f.strip() for f in args.features.split(',') if f.strip()]
    targs = [t.strip() for t in args.targets.split(',') if t.strip()]
    rows = []

    # derive additional labels
    derived = {}
    if 'on_letter' in targs:
        derived['on_letter'] = build_on_letter_labels(d)
    if 'delta' in targs or 'action' in targs:
        derived['delta'], derived['action'] = build_delta_and_action(d)
    if 'delta_cls' in targs:
        if 'delta' not in derived:
            derived['delta'], _ = build_delta_and_action(d)
        derived['delta_cls'] = np.array([to_delta_cls(dd) for dd in derived['delta']], dtype=object)

    for feat in feats:
        # Build feature array
        if feat == 'obs_raw_flat':
            X_all = np.array([r.reshape(-1) if is_arr(r) else None for r in d['obs_raw']], dtype=object)
        else:
            X_all = d[feat]
        mask = np.array([is_arr(x) for x in X_all])
        if mask.sum() < 10:
            print(f"{feat}: not enough samples ({mask.sum()})")
            continue
        X = np.stack(X_all[mask], axis=0)
        groups_all = d['episode'] if 'episode' in d.files else None
        groups = groups_all[mask] if groups_all is not None else None
        for targ in targs:
            if targ == 'next_pos':
                Y_all = d[targ][mask]
                ok = np.array([is_arr(y) for y in Y_all])
                if ok.sum() < 10:
                    print(f"R2({feat} -> {targ}): not enough samples")
                    continue
                Y = np.stack(Y_all[ok], axis=0)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()}")
                r2, mae = run_regression(X[ok], Y, args.split, groups=g)
                print(f"R2({feat} -> {targ}): {r2:.3f} | MAE: {mae:.3f}")
                rows.append({'feature': feat, 'target': targ, 'metric': 'R2', 'score': float(r2), 'n': int(ok.sum())})
                rows.append({'feature': feat, 'target': targ, 'metric': 'MAE', 'score': float(mae), 'n': int(ok.sum())})
            elif targ == 'on_letter':
                y_all = derived['on_letter'][mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"acc({feat} -> {targ}): not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                if args.use_mlp:
                    acc = run_classification_mlp(X[ok], y, args.split, groups=g)
                else:
                    acc = run_classification(X[ok], y, args.split, groups=g)
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                rows.append({'feature': feat, 'target': targ, 'metric': 'acc', 'score': float(acc), 'n': int(ok.sum()), 'baseline_majority': float(base)})
                if args.confusion:
                    # Re-train to print confusion
                    if g is not None:
                        gss = GroupShuffleSplit(n_splits=1, train_size=args.split, random_state=0)
                        tr, te = next(gss.split(np.arange(len(X[ok])), groups=g))
                    else:
                        idx = np.arange(len(X[ok])); np.random.shuffle(idx); s = int(args.split * len(X[ok])); tr, te = idx[:s], idx[s:]
                    clf = make_pipeline(StandardScaler(with_mean=True), (MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True) if args.use_mlp else LogisticRegression(max_iter=2000)))
                    clf.fit(X[ok][tr], y[tr])
                    pred = clf.predict(X[ok][te])
                    print(confusion_matrix(y[te], pred, labels=sorted(np.unique(y))))
            elif targ == 'action':
                y_all = derived['action'][mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"acc({feat} -> {targ}): not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                if args.use_mlp:
                    acc = run_classification_mlp(X[ok], y, args.split, groups=g)
                else:
                    acc = run_classification(X[ok], y, args.split, groups=g)
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                rows.append({'feature': feat, 'target': targ, 'metric': 'acc', 'score': float(acc), 'n': int(ok.sum()), 'baseline_majority': float(base)})
                if args.confusion:
                    if g is not None:
                        gss = GroupShuffleSplit(n_splits=1, train_size=args.split, random_state=0)
                        tr, te = next(gss.split(np.arange(len(X[ok])), groups=g))
                    else:
                        idx = np.arange(len(X[ok])); np.random.shuffle(idx); s = int(args.split * len(X[ok])); tr, te = idx[:s], idx[s:]
                    clf = make_pipeline(StandardScaler(with_mean=True), (MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True) if args.use_mlp else LogisticRegression(max_iter=2000)))
                    clf.fit(X[ok][tr], y[tr])
                    pred = clf.predict(X[ok][te])
                    print(confusion_matrix(y[te], pred, labels=sorted(np.unique(y))))
            elif targ == 'delta':
                y_all = derived['delta'][mask]
                ok = np.array([is_arr(yy) for yy in y_all])
                if ok.sum() < 10:
                    print(f"R2({feat} -> {targ}): not enough samples")
                    continue
                Y = np.stack(y_all[ok], axis=0)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()}")
                r2, mae = run_regression(X[ok], Y, args.split, groups=g)
                print(f"R2({feat} -> {targ}): {r2:.3f} | MAE: {mae:.3f}")
                rows.append({'feature': feat, 'target': targ, 'metric': 'R2', 'score': float(r2), 'n': int(ok.sum())})
                rows.append({'feature': feat, 'target': targ, 'metric': 'MAE', 'score': float(mae), 'n': int(ok.sum())})
            elif targ == 'delta_cls':
                y_all = derived['delta_cls'][mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"acc({feat} -> {targ}): not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                if args.use_mlp:
                    acc = run_classification_mlp(X[ok], y, args.split, groups=g)
                else:
                    acc = run_classification(X[ok], y, args.split, groups=g)
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                rows.append({'feature': feat, 'target': targ, 'metric': 'acc', 'score': float(acc), 'n': int(ok.sum()), 'baseline_majority': float(base)})
                if args.confusion:
                    if g is not None:
                        gss = GroupShuffleSplit(n_splits=1, train_size=args.split, random_state=0)
                        tr, te = next(gss.split(np.arange(len(X[ok])), groups=g))
                    else:
                        idx = np.arange(len(X[ok])); np.random.shuffle(idx); s = int(args.split * len(X[ok])); tr, te = idx[:s], idx[s:]
                    clf = make_pipeline(StandardScaler(with_mean=True), (MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True) if args.use_mlp else LogisticRegression(max_iter=2000)))
                    clf.fit(X[ok][tr], y[tr])
                    pred = clf.predict(X[ok][te])
                    print(confusion_matrix(y[te], pred, labels=sorted(np.unique(y))))
            else:
                print(f"Skipping unknown target {targ}")

    # write CSV
    if rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # collect all keys present to be robust
        all_keys = set()
        for r in rows:
            all_keys |= set(r.keys())
        fieldnames = sorted(list(all_keys))
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
