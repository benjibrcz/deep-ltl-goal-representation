#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import csv
from pathlib import Path


FEATURES = [
    'actor_prelogits',
    'actor_mid',
    'actor_in',
    'obs_conv',
    'obs_emb',
    'e_sigma',
    'critic_mid',
    'h_seq',
    'obs_raw_flat',
    'obs_local_flat',
]

TARGETS = [
    'action_next',
    'delta_cls_next',
    'on_letter_next',
]


def is_arr(x):
    return isinstance(x, np.ndarray)


def majority_baseline(y):
    vals, cnts = np.unique(y, return_counts=True)
    return cnts.max() / cnts.sum() if len(cnts) else 0.0


def build_on_letter_labels(d):
    feats = d['obs_raw']
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
        others = np.delete(f, agent_ch, axis=-1)
        letter_here = (others.sum(axis=-1) > 0)
        y.append(int((agent_mask & letter_here).any()))
    return np.array(y, dtype=object)


def signed_step(step, G):
    v = step % G
    if v == 0:
        return 0
    return 1 if v == 1 else -1


def build_delta_signed_and_action(d):
    pos, nxt = d['pos'], d['next_pos']
    # infer grid size
    f0 = next((f for f in d['obs_raw'] if is_arr(f)), None)
    G = f0.shape[0] if f0 is not None else 7
    y_delta, y_action = [], []
    logged_action = d['action'] if 'action' in d.files else None
    for k, (p, n) in enumerate(zip(pos, nxt)):
        if not (is_arr(p) and is_arr(n)):
            y_delta.append(None)
            y_action.append(None)
            continue
        dx = signed_step(int(n[0]) - int(p[0]), G)
        dy = signed_step(int(n[1]) - int(p[1]), G)
        y_delta.append([dx, dy])
        if logged_action is not None:
            try:
                y_action.append(int(logged_action[k]))
                continue
            except Exception:
                pass
        if   dx ==  0 and dy ==  1: act = 0
        elif dx ==  0 and dy == -1: act = 1
        elif dx ==  1 and dy ==  0: act = 2
        elif dx == -1 and dy ==  0: act = 3
        else: act = None
        y_action.append(act)
    return np.array(y_delta, dtype=object), np.array(y_action, dtype=object)


def to_delta_cls(dd):
    if not is_arr(dd):
        return None
    dx, dy = int(dd[0]), int(dd[1])
    if   dx == 0 and dy ==  1: return 0
    if   dx == 0 and dy == -1: return 1
    if   dx == 1 and dy ==  0: return 2
    if   dx == -1 and dy == 0: return 3
    return None


def time_shift_pairs(episodes, k=1):
    t_idx, tp_idx = [], []
    for e in np.unique(episodes):
        idx = np.where(episodes == e)[0]
        if len(idx) <= k:
            continue
        t_idx.append(idx[:-k])
        tp_idx.append(idx[k:])
    if not t_idx:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(t_idx), np.concatenate(tp_idx)


def run_clf(X, y, split=0.8, groups=None, use_mlp=False, want_conf=False, class_weight=None, pca_dim: int = 0):
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=0)
        tr, te = next(gss.split(np.arange(len(X)), groups=groups))
    else:
        idx = np.arange(len(X)); np.random.shuffle(idx)
        s = int(split * len(X)); tr, te = idx[:s], idx[s:]
    clf = (MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True)
           if use_mlp else LogisticRegression(max_iter=2000, class_weight=class_weight))
    if pca_dim and pca_dim > 0:
        pipe = make_pipeline(StandardScaler(with_mean=True), PCA(n_components=pca_dim, whiten=True, random_state=0), clf)
    else:
        pipe = make_pipeline(StandardScaler(with_mean=True), clf)
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    acc = accuracy_score(y[te], pred)
    cm = confusion_matrix(y[te], pred, labels=sorted(np.unique(y))) if want_conf else None
    return acc, cm


def delta_cross_entropy(X_phi, a_t, y, split=0.8, groups=None, class_weight=None, pca_dim: int = 0):
    # Compute ΔCE = CE([a_t, φ_t] -> y) - CE([a_t] -> y)
    n = len(X_phi)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, train_size=split, random_state=0)
        tr, te = next(gss.split(np.arange(n), groups=groups))
    else:
        idx = np.arange(n); np.random.shuffle(idx); s = int(split*n); tr, te = idx[:s], idx[s:]
    # one-hot prev action
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    a_oh = enc.fit_transform(a_t.reshape(-1,1))
    # model D: [a_t, φ_t]
    if pca_dim and pca_dim > 0:
        pca = PCA(n_components=pca_dim, whiten=True, random_state=0)
        Xp_tr = pca.fit_transform(X_phi[tr])
        Xp_te = pca.transform(X_phi[te])
        XD_tr = np.concatenate([a_oh[tr], Xp_tr], axis=1)
        XD_te = np.concatenate([a_oh[te], Xp_te], axis=1)
    else:
        XD_tr = np.concatenate([a_oh[tr], X_phi[tr]], axis=1)
        XD_te = np.concatenate([a_oh[te], X_phi[te]], axis=1)
    clfD = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, class_weight=class_weight))
    clfD.fit(XD_tr, y[tr])
    pD = clfD.predict_proba(XD_te)
    ceD = log_loss(y[te], pD)
    # baseline A: a_t only
    clfA = LogisticRegression(max_iter=2000, class_weight=class_weight)
    clfA.fit(a_oh[tr], y[tr])
    pA = clfA.predict_proba(a_oh[te])
    ceA = log_loss(y[te], pA)
    return ceD - ceA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--features', type=str, default=','.join(FEATURES))
    p.add_argument('--targets', type=str, default=','.join(TARGETS))
    p.add_argument('--shift', type=int, default=1, help='predict k steps ahead')
    p.add_argument('--split', type=float, default=0.8)
    p.add_argument('--use_mlp', action='store_true')
    p.add_argument('--confusion', action='store_true')
    p.add_argument('--balanced', action='store_true', help='Use class_weight=balanced for LogisticRegression')
    p.add_argument('--pca_dim', type=int, default=0, help='If >0, apply PCA to high-dim obs features')
    p.add_argument('--out_csv', type=str, default='interpretability/letter_world/results/probe_tpk.csv')
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.data, allow_pickle=True)
    feats = [f.strip() for f in args.features.split(',') if f.strip()]
    targs = [t.strip() for t in args.targets.split(',') if t.strip()]
    rows = []

    if 'episode' not in d.files:
        print("Dataset missing 'episode' for grouped splits; proceeding without groups.")
        episodes = None
    else:
        episodes = d['episode']

    # Derived labels at same step
    on_letter = build_on_letter_labels(d)
    delta_signed, action = build_delta_signed_and_action(d)
    delta_cls = np.array([to_delta_cls(dd) for dd in delta_signed], dtype=object)

    # After delta_cls is built
    # Consistency pass: learn mapping action_{t+1} <-> delta_{t+1} from data
    try:
        a_tp = np.asarray([aa for aa in action[tp_idx] if aa is not None], dtype=int)
        d_tp = np.asarray([dd for dd in delta_signed[tp_idx] if is_arr(dd)], dtype=object)
        from collections import Counter
        counts = {}
        for ai, dd in zip(a_tp, d_tp):
            k = (int(dd[0]), int(dd[1]))
            counts.setdefault(int(ai), Counter())
            counts[int(ai)][k] += 1
        A2D = {k: max(v, key=v.get) for k, v in counts.items() if len(v) > 0}
        D2A = {v: k for k, v in A2D.items()}
        def to_delta_cls_consistent(dd):
            if not is_arr(dd):
                return None
            return D2A.get((int(dd[0]), int(dd[1])), None)
        delta_cls = np.array([to_delta_cls_consistent(dd) for dd in delta_signed], dtype=object)
    except Exception:
        pass

    # Build time-shift indices
    if episodes is None:
        print('shift mode requires episodes for group-aware pairing; using naive indices.')
        n = len(action)
        t_idx = np.arange(0, max(0, n - args.shift))
        tp_idx = t_idx + args.shift
    else:
        t_idx, tp_idx = time_shift_pairs(episodes, k=args.shift)

    if len(t_idx) == 0:
        print('No valid (t, t+k) pairs found for shift', args.shift)
        return

    for feat in feats:
        # Build X at time t
        if feat == 'obs_raw_flat':
            X_all = np.array([r.reshape(-1) if is_arr(r) else None for r in d['obs_raw']], dtype=object)
        elif feat == 'obs_local_flat':
            # build local agent-centered crop (wrap-around) radius=2
            patches = []
            for f in d['obs_raw']:
                if not is_arr(f):
                    patches.append(None); continue
                H, W, C = f.shape
                flat = f.reshape(-1, C); sums = flat.sum(0)
                if (sums == 1).any():
                    agent_ch = int(np.where(sums == 1)[0][0])
                else:
                    agent_ch = int(np.argmin(sums))
                loc = np.argwhere(f[..., agent_ch] > 0.5)
                if len(loc) == 0:
                    patches.append(None); continue
                i, j = loc[0]
                radius = 2
                rows = [(i + r) % H for r in range(-radius, radius + 1)]
                cols = [(j + c) % W for c in range(-radius, radius + 1)]
                patch = f[np.ix_(rows, cols, range(C))].reshape(-1)
                patches.append(patch)
            X_all = np.array(patches, dtype=object)
        else:
            X_all = d[feat]
        # mask out invalid features at t indices
        feat_t = X_all[t_idx]
        feat_mask = np.array([is_arr(x) for x in feat_t])
        if feat_mask.sum() < 10:
            print(f"{feat}: not enough samples ({feat_mask.sum()})")
            continue
        X_t = np.stack(feat_t[feat_mask], axis=0)
        groups = (episodes[t_idx][feat_mask] if episodes is not None else None)

        for targ in targs:
            if targ == 'action_next':
                y_all = action[tp_idx][feat_mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"{feat} -> {targ}: not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                # class counts
                vals, cnts = np.unique(y, return_counts=True); print({int(v): int(c) for v, c in zip(vals, cnts)})
                use_pca = args.pca_dim > 0 and feat in ('obs_conv', 'obs_raw_flat')
                acc, cm = run_clf(X_t[ok], y, args.split, groups=g, use_mlp=args.use_mlp, want_conf=args.confusion, class_weight=('balanced' if args.balanced else None), pca_dim=(args.pca_dim if use_pca else 0))
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                if cm is not None:
                    print(cm)
                # AR(1) baseline: predict next action from current action
                act_t = action[t_idx][feat_mask][ok]
                act_t = np.asarray([a for a in act_t], dtype=int)
                print(f"[baseline AR(1)] acc(prev_action -> {targ}): {float((act_t==y).mean()):.3f}")
                # ΔCE conditional improvement over AR(1)
                try:
                    dce = delta_cross_entropy(X_t[ok], act_t, y, split=args.split, groups=g, class_weight=('balanced' if args.balanced else None), pca_dim=(args.pca_dim if use_pca else 0))
                    print(f"[ΔCE] CE([a_t, {feat}] -> {targ}) - CE(a_t -> {targ}) = {dce:.4f}")
                except Exception as e:
                    print(f"[ΔCE] failed: {e}")
                # collect row
                rows.append({
                    'feature': feat,
                    'target': targ,
                    'shift': args.shift,
                    'n': int(ok.sum()),
                    'metric': 'acc',
                    'score': float(acc),
                    'baseline_majority': float(base),
                    'baseline_ar1': float((act_t==y).mean()),
                    'delta_ce': float(dce) if 'dce' in locals() else np.nan,
                })

            elif targ == 'delta_cls_next':
                y_all = delta_cls[tp_idx][feat_mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"{feat} -> {targ}: not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                vals, cnts = np.unique(y, return_counts=True); print({int(v): int(c) for v, c in zip(vals, cnts)})
                use_pca = args.pca_dim > 0 and feat in ('obs_conv', 'obs_raw_flat')
                acc, cm = run_clf(X_t[ok], y, args.split, groups=g, use_mlp=args.use_mlp, want_conf=args.confusion, class_weight=('balanced' if args.balanced else None), pca_dim=(args.pca_dim if use_pca else 0))
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                if cm is not None:
                    print(cm)
                # AR(1) baseline: prev action as proxy for next delta class
                act_t = action[t_idx][feat_mask][ok]
                act_t = np.asarray([a for a in act_t], dtype=int)
                print(f"[baseline AR(1)] acc(prev_action -> {targ}): {float((act_t==y).mean()):.3f}")
                try:
                    dce = delta_cross_entropy(X_t[ok], act_t, y, split=args.split, groups=g, class_weight=('balanced' if args.balanced else None), pca_dim=(args.pca_dim if use_pca else 0))
                    print(f"[ΔCE] CE([a_t, {feat}] -> {targ}) - CE(a_t -> {targ}) = {dce:.4f}")
                except Exception as e:
                    print(f"[ΔCE] failed: {e}")
                rows.append({
                    'feature': feat,
                    'target': targ,
                    'shift': args.shift,
                    'n': int(ok.sum()),
                    'metric': 'acc',
                    'score': float(acc),
                    'baseline_majority': float(base),
                    'baseline_ar1': float((act_t==y).mean()),
                    'delta_ce': float(dce) if 'dce' in locals() else np.nan,
                })

            elif targ == 'on_letter_next':
                y_all = build_on_letter_labels(d)[tp_idx][feat_mask]
                ok = np.array([yy is not None for yy in y_all])
                if ok.sum() < 10 or len(np.unique(y_all[ok])) < 2:
                    print(f"{feat} -> {targ}: not enough class variety")
                    continue
                y = np.asarray(y_all[ok], dtype=int)
                base = majority_baseline(y)
                g = groups[ok] if groups is not None else None
                print(f"{feat} -> {targ}: n={ok.sum()} [baseline {base:.3f}]")
                vals, cnts = np.unique(y, return_counts=True); print({int(v): int(c) for v, c in zip(vals, cnts)})
                use_pca = args.pca_dim > 0 and feat in ('obs_conv', 'obs_raw_flat')
                acc, cm = run_clf(X_t[ok], y, args.split, groups=g, use_mlp=args.use_mlp, want_conf=args.confusion, class_weight=('balanced' if args.balanced else None), pca_dim=(args.pca_dim if use_pca else 0))
                print(f"acc({feat} -> {targ}): {acc:.3f}")
                if cm is not None:
                    print(cm)
                rows.append({
                    'feature': feat,
                    'target': targ,
                    'shift': args.shift,
                    'n': int(ok.sum()),
                    'metric': 'acc',
                    'score': float(acc),
                    'baseline_majority': float(base),
                    'baseline_ar1': np.nan,
                    'delta_ce': np.nan,
                })
            else:
                print(f"Skipping unknown target {targ}")

    # write CSV
    if rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ['feature', 'target', 'shift', 'n', 'metric', 'score', 'baseline_majority', 'baseline_ar1', 'delta_ce']
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()


