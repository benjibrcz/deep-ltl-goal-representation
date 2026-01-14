
#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
import csv
from pathlib import Path


def is_arr(x):
    return isinstance(x, np.ndarray)


def build_obs_local(d, radius=2):
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
        rows = [(i + r) % H for r in range(-radius, radius + 1)]
        cols = [(j + c) % W for c in range(-radius, radius + 1)]
        patch = f[np.ix_(rows, cols, range(C))].reshape(-1)
        patches.append(patch)
    return np.array(patches, dtype=object)


def delta_ce(X_base, X_test, y, groups, train_size=0.8):
    n = len(y)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=0)
    tr, te = next(gss.split(np.arange(n), groups=groups))
    base = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    base.fit(X_base[tr], y[tr])
    pA = base.predict_proba(X_base[te])
    ceA = log_loss(y[te], pA)
    test = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    test.fit(X_test[tr], y[tr])
    pD = test.predict_proba(X_test[te])
    ceD = log_loss(y[te], pD)
    return ceD - ceA


def ce_score(X, y, groups, train_size=0.8):
    n = len(y)
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=0)
    tr, te = next(gss.split(np.arange(n), groups=groups))
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[te])
    return float(log_loss(y[te], p, labels=clf.classes_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--features', type=str, default='actor_prelogits,actor_mid,obs_conv')
    ap.add_argument('--shift', type=int, default=1)
    ap.add_argument('--target', default='action_next', choices=['action_next', 'on_letter_next'])
    ap.add_argument('--out_csv', type=str, default='interpretability/letter_world/results/sufficiency.csv')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    episodes = d['episode'] if 'episode' in d.files else np.zeros(len(d['action']))
    features = [f.strip() for f in args.features.split(',') if f.strip()]

    rows = []
    for feat in features:
        # features at t
        Z_all = d[feat]
        mask = np.array([is_arr(z) for z in Z_all])
        if mask.sum() < 10:
            print(f"{feat}: not enough feature vectors")
            continue
        Z = np.stack(Z_all[mask])
        A = np.asarray(d['action'][mask], dtype=int)
        E = episodes[mask]
        # local obs at t
        OL_all = build_obs_local(d)
        OL = OL_all[mask]
        ok_ol = np.array([is_arr(o) for o in OL])
        OL = np.stack(OL[ok_ol])
        Z = Z[ok_ol]; A = A[ok_ol]; E = E[ok_ol]

        # build t, t+shift pairs within episodes
        t_idx, tp_idx = [], []
        for e in np.unique(E):
            idx = np.where(E == e)[0]
            if len(idx) <= args.shift:
                continue
            t_idx.append(idx[:-args.shift]); tp_idx.append(idx[args.shift:])
        if not t_idx:
            print(f'{feat}: No (t, t+shift) pairs')
            continue
        t = np.concatenate(t_idx); tp = np.concatenate(tp_idx)

        if args.target == 'action_next':
            y = A[tp]
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            a_oh = enc.fit_transform(A[t].reshape(-1, 1))
            dA = delta_ce(a_oh, np.concatenate([a_oh, Z[t]], axis=1), y, E[t])
            X_base = np.concatenate([a_oh, OL[t]], axis=1)
            X_test = np.concatenate([a_oh, OL[t], Z[t]], axis=1)
            dAZ = delta_ce(X_base, X_test, y, E[t])
            print(f"{feat}: ΔCE(z_t | a_t)={dA:.4f} | ΔCE(z_t | a_t, obs_local)={dAZ:.4f}")
            rows.append({'feature': feat, 'shift': args.shift, 'target': 'action_next', 'delta_ce_a': float(dA), 'delta_ce_a_obs': float(dAZ), 'n': int(len(t))})
            # Absolute CE ablations
            ce_at = ce_score(a_oh, y, E[t])
            ce_z = ce_score(Z[t], y, E[t])
            ce_obs = ce_score(OL[t], y, E[t])
            ce_z_obs = ce_score(np.concatenate([Z[t], OL[t]], axis=1), y, E[t])
            ce_at_z = ce_score(np.concatenate([a_oh, Z[t]], axis=1), y, E[t])
            ce_at_obs = ce_score(np.concatenate([a_oh, OL[t]], axis=1), y, E[t])
            ce_at_z_obs = ce_score(np.concatenate([a_oh, Z[t], OL[t]], axis=1), y, E[t])
            print(f"{feat}: CE(a_t)={ce_at:.3f} CE(z)={ce_z:.3f} CE(obs)={ce_obs:.3f} CE(z+obs)={ce_z_obs:.3f}")
            print(f"{feat}: CE(a_t+z)={ce_at_z:.3f} CE(a_t+obs)={ce_at_obs:.3f} CE(a_t+z+obs)={ce_at_z_obs:.3f}")
            rows[-1].update({'ce_at': ce_at, 'ce_z': ce_z, 'ce_obs': ce_obs, 'ce_z_obs': ce_z_obs, 'ce_at_z': ce_at_z, 'ce_at_obs': ce_at_obs, 'ce_at_z_obs': ce_at_z_obs})
        else:
            # on_letter_{t+shift}
            def build_on_letter_labels_local(dataset):
                feats = dataset['obs_raw']
                yloc = []
                for f in feats:
                    if not is_arr(f):
                        yloc.append(None); continue
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
                    yloc.append(int((agent_mask & letter_here).any()))
                return np.array(yloc, dtype=object)
            y_all = build_on_letter_labels_local(d)
            y = np.asarray(y_all[tp], dtype=int)
            dZ = delta_ce(OL[t], np.concatenate([OL[t], Z[t]], axis=1), y, E[t])
            print(f"{feat}: ΔCE(z_t | obs_local)={dZ:.4f}")
            rows.append({'feature': feat, 'shift': args.shift, 'target': 'on_letter_next', 'delta_ce_obs': float(dZ), 'n': int(len(t))})
            # Absolute CE ablations (obs vs z vs z+obs)
            ce_obs = ce_score(OL[t], y, E[t])
            ce_z = ce_score(Z[t], y, E[t])
            ce_z_obs = ce_score(np.concatenate([Z[t], OL[t]], axis=1), y, E[t])
            print(f"{feat}: CE(obs)={ce_obs:.3f} CE(z)={ce_z:.3f} CE(z+obs)={ce_z_obs:.3f}")
            rows[-1].update({'ce_obs': ce_obs, 'ce_z': ce_z, 'ce_z_obs': ce_z_obs})

    # write CSV
    if rows:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(set().union(*[r.keys() for r in rows]))
        with out_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved sufficiency results to {out_path}")


if __name__ == '__main__':
    main()

