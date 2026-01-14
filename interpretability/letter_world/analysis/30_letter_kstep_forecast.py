#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def infer_next_letter_from_obs(obs_next_raw: np.ndarray) -> np.ndarray:
    X = np.asarray(obs_next_raw)
    if X.dtype == object:
        X = np.stack(list(X), axis=0)
    # agent channel = sparsest channel across grid
    ch_sums = X.reshape(X.shape[0], -1, X.shape[-1]).sum(axis=1).mean(axis=0)
    agent_ch = int(np.argmin(ch_sums))
    # letter under agent = argmax letter channel where agent=1; fallback to -1
    N, H, W, C = X.shape
    agent_mask = (X[..., agent_ch] > 0.5)
    letter_idx = np.full((N,), -1, dtype=int)
    for i in range(N):
        loc = np.argwhere(agent_mask[i])
        if len(loc) == 0:
            continue
        r, c = loc[0]
        # among non-agent channels, pick active letter at (r,c)
        vals = X[i, r, c, :]
        vals[agent_ch] = 0.0
        letter_idx[i] = int(np.argmax(vals))
    return letter_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='branched CLEAN NPZ (supports K=1)')
    ap.add_argument('--feature_key', default='feature_t')
    ap.add_argument('--action_key', default='action', help='or branched_action')
    ap.add_argument('--next_key', default='obs_next_raw')
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--variant', default='plus', choices=['base','plus','action_only'])
    ap.add_argument('--min_k', type=int, default=3)
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/letter_kstep_forecast.csv')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    F = D[args.feature_key]
    if isinstance(F, np.ndarray) and F.dtype == object:
        F = np.vstack([np.asarray(z).ravel() for z in F]).astype(np.float32)
    else:
        F = np.asarray(F)
        if F.ndim > 2:
            F = F.reshape(F.shape[0], -1)
        F = F.astype(np.float32)
    # group id per base
    B = np.asarray(D['base_id'] if 'base_id' in D.files else D['source_id']).astype(int)
    # Targets: if K==1, infer from next obs; if K>1, use stored letter_next_id
    if args.K == 1:
        next_obs = D[args.next_key]
        y = infer_next_letter_from_obs(next_obs)
    else:
        if 'letter_next_id' not in D.files or 'a_seq' not in D.files:
            raise SystemExit('K>1 requires branched_kstep.npz with a_seq and letter_next_id')
        y = np.asarray(D['letter_next_id']).astype(int)

    # filter to rows with valid letter id
    ok = (y >= 0)
    F, B, y = F[ok], B[ok], y[ok]

    # design matrix by variant
    if args.variant == 'base':
        X = F
    elif args.variant == 'action_only':
        if args.K == 1:
            # use single-step actions; derive from a_seq if action key missing
            if args.action_key in D.files:
                A = np.asarray(D[args.action_key])[ok].astype(int)
            elif 'a_seq' in D.files:
                A = np.array([int(seq[0]) for seq in D['a_seq'][ok]], dtype=int)
            else:
                raise SystemExit("No action or a_seq found for K=1 action_only variant")
            Kact = int(A.max()) + 1
            X = np.eye(Kact, dtype=np.float32)[A]
        else:
            # multi-step action sequence one-hot flattened
            if 'a_seq' not in D.files:
                raise SystemExit('K>1 requires a_seq for action encoding')
            Aseq = np.array(D['a_seq'][ok].tolist())
            Kact = int(Aseq.max()) + 1
            oh = np.eye(Kact, dtype=np.float32)
            X = oh[Aseq].reshape(Aseq.shape[0], -1)
    else:
        if args.K == 1:
            if args.action_key in D.files:
                A = np.asarray(D[args.action_key])[ok].astype(int)
            elif 'a_seq' in D.files:
                A = np.array([int(seq[0]) for seq in D['a_seq'][ok]], dtype=int)
            else:
                raise SystemExit("No action or a_seq found for K=1 plus variant")
            Kact = int(A.max()) + 1
            X = np.concatenate([F, np.eye(Kact, dtype=np.float32)[A]], axis=1)
        else:
            if 'a_seq' not in D.files:
                raise SystemExit('K>1 requires a_seq for action encoding')
            Aseq = np.array(D['a_seq'][ok].tolist())
            Kact = int(Aseq.max()) + 1
            oh = np.eye(Kact, dtype=np.float32)
            Aflat = oh[Aseq].reshape(Aseq.shape[0], -1)
            X = np.concatenate([F, Aflat], axis=1)

    # split by base id to keep branches together
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(np.arange(len(X)), groups=B))

    # fit multinomial logistic
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=1000, multi_class='multinomial'))
    clf.fit(X[tr], y[tr])
    yhat = clf.predict(X[te])
    acc = float(accuracy_score(y[te], yhat))

    # write CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant','K','acc','n_test'])
        w.writeheader()
        w.writerow(dict(variant=args.variant, K=args.K, acc=acc, n_test=int(len(te))))
    print('Saved letter K-step forecast to', out)


if __name__ == '__main__':
    main()


