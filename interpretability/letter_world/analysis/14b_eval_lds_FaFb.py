#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression


def is_arr(x):
    return isinstance(x, np.ndarray)


def load_lds(path):
    q = np.load(path, allow_pickle=True)
    A = q['A']; B = q['B']; b = q['b']
    A2 = q['A2'] if 'A2' in q.files else None
    mu = q['mu'] if 'mu' in q.files else None
    sigma = q['sigma'] if 'sigma' in q.files else None
    return A, B, b, A2, mu, sigma


def step_affine(A, B, b, z, u_onehot):
    return (A @ z + B @ u_onehot + b)


def find_first_two_letters(ep_idx: np.ndarray, letters: np.ndarray):
    t_a = None; t_b = None; A_id = None
    for tloc in range(len(ep_idx)):
        li = int(letters[ep_idx[tloc]])
        if li < 0:
            continue
        if t_a is None:
            t_a, A_id = tloc, li
        elif li != A_id:
            t_b = tloc
            break
    return t_a, t_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='feature_t')
    ap.add_argument('--lds', required=True)
    ap.add_argument('--step_stride', type=int, default=5, help='evaluate every N steps')
    ap.add_argument('--max_k', type=int, default=100, help='maximum horizon to consider if B is far')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/lds_FaFb_per5.csv')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    if 'letter_id' not in d.files:
        raise SystemExit('letter_id missing; use sequential rollouts with letter_id (03c_log_rollouts_seq.py)')

    Z_all = d[args.feature]
    A_all = d['action']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(Z_all))
    L_all = d['letter_id']

    mask = np.array([is_arr(z) for z in Z_all])
    Z = np.stack(Z_all[mask])
    A = np.asarray(A_all[mask], dtype=int)
    E = np.asarray(E_all[mask])
    L = np.asarray(L_all[mask], dtype=int)

    # Train/test split by episode
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z, groups=E))

    # Load LDS and scale if provided
    A_est, B_est, b_est, A2_est, mu, sigma = load_lds(args.lds)
    if mu is not None and sigma is not None:
        sigma_safe = np.maximum(sigma, 1e-8)
        Z_s = (Z - mu) / sigma_safe
    else:
        Z_s = Z

    # Same-step decoder on train episodes
    dec = LogisticRegression(max_iter=2000)
    dec.fit(Z_s[tr], A[tr])

    K = int(A.max()) + 1

    # Aggregators per k
    from collections import defaultdict
    counts_total = defaultdict(int)
    hits_total = defaultdict(int)
    counts_pre = defaultdict(int)
    hits_pre = defaultdict(int)
    counts_post = defaultdict(int)
    hits_post = defaultdict(int)
    episodes_used = 0
    tA_list = []
    tB_list = []

    # Iterate test episodes only
    for e in np.unique(E[te]):
        ep_idx = np.where(E == e)[0]
        ep_te = np.intersect1d(ep_idx, te)
        if len(ep_te) < 3:
            continue
        # require Fa & Fb structure
        t_a, t_b = find_first_two_letters(ep_te, L)
        if t_a is None or t_b is None:
            continue
        episodes_used += 1
        tA_list.append(int(t_a))
        tB_list.append(int(t_b))

        # For each start j within episode, roll by k in multiples of stride, stop before B
        for j_local in range(0, len(ep_te)):
            for k in range(args.step_stride, min(args.max_k, len(ep_te)) + 1, args.step_stride):
                if j_local + k >= len(ep_te):
                    break
                # ensure contiguity in compressed indexing
                if (ep_te[j_local + k] - ep_te[j_local]) != k:
                    continue
                # stop if beyond B
                if (j_local + k) >= t_b:
                    break
                ti = int(ep_te[j_local])
                tp = int(ep_te[j_local + k])
                # roll with true actions for k steps
                if A2_est is not None:
                    if ti - 1 < 0 or E[ti - 1] != E[ti]:
                        continue
                    z_prev = Z_s[ti - 1].copy()
                z_cur = Z_s[ti].copy()
                ok = True
                for s in range(k):
                    a = int(A[ti + s])
                    u = np.eye(K)[a]
                    if A2_est is not None:
                        z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ u + b_est)
                        z_prev, z_cur = z_cur, z_next
                    else:
                        z_cur = step_affine(A_est, B_est, b_est, z_cur, u)
                    # guard episode boundary
                    if (ti + s + 1) >= len(E) or E[ti + s + 1] != E[ti]:
                        ok = False; break
                if not ok:
                    continue
                a_pred = int(dec.predict(z_cur.reshape(1, -1))[0])
                correct = int(a_pred == int(A[tp]))

                counts_total[k] += 1
                hits_total[k] += correct
                if (j_local + k) < t_a:
                    counts_pre[k] += 1; hits_pre[k] += correct
                else:
                    counts_post[k] += 1; hits_post[k] += correct

        # fallback pass anchored at episode start only (helps when windows are short)
        j0 = 0
        for k in range(args.step_stride, min(args.max_k, len(ep_te)) + 1, args.step_stride):
            if j0 + k >= len(ep_te):
                break
            if (ep_te[j0 + k] - ep_te[j0]) != k:
                continue
            if (j0 + k) >= t_b:
                break
            ti = int(ep_te[j0])
            tp = int(ep_te[j0 + k])
            if A2_est is not None:
                if ti - 1 < 0 or E[ti - 1] != E[ti]:
                    continue
                z_prev = Z_s[ti - 1].copy()
            z_cur = Z_s[ti].copy()
            ok = True
            for s in range(k):
                a = int(A[ti + s])
                u = np.eye(K)[a]
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ u + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = step_affine(A_est, B_est, b_est, z_cur, u)
                if (ti + s + 1) >= len(E) or E[ti + s + 1] != E[ti]:
                    ok = False; break
            if not ok:
                continue
            a_pred = int(dec.predict(z_cur.reshape(1, -1))[0])
            correct = int(a_pred == int(A[tp]))
            counts_total[k] += 1
            hits_total[k] += correct
            if (j0 + k) < t_a:
                counts_pre[k] += 1; hits_pre[k] += correct
            else:
                counts_post[k] += 1; hits_post[k] += correct

    # Write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    ks = sorted(counts_total.keys())
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['k','n_total','acc_total','n_preA','acc_preA','n_postA','acc_postA','frac_postA','episodes_used','tA_mean','tB_mean'])
        w.writeheader()
        for k in ks:
            n_tot = counts_total[k]
            n_pre = counts_pre[k]
            n_post = counts_post[k]
            acc_tot = (hits_total[k] / n_tot) if n_tot > 0 else float('nan')
            acc_pre = (hits_pre[k] / n_pre) if n_pre > 0 else float('nan')
            acc_post = (hits_post[k] / n_post) if n_post > 0 else float('nan')
            frac_post = (n_post / n_tot) if n_tot > 0 else float('nan')
            tA_mean = float(np.mean(tA_list)) if tA_list else float('nan')
            tB_mean = float(np.mean(tB_list)) if tB_list else float('nan')
            w.writerow(dict(k=int(k), n_total=int(n_tot), acc_total=acc_tot,
                            n_preA=int(n_pre), acc_preA=acc_pre,
                            n_postA=int(n_post), acc_postA=acc_post,
                            frac_postA=frac_post, episodes_used=int(episodes_used),
                            tA_mean=tA_mean, tB_mean=tB_mean))
    print('Saved Fa&Fb per-stride rollout accuracy to', out)


if __name__ == '__main__':
    main()


