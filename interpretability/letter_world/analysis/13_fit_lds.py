#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


def is_arr(x):
    return isinstance(x, np.ndarray)


def stack_pairs(Z, A, episodes, shift=1, order=1):
    t_list, tp_list, tm1_list, a_list, g_list = [], [], [], [], []
    for e in np.unique(episodes):
        ep_idx = np.where(episodes == e)[0]
        if len(ep_idx) <= shift:
            continue
        # enforce contiguous windows inside the episode
        for j in range(len(ep_idx) - shift):
            t = int(ep_idx[j])
            tp = int(ep_idx[j + shift])
            if tp - t != shift:
                continue
            if order >= 2:
                tm1 = t - 1
                if tm1 < ep_idx[0]:
                    continue
            t_list.append(t)
            tp_list.append(tp)
            if order >= 2:
                tm1_list.append(tm1)
            a_list.append(int(A[t]))
            g_list.append(int(e))
    if not t_list:
        return None
    Zt = Z[t_list]
    Ztp = Z[tp_list]
    Ztm1 = Z[tm1_list] if order >= 2 else None
    at = np.asarray(a_list, dtype=int)
    ge = np.asarray(g_list, dtype=int)
    return Zt, Ztm1, Ztp, at, ge


def onehot(a, K):
    M = np.zeros((len(a), K), dtype=float)
    M[np.arange(len(a)), a.astype(int)] = 1.0
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--shift', type=int, default=1)
    ap.add_argument('--order', type=int, default=1, choices=[1,2])
    ap.add_argument('--scale', action='store_true')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = np.asarray(d['action'])
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(Z_all))

    mask = np.array([is_arr(z) for z in Z_all])
    if mask.sum() < 10:
        print('Not enough feature vectors to fit LDS')
        return
    Z = np.stack(Z_all[mask])
    A = A_all[mask]
    E = E_all[mask]

    pairs = stack_pairs(Z, A, E, args.shift, order=args.order)
    if pairs is None:
        print('No valid (t, t+k) pairs found')
        return
    Zt, Ztm1, Ztp, at, groups = pairs
    K = int(A.max()) + 1
    U = onehot(at, K)
    if args.order >= 2:
        X = np.concatenate([Zt, Ztm1, U, np.ones((len(Zt), 1))], axis=1)
    else:
        X = np.concatenate([Zt, U, np.ones((len(Zt), 1))], axis=1)
    Y = Ztp

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(X, groups=groups))

    # optional scaling of Z features (not one-hot or bias)
    scaler = None
    if args.scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        if args.order >= 2:
            # fit on concatenated Z parts of training set only
            Dz = Z.shape[1]
            Ztr = Zt[tr]
            Ztrm1 = Ztm1[tr]
            scaler.fit(np.concatenate([Ztr, Ztrm1], axis=0))
            Zt_s = scaler.transform(Zt)
            Ztm1_s = scaler.transform(Ztm1)
            Ztp_s = scaler.transform(Ztp)
            Xz_tr = np.concatenate([Zt_s, Ztm1_s], axis=1)
        else:
            scaler.fit(Zt[tr])
            Zt_s = scaler.transform(Zt)
            Ztp_s = scaler.transform(Ztp)
            Xz_tr = Zt_s
        # rebuild X with scaled Z blocks
        if args.order >= 2:
            Xs = np.concatenate([Zt_s, Ztm1_s, U, np.ones((len(Zt), 1))], axis=1)
        else:
            Xs = np.concatenate([Zt_s, U, np.ones((len(Zt), 1))], axis=1)
        Ys = Ztp_s
        reg = Ridge(alpha=args.alpha, fit_intercept=True).fit(Xs[tr], Ys[tr])
        Yhat = reg.predict(Xs[te])
        Ytrue = Ys[te]
    else:
        reg = Ridge(alpha=args.alpha, fit_intercept=True).fit(X[tr], Y[tr])
        Yhat = reg.predict(X[te])
        Ytrue = Y[te]

    ss_res = np.sum((Ytrue - Yhat) ** 2)
    ss_tot = np.sum((Ytrue - Ytrue.mean(0)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"{args.feature}: R2(z_t, a_t -> z_{args.shift+1}) = {r2:.3f} (order={args.order}, scaled={bool(args.scale)})")

    D = Z.shape[1]
    # reg.coef_: shape (D_out, D_in+K+1)
    W = reg.coef_.T  # (input_dim, D_out)
    if args.order >= 2:
        A1_est = W[:D, :].T
        A2_est = W[D:2*D, :].T
        B_est = W[2*D:2*D + K, :].T
    else:
        A1_est = W[:D, :].T
        A2_est = None
        B_est = W[D:D + K, :].T
    b_est = reg.intercept_  # (D_out,)
    # spectral norm (approx)
    try:
        spec = np.linalg.norm(A1_est, 2)
    except Exception:
        spec = float(np.linalg.norm(A1_est))
    print("A1 spectral norm (approx):", spec)

    out = f"{args.feature}_lds_shift{args.shift}.npz"
    save_dict = dict(A=A1_est, B=B_est, b=b_est, r2=r2, order=args.order)
    if A2_est is not None:
        save_dict['A2'] = A2_est
    if scaler is not None:
        save_dict['mu'] = scaler.mean_
        save_dict['sigma'] = scaler.scale_
    np.savez_compressed(out, **save_dict)
    print("Saved", out)


if __name__ == '__main__':
    main()


