#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def onehot(a: np.ndarray, K: int) -> np.ndarray:
    M = np.zeros((len(a), K), dtype=np.float32)
    M[np.arange(len(a)), a.astype(int)] = 1.0
    return M


def build_discounted_visitation(pos_rc: np.ndarray, epi: np.ndarray, H: int, gamma: float, grid: int,
                                k_min: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # Build v_t of shape [N, G] by scanning future positions up to H within each episode
    N = len(pos_rc)
    G = grid * grid
    V = np.zeros((N, G), dtype=np.float32)
    # index positions per episode
    for e in np.unique(epi):
        idx = np.where(epi == e)[0]
        r = pos_rc[idx]
        T = len(idx)
        for t_local in range(T):
            t_global = idx[t_local]
            # accumulate from k_min..H respecting episode end
            for k in range(k_min, H + 1):
                if t_local + k >= T:
                    break
                rr, cc = int(r[t_local + k, 0]), int(r[t_local + k, 1])
                V[t_global, rr * grid + cc] += (gamma ** (k - 1))
    return V, np.arange(N)


def r2_macro_stable(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    var = y_true.var(axis=0)
    keep = var > 1e-8
    if not np.any(keep):
        return float('nan')
    yt = y_true[:, keep]
    yp = y_pred[:, keep]
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    mean = yt.mean(axis=0, keepdims=True)
    ss_tot = np.sum((yt - mean) ** 2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot
    return float(np.mean(r2))


def cosine_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    a = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + eps)
    b = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + eps)
    return float(np.mean(np.sum(a * b, axis=1)))


def mse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_true - y_pred
    mse = float(np.mean(np.mean(err ** 2, axis=1)))
    mae = float(np.mean(np.mean(np.abs(err), axis=1)))
    return mse, mae


def topk_metrics(y_true: np.ndarray, y_pred: np.ndarray, ks=(3, 5)) -> dict:
    out = {}
    G = y_true.shape[1]
    for K in ks:
        Kc = min(K, G)
        top_pred = np.argpartition(-y_pred, Kc - 1, axis=1)[:, :Kc]
        true_pos = (y_true > 0)
        precs = []
        recs = []
        for i in range(y_true.shape[0]):
            tp = int(true_pos[i, top_pred[i]].sum())
            precs.append(tp / Kc)
            recs.append(tp / max(1, int(true_pos[i].sum())))
        out[f'top{K}_prec'] = float(np.mean(precs))
        out[f'top{K}_rec'] = float(np.mean(recs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='rollouts npz with feature_t/action/agent_pos/episode')
    ap.add_argument('--feature', default='feature_t', help='key for Z_t')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--grid', type=int, default=7)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/sr_probe.csv')
    ap.add_argument('--branched', type=str, default=None, help='Optional branched dataset for branch-selection test')
    ap.add_argument('--branched_feature', type=str, default=None, help='Feature key to use from branched NPZ (e.g., feature)')
    ap.add_argument('--permute_labels', action='store_true')
    ap.add_argument('--short_horizon', type=int, default=0, help='If >0, replace horizon with this value for near-future SR')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    Z_all = D[args.feature]
    A_all = np.asarray(D['action']).astype(int)
    P_all = D['agent_pos'] if 'agent_pos' in D.files else None
    if 'episode' not in D.files:
        raise SystemExit('episode is required (sequential rollouts). Use a rollouts NPZ, not the branched dataset.')
    E_all = D['episode']
    # coerce features to dense 2D
    if isinstance(Z_all, np.ndarray) and Z_all.dtype == object:
        Z = np.vstack([np.asarray(z).ravel() for z in Z_all]).astype(np.float32)
    else:
        Z = np.asarray(Z_all)
        if Z.ndim > 2:
            Z = Z.reshape(Z.shape[0], -1)
        Z = Z.astype(np.float32)

    # basic mask
    N = min(len(Z), len(A_all), len(P_all), len(E_all)) if P_all is not None else min(len(Z), len(A_all))
    Z, A_all = Z[:N], A_all[:N]
    if P_all is None:
        raise SystemExit('agent_pos required to build discounted visitation targets')
    P = np.asarray(P_all)[:N]
    E = np.asarray(E_all)[:N]

    # build targets with k>=2 masking
    V, keep_idx = build_discounted_visitation(P, E, H=args.horizon, gamma=args.gamma, grid=args.grid, k_min=2)
    Z, A, V, E = Z[keep_idx], A_all[keep_idx], V[keep_idx], E[keep_idx]

    # split by episode (require multiple episodes)
    if len(np.unique(E)) < 2:
        raise SystemExit('Need at least 2 episodes for grouped split; collect longer sequential rollouts.')
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    tr, te = next(gss.split(np.arange(len(Z)), groups=E))
    if args.permute_labels:
        rng = np.random.RandomState(0)
        rng.shuffle(V[te])

    K = int(A.max()) + 1
    Aoh = onehot(A, K)

    X_base = Z
    X_plus = np.concatenate([Z, Aoh], axis=1)
    X_act = Aoh

    # standardize targets on train (per-dimension)
    y_mean = V[tr].mean(axis=0, keepdims=True)
    y_std = V[tr].std(axis=0, keepdims=True) + 1e-8
    Vz_tr = (V[tr] - y_mean) / y_std
    Vz_te = (V[te] - y_mean) / y_std

    # probes (Ridge on standardized targets)
    probe = lambda: make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=args.alpha))
    m_base = probe().fit(X_base[tr], Vz_tr)
    m_plus = probe().fit(X_plus[tr], Vz_tr)
    m_act  = probe().fit(X_act[tr],  Vz_tr)

    Yb = m_base.predict(X_base[te]) * y_std + y_mean
    Yp = m_plus.predict(X_plus[te]) * y_std + y_mean
    Ya = m_act.predict(X_act[te]) * y_std + y_mean

    # metrics
    out_rows = []
    for name, Yhat in [('base', Yb), ('plus', Yp), ('action_only', Ya)]:
        r2 = r2_macro_stable(V[te], Yhat)
        cos = cosine_macro(V[te], Yhat)
        mse, mae = mse_mae(V[te], Yhat)
        t = topk_metrics(V[te], Yhat, ks=(3, 5))
        row = dict(split='test', variant=name, action='all', r2_macro=r2, cosine_macro=cos, mse=mse, mae=mae)
        row.update(t)
        out_rows.append(row)

        # action-balanced macro
        scores = []
        for a in range(K):
            idx = np.where(A[te] == a)[0]
            if len(idx) < 5:
                continue
            scores.append((r2_macro_stable(V[te][idx], Yhat[idx]), cosine_macro(V[te][idx], Yhat[idx])))
        if scores:
            r2_ab = float(np.mean([s[0] for s in scores]))
            cos_ab = float(np.mean([s[1] for s in scores]))
            out_rows.append(dict(split='test', variant=name+'_ab', action='macro', r2_macro=r2_ab, cosine_macro=cos_ab))

    # Shuffled baselines on test set (no retrain): shuffle actions/features
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(te))
    # plus_shuffled: shuffle action one-hot on test
    Xp_shuf = np.concatenate([Z[te], Aoh[te][perm]], axis=1)
    Yps = m_plus.predict(Xp_shuf) * y_std + y_mean
    out_rows.append(dict(split='test', variant='plus_shuffled', action='all',
                         r2_macro=r2_macro_stable(V[te], Yps),
                         cosine_macro=cosine_macro(V[te], Yps),
                         mse=mse_mae(V[te], Yps)[0], mae=mse_mae(V[te], Yps)[1]))
    # base_shuffled: shuffle features on test
    Xb_shuf = Z[te][perm]
    Ybs = m_base.predict(Xb_shuf) * y_std + y_mean
    out_rows.append(dict(split='test', variant='base_shuffled', action='all',
                         r2_macro=r2_macro_stable(V[te], Ybs),
                         cosine_macro=cosine_macro(V[te], Ybs),
                         mse=mse_mae(V[te], Ybs)[0], mae=mse_mae(V[te], Ybs)[1]))

    # plus_train_shuffled: shuffle action labels on train for plus model
    tr_perm = rng.permutation(len(tr))
    Xp_tr_shuf = np.concatenate([Z[tr], Aoh[tr][tr_perm]], axis=1)
    m_plus_shuf = probe().fit(Xp_tr_shuf, Vz_tr)
    Yps_trainshuf = m_plus_shuf.predict(X_plus[te]) * y_std + y_mean
    out_rows.append(dict(split='test', variant='plus_train_shuffled', action='all',
                         r2_macro=r2_macro_stable(V[te], Yps_trainshuf),
                         cosine_macro=cosine_macro(V[te], Yps_trainshuf),
                         mse=mse_mae(V[te], Yps_trainshuf)[0], mae=mse_mae(V[te], Yps_trainshuf)[1]))

    # per-action fits (condition)
    for a in range(K):
        msk_tr = (A[tr] == a)
        msk_te = (A[te] == a)
        if msk_tr.sum() < 10 or msk_te.sum() < 10:
            continue
        mb = probe().fit(X_base[tr][msk_tr], V[tr][msk_tr])
        mp = probe().fit(X_plus[tr][msk_tr], V[tr][msk_tr])
        ma = probe().fit(X_act[tr][msk_tr],  V[tr][msk_tr])
        for name, model in [('base', mb), ('plus', mp), ('action_only', ma)]:
            X_sel = {'base': X_base, 'plus': X_plus, 'action_only': X_act}[name][te][msk_te]
            Yhat = model.predict(X_sel) * y_std + y_mean
            r2 = r2_macro_stable(V[te][msk_te], Yhat)
            cos = cosine_macro(V[te][msk_te], Yhat)
            mse, mae = mse_mae(V[te][msk_te], Yhat)
            t = topk_metrics(V[te][msk_te], Yhat, ks=(3, 5))
            row = dict(split='test', variant=name, action=int(a), r2_macro=r2, cosine_macro=cos, mse=mse, mae=mae)
            row.update(t)
            out_rows.append(row)

    # Report deltas under action balancing (cosine)
    def get_metric(rows, variant_key):
        vals = [r['cosine_macro'] for r in rows if r.get('variant') == variant_key and r.get('action') == 'macro']
        return float(vals[0]) if vals else float('nan')
    plus_ab = get_metric(out_rows, 'plus_ab')
    base_ab = get_metric(out_rows, 'base_ab')
    if not (np.isnan(plus_ab) or np.isnan(base_ab)):
        out_rows.append(dict(split='test', variant='delta_ab', action='macro', cosine_macro=plus_ab - base_ab))

    # Optional branch selection using branched dataset
    if args.branched:
        BD = np.load(args.branched, allow_pickle=True)
        F_key = args.branched_feature if args.branched_feature else ('feature_t' if 'feature_t' in BD.files else 'feature')
        F = BD[F_key]
        A_b = np.asarray(BD['action']).astype(int)
        BIDs = BD['base_id'] if 'base_id' in BD.files else BD['source_id']
        # derive next absolute cell index per row (from agent_pos_next or obs_next_raw)
        if 'agent_pos_next' in BD.files:
            PosN = np.asarray(BD['agent_pos_next'])
            y_onehot = np.zeros((len(PosN), args.grid * args.grid), dtype=np.float32)
            y_onehot[np.arange(len(PosN)), (PosN[:, 0] * args.grid + PosN[:, 1]).astype(int)] = 1.0
        else:
            ORn = BD['obs_next_raw']
            ORn = np.asarray(ORn)
            ch_sums = ORn.reshape(len(ORn), -1, ORn.shape[-1]).sum(axis=1).mean(axis=0)
            agent_ch = int(np.argmin(ch_sums))
            idx = np.argmax(ORn[..., agent_ch].reshape(len(ORn), -1), axis=1)
            y_onehot = np.zeros((len(idx), args.grid * args.grid), dtype=np.float32)
            y_onehot[np.arange(len(idx)), idx.astype(int)] = 1.0
        # coerce features to 2D
        if isinstance(F, np.ndarray) and F.dtype == object:
            F = np.vstack([np.asarray(z).ravel() for z in F]).astype(np.float32)
        else:
            F = np.asarray(F)
            if F.ndim > 2:
                F = F.reshape(F.shape[0], -1)
            F = F.astype(np.float32)
        # build design matrices matching trained scalers' expected dims
        Aoh_b = onehot(A_b, K)
        # transform inputs through the fitted scalers from m_base/m_plus pipelines
        scaler_base = m_base.named_steps['standardscaler']
        scaler_plus = m_plus.named_steps['standardscaler']
        scaler_act  = m_act.named_steps['standardscaler']
        Xb_base_raw = F
        Xb_plus_raw = np.concatenate([F, Aoh_b], axis=1)
        Xb_act_raw  = Aoh_b
        Xb_base = scaler_base.transform(Xb_base_raw)
        Xb_plus = scaler_plus.transform(Xb_plus_raw)
        Xb_act  = scaler_act.transform(Xb_act_raw)
        # predict vectors
        Vhat_base = m_base.named_steps['ridge'].predict(Xb_base) * y_std + y_mean
        Vhat_plus = m_plus.named_steps['ridge'].predict(Xb_plus) * y_std + y_mean
        Vhat_act  = m_act.named_steps['ridge'].predict(Xb_act)   * y_std + y_mean
        # branch selection per base: pick action with highest cosine to true one-hot
        from collections import defaultdict
        idx_by_base = defaultdict(list)
        for i, b in enumerate(np.asarray(BIDs).astype(int)):
            idx_by_base[int(b)].append(i)
        def branch_acc(Vhat):
            wins = []
            for b, idxs in idx_by_base.items():
                if len(idxs) < 2:
                    continue
                # filter: require distinct next cells among branches
                next_idxs = [int(np.argmax(y_onehot[i])) for i in idxs]
                if len(set(next_idxs)) < 2:
                    continue
                scores = []
                for i in idxs:
                    true = y_onehot[i]
                    s = float(np.dot(Vhat[i], true) / (np.linalg.norm(Vhat[i]) + 1e-8))
                    scores.append(s)
                pick = idxs[int(np.argmax(scores))]
                # correctness: check if picked row corresponds to the row with max true next-cell index among branches
                wins.append(int(pick == idxs[np.argmax([np.argmax(y_onehot[i]) for i in idxs])]))
            return float(np.mean(wins)) if wins else float('nan')
        out_rows.append(dict(split='test', variant='branch_acc_base', action='all', branch_acc=branch_acc(Vhat_base)))
        out_rows.append(dict(split='test', variant='branch_acc_plus', action='all', branch_acc=branch_acc(Vhat_plus)))
        out_rows.append(dict(split='test', variant='branch_acc_action_only', action='all', branch_acc=branch_acc(Vhat_act)))

    # write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        if out_rows:
            keys = sorted(set().union(*[r.keys() for r in out_rows]))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(out_rows)
        else:
            w = csv.writer(f); w.writerow(['message']); w.writerow(['No rows'])
    print('Saved SR probe to', out)


if __name__ == '__main__':
    main()


