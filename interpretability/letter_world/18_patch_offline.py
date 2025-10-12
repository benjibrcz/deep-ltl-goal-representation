#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def is_arr(x):
    return isinstance(x, np.ndarray)


def swap(z, z_src, alpha):
    return (1.0 - alpha) * z + alpha * z_src


def steer(z, d, alpha):
    return z + alpha * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--mode', choices=['swap', 'steer'], default='steer')
    ap.add_argument('--dir', type=str, default='PC:1', help='Direction spec (unused simple PC fallback)')
    ap.add_argument('--alpha_grid', type=str, default='0,0.25,0.5,1.0', help='Comma-separated or space-separated alphas')
    ap.add_argument('--match', choices=['none', 'pos', 'action'], default='pos')
    ap.add_argument('--target_action', type=int, default=None, help='If set, compute targeted flip rate to this action')
    ap.add_argument('--base_action', type=int, default=None, help='If set, restrict targeted evaluation to these base actions (A->B)')
    ap.add_argument('--lds', type=str, default=None, help='Optional LDS npz for rollouts (future)')
    ap.add_argument('--roll_k', type=int, default=0, help='If >0, attempt k-step rollout decoding (requires --lds)')
    ap.add_argument('--delta', type=float, default=0.5, help='Margin cushion (logit units) for per-example normalized step')
    ap.add_argument('--orthogonalize', action='store_true', help='Project direction to reduce collateral effects')
    ap.add_argument('--out', type=str, default='interpretability/letter_world/results/patch_offline.csv')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = d['action']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(A_all))

    mask = np.array([is_arr(z) for z in Z_all])
    idx_mask = np.where(mask)[0]
    Z = np.stack([Z_all[i] for i in idx_mask])
    A = np.asarray(A_all, int)[idx_mask]
    E = np.asarray(E_all)[idx_mask]

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z, groups=E))

    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, multi_class='multinomial'))
    clf.fit(Z[tr], A[tr])

    # Choose direction v
    Z_tr = Z[tr]
    mu = Z_tr.mean(0, keepdims=True)
    Zc = Z_tr - mu
    v = None
    w_unscaled = None; b_unscaled = None; norm2 = None
    if args.target_action is not None and args.base_action is not None:
        base = int(args.base_action); tgt = int(args.target_action)
        if args.feature == 'actor_prelogits':
            D = Z.shape[1]
            v = np.zeros(D, dtype=float)
            if 0 <= tgt < D:
                v[tgt] += 1.0
            if 0 <= base < D:
                v[base] -= 1.0
            w_unscaled = v.copy(); b_unscaled = 0.0; norm2 = float(np.dot(w_unscaled, w_unscaled))
        else:
            # derive from trained logistic regression weights; undo scaler
            try:
                scaler = clf.named_steps.get('standardscaler', None)
                logreg = clf.named_steps.get('logisticregression', None)
                if scaler is not None and logreg is not None and hasattr(logreg, 'coef_'):
                    W_std = logreg.coef_  # [K, D_std]
                    b_std = logreg.intercept_
                    scale = getattr(scaler, 'scale_', None)
                    mu = getattr(scaler, 'mean_', None)
                    w_std = W_std[tgt] - W_std[base]
                    if scale is not None:
                        w_unscaled = w_std / (scale + 1e-8)
                    else:
                        w_unscaled = w_std.copy()
                    # intercept transform: b_unscaled = (b_t - b_b) - w_std · (mu/scale)
                    b_unscaled = float((b_std[tgt] - b_std[base]) - np.sum(w_std * ((mu/scale) if (mu is not None and scale is not None) else 0.0)))
                    v = w_unscaled.copy()
                    norm2 = float(np.dot(w_unscaled, w_unscaled))
                    # Optional orthogonalization to reduce collateral
                    if args.orthogonalize:
                        C = []
                        K = W_std.shape[0]
                        for j in range(K):
                            if j not in (base, tgt):
                                c_std = W_std[j] - W_std[base]
                                C.append(c_std / (scale + 1e-8) if scale is not None else c_std)
                        if C:
                            C = np.stack(C, axis=0)  # [k, D]
                            # Orthonormal basis via SVD
                            U, _, _ = np.linalg.svd(C, full_matrices=False)
                            P_orth = np.eye(U.shape[1]) - U.T @ U
                            w_unscaled = (P_orth @ w_unscaled)
                            v = w_unscaled.copy()
                            norm2 = float(np.dot(w_unscaled, w_unscaled) + 1e-8)
            except Exception:
                v = None
    if v is None:
        # fallback to PC1 if no targeted direction was formed
        v = np.random.randn(Zc.shape[1]); v /= (np.linalg.norm(v) + 1e-8)
        for _ in range(10):
            v = (Zc.T @ (Zc @ v))
            v /= (np.linalg.norm(v) + 1e-8)

    # Support comma or space separated lists
    sep = ',' if ',' in args.alpha_grid else ' '
    alphas = [float(x) for x in args.alpha_grid.split(sep) if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # source indices (pos-matched optional)
    pos = d['pos'][mask]
    for alpha in alphas:
        Zp = Z[te].copy()
        if args.mode == 'steer':
            if w_unscaled is not None and norm2 is not None:
                # per-example normalized step using margin
                if args.feature == 'actor_prelogits':
                    # margin = z[tgt] - z[base]
                    m = Z[te][:, tgt] - Z[te][:, base]
                else:
                    m = Z[te] @ w_unscaled + b_unscaled
                s = (-m / (norm2 + 1e-8) + args.delta)  # [N]
                Zp = Z[te] + (alpha * s)[:, None] * w_unscaled[None, :]
            else:
                Zp = Zp + alpha * v
        else:
            # swap: naive random same-size source
            src_idx = np.random.permutation(len(Zp))
            Zp = (1.0 - alpha) * Zp + alpha * Z[te][src_idx]

        p_base = clf.predict_proba(Z[te])
        p_patch = clf.predict_proba(Zp)
        ce_base = log_loss(A[te], p_base, labels=clf[-1].classes_)
        ce_patch = log_loss(A[te], p_patch, labels=clf[-1].classes_)
        base_top = np.argmax(p_base, -1)
        patch_top = np.argmax(p_patch, -1)
        flips = (base_top != patch_top).mean()
        rec = dict(feature=args.feature, mode=args.mode, alpha=alpha, dce=float(ce_patch - ce_base), flip_rate=float(flips))
        # Targeted flip metrics (A->B or ->B)
        if args.target_action is not None:
            mask = np.ones_like(base_top, dtype=bool)
            if args.base_action is not None:
                mask &= (base_top == int(args.base_action))
            tf = ((patch_top == int(args.target_action)) & mask).mean()
            rec['targeted_flip_rate'] = float(tf)
        # Tags: on_letter (current), cell_type
        tags = {}
        if 'obs_raw' in d.files:
            obs_all = np.asarray(d['obs_raw'], dtype=object)[idx_mask]
            def is_on_letter(f):
                if not isinstance(f, np.ndarray): return None
                C = f.shape[-1]
                ch_sums = f.reshape(-1, C).sum(axis=0)
                agent_ch = int(np.where(ch_sums == 1)[0][0]) if (ch_sums == 1).any() else int(np.argmin(ch_sums))
                agent_mask = f[..., agent_ch] > 0.5
                letter_here = (np.sum(np.delete(f, agent_ch, axis=-1), axis=-1) > 0)
                return int((agent_mask & letter_here).any())
            tags['on_letter'] = np.array([is_on_letter(f) for f in obs_all[te]], dtype=object)
        if 'pos' in d.files:
            pos_te = np.asarray(d['pos'], dtype=object)[idx_mask][te]
            H = int(np.max(pos_te[:, 0])) + 1; W = int(np.max(pos_te[:, 1])) + 1
            def cell_type(p):
                if not isinstance(p, np.ndarray): return 'unknown'
                r, c = int(p[0]), int(p[1])
                is_edge = (r in (0, H-1)) or (c in (0, W-1))
                if is_edge:
                    corners = {(0,0),(0,W-1),(H-1,0),(H-1,W-1)}
                    return 'corner' if (r, c) in corners else 'edge'
                return 'center'
            tags['cell_type'] = np.array([cell_type(p) for p in pos_te], dtype=object)
        # Aggregate simple tag means for flips
        if tags:
            for key, vals in tags.items():
                try:
                    ok = np.array([v is not None for v in vals])
                    if ok.any():
                        # example: mean flip in on_letter==1 vs 0 not computed here; store counts and leave pivoting to downstream
                        rec[f'tag_{key}'] = 'present'
                except Exception:
                    pass
        rows.append(rec)

    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved offline patch results to {out_path}")


if __name__ == '__main__':
    main()


