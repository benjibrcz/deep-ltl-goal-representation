#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def load_seq(path: str, feature_key: str):
    D = np.load(path, allow_pickle=True)
    Z = D[feature_key]
    if isinstance(Z, np.ndarray) and Z.dtype == object:
        Z = np.vstack([np.asarray(z).ravel() for z in Z]).astype(np.float32)
    else:
        Z = np.asarray(Z)
        if Z.ndim > 2:
            Z = Z.reshape(Z.shape[0], -1)
        Z = Z.astype(np.float32)
    E = np.asarray(D['episode']) if 'episode' in D.files else np.zeros(len(Z))
    L = np.asarray(D['letter_id']) if 'letter_id' in D.files else None
    if L is None:
        raise SystemExit("letter_id missing; regenerate sequential rollouts with 03c_log_rollouts_seq.py")
    return Z, E, L


def find_first_hit(idx: np.ndarray, letters: np.ndarray, target: int, start_local: int = 0):
    for tloc in range(start_local, len(idx)):
        if int(letters[idx[tloc]]) == target:
            return tloc
    return None


def collect_phase_pairs_auto(Z: np.ndarray, E: np.ndarray, L: np.ndarray,
                             min_phase_len: int = 3, max_per_phase: int = 5000):
    Za, dZa, Ea = [], [], []
    Zb, dZb, Eb = [], [], []
    total_eps = 0
    kept_eps = 0
    # iterate episodes
    for e in np.unique(E):
        idx = np.where(E == e)[0]
        if len(idx) < 3:
            continue
        total_eps += 1
        # detect A then B automatically as first and second distinct letters encountered
        t_a = None; t_b = None; A_id = None; B_id = None
        for tloc in range(0, len(idx)):
            li = int(L[idx[tloc]])
            if li < 0:
                continue
            if t_a is None:
                t_a = tloc; A_id = li; continue
            if li != A_id:
                t_b = tloc; B_id = li; break
        if t_a is None or t_b is None:
            continue
        kept_eps += 1
        # build consecutive pairs strictly within episode: use idx[k] -> idx[k+1]
        pre_pairs = []
        for k in range(0, max(0, t_a)):
            if k + 1 >= len(idx):
                break
            i, j = idx[k], idx[k + 1]
            pre_pairs.append((i, j))

        post_pairs = []
        upper = min(t_b, len(idx) - 1)
        for k in range(t_a, upper):
            if k + 1 >= len(idx):
                break
            i, j = idx[k], idx[k + 1]
            post_pairs.append((i, j))

        # skip very short phases
        if len(pre_pairs) < min_phase_len or len(post_pairs) < min_phase_len:
            continue

        kept_eps += 1
        for i, j in pre_pairs:
            Za.append(Z[i]); dZa.append(Z[j] - Z[i]); Ea.append(e)
        for i, j in post_pairs:
            Zb.append(Z[i]); dZb.append(Z[j] - Z[i]); Eb.append(e)

    # cap sample counts for stability
    def cap(X, Y, G, cap_n):
        X, Y, G = np.asarray(X), np.asarray(Y), np.asarray(G)
        if len(X) <= cap_n:
            return X, Y, G
        rng = np.random.RandomState(0)
        sel = rng.choice(len(X), size=cap_n, replace=False)
        return X[sel], Y[sel], G[sel]

    Za, dZa, Ea = cap(Za, dZa, Ea, max_per_phase)
    Zb, dZb, Eb = cap(Zb, dZb, Eb, max_per_phase)

    stats = dict(total_eps=int(total_eps), kept_eps=int(kept_eps), pre_pairs=int(len(Za)), post_pairs=int(len(Zb)))
    return (np.array(Za), np.array(dZa), np.array(Ea)), (np.array(Zb), np.array(dZb), np.array(Eb)), stats


def collect_phase_pairs_auto_anchored(Z: np.ndarray, E: np.ndarray, L: np.ndarray,
                                      W_pre: int = 10, W_post: int = 10, max_per_phase: int = 5000):
    Za, dZa, Ea = [], [], []
    Zb, dZb, Eb = [], [], []
    total_eps = 0
    kept_eps = 0
    for e in np.unique(E):
        idx = np.where(E == e)[0]
        if len(idx) < 3:
            continue
        total_eps += 1
        # find first letter hit (any letter); do not require second distinct letter
        t_a = None
        for tloc in range(len(idx)):
            li = int(L[idx[tloc]])
            if li >= 0:
                t_a = tloc
                break
        if t_a is None:
            continue
        pre_start = max(0, t_a - W_pre)
        pre_end = max(0, t_a)
        post_start = t_a
        post_end = min(len(idx) - 1, t_a + W_post)
        pre_pairs = [(idx[k], idx[k + 1]) for k in range(pre_start, max(pre_end - 1, pre_start)) if k + 1 < len(idx)]
        post_pairs = [(idx[k], idx[k + 1]) for k in range(post_start, max(post_end, post_start)) if k + 1 < len(idx)]
        if len(pre_pairs) == 0 or len(post_pairs) == 0:
            continue
        kept_eps += 1
        for i, j in pre_pairs:
            Za.append(Z[i]); dZa.append(Z[j] - Z[i]); Ea.append(e)
        for i, j in post_pairs:
            Zb.append(Z[i]); dZb.append(Z[j] - Z[i]); Eb.append(e)

    rng = np.random.RandomState(0)
    def cap(X, Y, G, capn):
        X, Y, G = np.asarray(X), np.asarray(Y), np.asarray(G)
        if len(X) <= capn:
            return X, Y, G
        sel = rng.choice(len(X), size=capn, replace=False)
        return X[sel], Y[sel], G[sel]

    Za, dZa, Ea = cap(Za, dZa, Ea, max_per_phase)
    Zb, dZb, Eb = cap(Zb, dZb, Eb, max_per_phase)
    stats = dict(total_eps=int(total_eps), kept_eps=int(kept_eps), pre_pairs=int(len(Za)), post_pairs=int(len(Zb)))
    return (np.array(Za), np.array(dZa), np.array(Ea)), (np.array(Zb), np.array(dZb), np.array(Eb)), stats


def r2_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    var = y_true.var(axis=0)
    keep = var > 1e-8
    if not np.any(keep):
        return float('nan')
    yt, yp = y_true[:, keep], y_pred[:, keep]
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + eps
    return float(np.mean(1.0 - ss_res / ss_tot))


def mae_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(y_true - y_pred), axis=1)))


def cosine_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    a = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + eps)
    b = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + eps)
    return float(np.mean(np.sum(a * b, axis=1)))


def eval_block(model: Ridge, scaler: StandardScaler, Z_in: np.ndarray, dZ: np.ndarray):
    Zs = scaler.transform(Z_in)
    pred = model.predict(Zs)
    return dict(r2=r2_macro(dZ, pred), mae=mae_macro(dZ, pred), cos=cosine_macro(dZ, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='feature_t')
    # auto-detect A then B per episode; no explicit pair needed
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--min_phase_len', type=int, default=3)
    ap.add_argument('--anchored', action='store_true')
    ap.add_argument('--w_pre', type=int, default=10)
    ap.add_argument('--w_post', type=int, default=10)
    ap.add_argument('--permute_phase', action='store_true')
    ap.add_argument('--action_only', action='store_true', help='use one-hot action instead of z_t (not implemented here)')
    ap.add_argument('--template_only', action='store_true', help='Compute drift templates and cosine alignment; skip ridge fits')
    ap.add_argument('--balance', action='store_true', help='Downsample the larger phase to the smaller for template stats')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/phase_dynamics_ab.csv')
    args = ap.parse_args()

    Z, E, L = load_seq(args.data, args.feature)

    if args.anchored:
        (Za, dZa, Ea), (Zb, dZb, Eb), stats = collect_phase_pairs_auto_anchored(Z, E, L, W_pre=args.w_pre, W_post=args.w_post)
    else:
        (Za, dZa, Ea), (Zb, dZb, Eb), stats = collect_phase_pairs_auto(Z, E, L, min_phase_len=args.min_phase_len)
    print(f"Phase detection: episodes={stats['total_eps']} kept={stats['kept_eps']} | pre_pairs={stats['pre_pairs']} post_pairs={stats['post_pairs']}")
    # sanity prints
    if len(dZa) and len(dZb):
        print("pre mean |Δz|:", float(np.linalg.norm(dZa, axis=1).mean()))
        print("post mean |Δz|:", float(np.linalg.norm(dZb, axis=1).mean()))
    if (len(Za) < 50 or len(Zb) < 50) and not args.template_only:
        raise SystemExit('Too few samples in one of the phases; collect longer rollouts or adjust pair.')

    # optionally permute phase labels (control)
    if args.permute_phase:
        rng = np.random.RandomState(0)
        rng.shuffle(Za); rng.shuffle(dZa)
        rng.shuffle(Zb); rng.shuffle(dZb)

    # Template-only drift alignment path
    if args.template_only:
        out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        dZa_arr = np.asarray(dZa)
        dZb_arr = np.asarray(dZb)
        if args.balance and len(dZa_arr) > 0 and len(dZb_arr) > 0:
            n = min(len(dZa_arr), len(dZb_arr))
            rng = np.random.RandomState(0)
            sel_a = rng.choice(len(dZa_arr), size=n, replace=False)
            sel_b = rng.choice(len(dZb_arr), size=n, replace=False)
            dZa_arr = dZa_arr[sel_a]
            dZb_arr = dZb_arr[sel_b]
        eps = 1e-12
        def l2norm(X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X[None, :]
            nrm = np.linalg.norm(X, axis=1, keepdims=True) + eps
            return X / nrm
        if len(dZa_arr) == 0 or len(dZb_arr) == 0:
            print('Insufficient pairs for template stats.')
            with out.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['variant','n_pre','n_post'])
                w.writeheader()
                w.writerow(dict(variant='template', n_pre=int(len(dZa_arr)), n_post=int(len(dZb_arr))))
            print('Saved phase dynamics results to', out)
            return
        mu_pre = l2norm(dZa_arr.mean(axis=0, keepdims=True))[0]
        mu_post = l2norm(dZb_arr.mean(axis=0, keepdims=True))[0]
        A = l2norm(dZa_arr)
        B = l2norm(dZb_arr)
        cos_pre_pre = float((A @ mu_pre).mean())
        cos_pre_post = float((A @ mu_post).mean())
        cos_post_post = float((B @ mu_post).mean())
        cos_post_pre = float((B @ mu_pre).mean())
        align_pre = float(((A @ mu_pre) > (A @ mu_post)).mean())
        align_post = float(((B @ mu_post) > (B @ mu_pre)).mean())
        template_cos = float(mu_pre @ mu_post)
        # Proper null via phase-label shuffle: re-make templates from random split of pooled steps
        rng = np.random.RandomState(0)
        pool = np.vstack([A, B])
        nA, nB = len(A), len(B)
        null_align_pre = []
        null_align_post = []
        for _ in range(1000):
            sel = rng.permutation(len(pool))
            A_null = pool[sel[:nA]]
            B_null = pool[sel[nA:]]
            muA = l2norm(A_null.mean(axis=0, keepdims=True))[0]
            muB = l2norm(B_null.mean(axis=0, keepdims=True))[0]
            null_align_pre.append(float(((A @ muA) > (A @ muB)).mean()))
            null_align_post.append(float(((B @ muB) > (B @ muA)).mean()))
        align_pre_null_mean = float(np.mean(null_align_pre))
        align_post_null_mean = float(np.mean(null_align_post))
        # Template-swap sanity
        align_pre_swap = float(((A @ mu_post) > (A @ mu_pre)).mean())
        align_post_swap = float(((B @ mu_pre) > (B @ mu_post)).mean())
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['variant','n_pre','n_post','cos_pre_pre','cos_pre_post','cos_post_post','cos_post_pre','align_pre','align_post','align_pre_null_mean','align_post_null_mean','align_pre_swap','align_post_swap','template_cos'])
            w.writeheader()
            w.writerow(dict(variant='template', n_pre=int(len(dZa)), n_post=int(len(dZb)),
                            cos_pre_pre=cos_pre_pre, cos_pre_post=cos_pre_post,
                            cos_post_post=cos_post_post, cos_post_pre=cos_post_pre,
                            align_pre=align_pre, align_post=align_post,
                            align_pre_null_mean=align_pre_null_mean, align_post_null_mean=align_post_null_mean,
                            align_pre_swap=align_pre_swap, align_post_swap=align_post_swap,
                            template_cos=template_cos))
        print('Saved phase dynamics results to', out)
        return

    # split within each phase by episode groups
    def split(Zp, dZp, Gp):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        tr, te = next(gss.split(np.arange(len(Zp)), groups=Gp))
        return Zp[tr], Zp[te], dZp[tr], dZp[te]

    Za_tr, Za_te, dZa_tr, dZa_te = split(Za, dZa, Ea)
    Zb_tr, Zb_te, dZb_tr, dZb_te = split(Zb, dZb, Eb)

    # shared scaler across phases
    scaler = StandardScaler(with_mean=True).fit(np.vstack([Za_tr, Zb_tr]))
    Za_tr_s, Za_te_s = scaler.transform(Za_tr), scaler.transform(Za_te)
    Zb_tr_s, Zb_te_s = scaler.transform(Zb_tr), scaler.transform(Zb_te)

    # fit ridge Δz ~ z in each phase
    Ra = Ridge(alpha=args.alpha).fit(Za_tr_s, dZa_tr)
    Rb = Ridge(alpha=args.alpha).fit(Zb_tr_s, dZb_tr)

    # evaluate in-domain vs cross-domain
    pre_in = eval_block(Ra, scaler, Za_te, dZa_te)
    pre_x  = eval_block(Rb, scaler, Za_te, dZa_te)
    post_in= eval_block(Rb, scaler, Zb_te, dZb_te)
    post_x = eval_block(Ra, scaler, Zb_te, dZb_te)

    # write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['alpha','variant','phase','r2','mae','cos'])
        w.writeheader()
        w.writerow(dict(alpha=args.alpha, variant='in',  phase='pre',  r2=pre_in['r2'],  mae=pre_in['mae'],  cos=pre_in['cos']))
        w.writerow(dict(alpha=args.alpha, variant='cross',phase='pre',  r2=pre_x['r2'],   mae=pre_x['mae'],   cos=pre_x['cos']))
        w.writerow(dict(alpha=args.alpha, variant='in',  phase='post', r2=post_in['r2'], mae=post_in['mae'], cos=post_in['cos']))
        w.writerow(dict(alpha=args.alpha, variant='cross',phase='post', r2=post_x['r2'],  mae=post_x['mae'],  cos=post_x['cos']))
    print('Saved phase dynamics results to', out)


if __name__ == '__main__':
    main()


