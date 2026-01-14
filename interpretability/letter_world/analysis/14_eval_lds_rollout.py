#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings as _warnings
_warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
from sklearn.model_selection import GroupShuffleSplit


def is_arr(x):
    return isinstance(x, np.ndarray)


def load_lds(path):
    q = np.load(path, allow_pickle=True)
    A = q['A']; B = q['B']; b = q['b']
    A2 = q['A2'] if 'A2' in q.files else None
    mu = q['mu'] if 'mu' in q.files else None
    sigma = q['sigma'] if 'sigma' in q.files else None
    return A, B, b, A2, mu, sigma


def step(A, B, b, z, u_onehot):
    return (A @ z + B @ u_onehot + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--lds', required=True)
    ap.add_argument('--kmax', type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = d['action']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(Z_all))
    mask = np.array([is_arr(z) for z in Z_all])
    Z = np.stack(Z_all[mask])
    A = np.asarray(A_all[mask], dtype=int)
    E = E_all[mask]

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z, groups=E))

    # Load LDS and scale features if scaler provided; decoder is trained in this scaled space
    A_est, B_est, b_est, A2_est, mu, sigma = load_lds(args.lds)
    if mu is not None and sigma is not None:
        sigma_safe = np.maximum(sigma, 1e-8)
        Z_s = (Z - mu) / sigma_safe
    else:
        Z_s = Z

    # decoder: same-step action from scaled feature space
    dec = LogisticRegression(max_iter=2000)
    dec.fit(Z_s[tr], A[tr])

    # extra decoders (same-step, scaled space)
    # pos id (grid cell)
    if 'pos' in d.files:
        POS = d['pos']
        H = int(np.max(POS[:, 0])) + 1
        W = int(np.max(POS[:, 1])) + 1
        pos_id = (POS[:, 0] * W + POS[:, 1]).astype(int)
        pos_id = pos_id[mask]
        pos_dec = LogisticRegression(max_iter=2000)
        pos_dec.fit(Z_s[tr], pos_id[tr])
    else:
        pos_dec = None

    # grid size from obs_raw for reachable baseline (fallback to H)
    grid_n = None
    try:
        if 'obs_raw' in d.files:
            for f in d['obs_raw']:
                if isinstance(f, np.ndarray):
                    grid_n = int(f.shape[0]); break
    except Exception:
        pass
    if grid_n is None and 'pos' in d.files:
        grid_n = int(max(H, W))

    # on_letter label (from obs_raw)
    def build_on_letter_labels_local(dataset):
        feats = dataset['obs_raw']
        yloc = []
        for f in feats:
            if not isinstance(f, np.ndarray):
                yloc.append(None); continue
            Hh, Ww, Cc = f.shape
            flat = f.reshape(-1, Cc)
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
    try:
        onl_all = build_on_letter_labels_local(d)[mask].astype(float)
        onl_ok = np.array([v in (0, 1) for v in onl_all])
        onl_dec = LogisticRegression(max_iter=2000)
        onl_dec.fit(Z_s[tr & onl_ok[tr]], onl_all[tr & onl_ok[tr]].astype(int))
    except Exception:
        onl_dec = None

    # q (LDBA state)
    if 'q' in d.files:
        q_all = np.asarray(d['q'])[mask].astype(int)
        q_dec = LogisticRegression(max_iter=2000)
        q_dec.fit(Z_s[tr], q_all[tr])
    else:
        q_dec = None

    # value regression
    if 'value' in d.files:
        val_all = np.asarray(d['value'])[mask].astype(float)
        from sklearn.linear_model import LinearRegression
        val_dec = LinearRegression()
        val_dec.fit(Z_s[tr], val_all[tr])
    else:
        val_dec = None

    K = int(A.max()) + 1

    # State-only baseline (upper bound for no rollout)
    state_only = float(dec.score(Z_s[te], A[te]))
    print(f"STATE-ONLY (no rollout) acc={state_only:.3f}")

    # B column norms and pairwise cosine similarity (action input identifiability)
    from numpy.linalg import norm
    import itertools as _it
    def _cos(u, v):
        n = norm(u) * norm(v)
        return float(u @ v / n) if n > 0 else 0.0
    b_norms = [float(norm(B_est[:, i])) for i in range(B_est.shape[1])]
    pairs = list(_it.combinations(range(B_est.shape[1]), 2))
    sims = [_cos(B_est[:, i], B_est[:, j]) for (i, j) in pairs] if pairs else []
    if sims:
        print(f"B column norms: {b_norms}")
        print(f"mean cos(B[:,i],B[:,j])={float(np.mean(sims)):.3f} (min={float(np.min(sims)):.3f}, max={float(np.max(sims)):.3f})")

    for k in range(1, args.kmax + 1):
        idx_t, idx_tp = [], []
        for e in np.unique(E[te]):
            ep = np.where(E == e)[0]
            ep_te = np.intersect1d(ep, te)
            if len(ep_te) <= k:
                continue
            # enforce contiguity windows strictly inside this episode
            for j in range(len(ep_te) - k):
                t0 = int(ep_te[j]); tp0 = int(ep_te[j + k])
                if tp0 - t0 != k:
                    continue
                idx_t.append(t0); idx_tp.append(tp0)
        if not idx_t:
            print(f"k={k}: no valid pairs")
            continue
        t = np.asarray(idx_t, dtype=int)
        tp = np.asarray(idx_tp, dtype=int)

        # Leak checks and baselines
        # 1) match_prev baseline on targets
        prev_mask = [i for i in range(len(tp)) if (tp[i]-1) >= 0 and E[tp[i]-1] == E[tp[i]]]
        if prev_mask:
            match_prev_baseline = float(np.mean([A[tp[i]-1] == A[tp[i]] for i in prev_mask]))
        else:
            match_prev_baseline = float('nan')
        print(f"k={k}: match_prev_baseline (a_(t+k-1)==a_(t+k))={match_prev_baseline:.3f}")
        # 2) target action distribution
        from collections import Counter
        counts = Counter(A[tp])
        print(f"k={k}: target action distribution: {dict(counts)}")
        # 3) median run-length around target
        def runlen_center(i_idx: int) -> int:
            i = int(i_idx)
            a0 = A[i]; L = 1
            j = i - 1
            while j >= 0 and E[j] == E[i] and A[j] == a0:
                L += 1; j -= 1
            j = i + 1
            while j < len(A) and E[j] == E[i] and A[j] == a0:
                L += 1; j += 1
            return L
        runs = [runlen_center(x) for x in tp]
        try:
            import numpy as _np
            med_run = float(_np.median(runs))
        except Exception:
            med_run = float('nan')
        print(f"k={k}: median run-length at target: {med_run:.1f}")

        hit = []
        match_prev_list = []
        match_next_list = []
        ok_idx = []
        rolled_zs = []
        for i in range(len(t)):
            ti, tpi = int(t[i]), int(tp[i])
            # For VARX(2), require t-1 exists and is same episode
            if A2_est is not None:
                if ti - 1 < 0 or E[ti - 1] != E[ti]:
                    continue
                z_prev_s = Z_s[ti - 1].copy()
            zhat_s = Z_s[ti].copy()
            ok = True
            for s in range(k):
                if ti + s >= len(Z_s) or E[ti + s] != E[ti]:
                    ok = False; break
                ai = A[ti + s]
                u = np.eye(K)[ai]
                if A2_est is not None:
                    z_next_s = (A_est @ zhat_s + A2_est @ z_prev_s + B_est @ u + b_est)
                    z_prev_s = zhat_s
                    zhat_s = z_next_s
                else:
                    zhat_s = (A_est @ zhat_s + B_est @ u + b_est)
            if not ok:
                continue
            a_pred = dec.predict(zhat_s.reshape(1, -1))[0]
            hit.append(int(a_pred == A[tpi]))
            ok_idx.append(i)
            rolled_zs.append(zhat_s.copy())
            # triage: compare to next vs previous action
            prev_ok = (tpi - 1) >= 0 and E[tpi - 1] == E[tpi]
            if prev_ok:
                prev = A[tpi - 1]
                match_prev_list.append(int(a_pred == prev))
            match_next_list.append(int(a_pred == A[tpi]))
        if not hit:
            print(f"k={k}: no valid pairs")
            continue
        acc = float(np.mean(hit))
        # AR(1) baseline on same pairs (tp-1 vs tp)
        ar_pairs = [i for i in range(len(t)) if (t[i] < tp[i]) and (tp[i]-1) >= 0 and E[tp[i]-1] == E[tp[i]]]
        if ar_pairs:
            ar1 = float(np.mean([A[tp[i]-1] == A[tp[i]] for i in ar_pairs]))
        else:
            ar1 = float('nan')
        mp = np.mean(match_prev_list) if match_prev_list else float('nan')
        mn = np.mean(match_next_list) if match_next_list else float('nan')
        print(f"k={k}: n={len(hit)} acc={acc:.3f} | AR(1)={ar1:.3f} | match_prev={mp:.3f} match_next={mn:.3f}")

        # No-dynamics ablation: z_{t+1} = b only (ignore inputs and state)
        acc_nd = []
        for i in range(len(t)):
            ti, tpi = int(t[i]), int(tp[i])
            z_cur = b_est.copy()
            for _ in range(k-1):  # already at t+1 after first b
                z_cur = b_est.copy()
            a_pred = dec.predict(z_cur.reshape(1, -1))[0]
            acc_nd.append(int(a_pred == A[tpi]))
        print(f"k={k}: NO-DYNAMICS acc={float(np.mean(acc_nd)):.3f}")

        # Action-history baseline: predict a_{t+k} from [a_t..a_{t+k-1}]
        AH = np.zeros((len(t), K*k), dtype=float)
        for i in range(len(t)):
            ti = int(t[i])
            for s in range(k):
                ai = A[ti + s]
                AH[i, s*K + int(ai)] = 1.0
        # simple split inside test set
        split = max(1, int(0.5 * len(AH)))
        clf_hist = LogisticRegression(max_iter=2000)
        clf_hist.fit(AH[:split], A[tp[:split]])
        ah_acc = float(clf_hist.score(AH[split:], A[tp[split:]])) if len(AH) - split > 0 else float('nan')
        print(f"k={k}: ACTION-HISTORY baseline acc={ah_acc:.3f}")

        # Zig-zag slice: at least one action change in inputs
        def has_switch(ti: int, kk: int) -> bool:
            for s in range(1, kk):
                if A[ti + s] != A[ti + s - 1]:
                    return True
            return False
        zig_mask = np.array([has_switch(int(t[i]), k) for i in ok_idx])
        if zig_mask.any():
            acc_zig = float(np.mean(np.array(hit)[zig_mask]))
            print(f"k={k}: zigzag n={int(zig_mask.sum())} acc={acc_zig:.3f}")

        # 1) SHIFTED(+1) control using rolled z_k
        if ok_idx:
            t_ok = t[ok_idx]
            tp_ok = tp[ok_idx]
            # find valid tp+1 within same episode
            tp_p1 = []
            mask_p1 = []
            for tpi in tp_ok:
                if (tpi + 1) < len(A) and E[tpi + 1] == E[tpi]:
                    tp_p1.append(tpi + 1)
                    mask_p1.append(True)
                else:
                    tp_p1.append(-1)
                    mask_p1.append(False)
            mask_p1 = np.array(mask_p1, dtype=bool)
            if mask_p1.any():
                rolled_arr = np.array(rolled_zs, dtype=object)[mask_p1]
                y_shift = A[np.array(tp_p1)[mask_p1]]
                shift_pred = dec.predict(np.stack(rolled_arr))
                shift_acc = float(np.mean(shift_pred == y_shift))
                print(f"k={k}: SHIFTED(+1) acc={shift_acc:.3f}")

        # 2) REVERSED-ACTIONS rollout
        rev_hits = []
        for i in range(len(ok_idx)):
            ti, tpi = int(t[ok_idx[i]]), int(tp[ok_idx[i]])
            if A2_est is not None and not (ti - 1 >= 0 and E[ti - 1] == E[ti]):
                continue
            z_cur = Z_s[ti].copy()
            z_prev = Z_s[ti - 1].copy() if A2_est is not None else None
            for s in range(k):
                a = A[ti + (k - 1 - s)]
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ np.eye(K)[a] + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = (A_est @ z_cur + B_est @ np.eye(K)[a] + b_est)
            rev_hits.append(int(dec.predict(z_cur[None])[0] == A[tpi]))
        if rev_hits:
            print(f"k={k}: REVERSED-ACTIONS acc={float(np.mean(rev_hits)):.3f}")

        # 3) CROSS-EPISODE-ACTIONS rollout
        rng = np.random.default_rng(0)
        cross_hits = []
        test_eps = np.unique(E[te])
        for i in range(len(ok_idx)):
            ti, tpi = int(t[ok_idx[i]]), int(tp[ok_idx[i]])
            others = [e for e in test_eps if e != E[ti]]
            if not others:
                continue
            e2 = rng.choice(others)
            ep_idxs = np.where(E == e2)[0]
            if len(ep_idxs) <= k:
                continue
            # sample contiguous window
            # find valid start positions ensuring contiguity length k
            starts = []
            for j in range(len(ep_idxs) - k):
                if ep_idxs[j + k] - ep_idxs[j] == k:
                    starts.append(ep_idxs[j])
            if not starts:
                continue
            j0 = int(rng.choice(starts))
            acts = [A[j0 + s] for s in range(k)]
            if A2_est is not None and not (ti - 1 >= 0 and E[ti - 1] == E[ti]):
                continue
            z_cur = Z_s[ti].copy()
            z_prev = Z_s[ti - 1].copy() if A2_est is not None else None
            for a in acts:
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ np.eye(K)[a] + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = (A_est @ z_cur + B_est @ np.eye(K)[a] + b_est)
            cross_hits.append(int(dec.predict(z_cur[None])[0] == A[tpi]))
        if cross_hits:
            print(f"k={k}: CROSS-EPISODE-ACTIONS acc={float(np.mean(cross_hits)):.3f}")

        # BAG-OF-ACTIONS (shuffle order but same multiset)
        bag_hits = []
        rng2 = np.random.default_rng(123)
        for i in range(len(ok_idx)):
            ti, tpi = int(t[ok_idx[i]]), int(tp[ok_idx[i]])
            if A2_est is not None and not (ti - 1 >= 0 and E[ti - 1] == E[ti]):
                continue
            acts = [A[ti + s] for s in range(k)]
            rng2.shuffle(acts)
            z_cur = Z_s[ti].copy()
            z_prev = Z_s[ti - 1].copy() if A2_est is not None else None
            for a in acts:
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ np.eye(K)[a] + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = (A_est @ z_cur + B_est @ np.eye(K)[a] + b_est)
            bag_hits.append(int(dec.predict(z_cur[None])[0] == A[tpi]))
        if bag_hits:
            print(f"k={k}: BAG-OF-ACTIONS acc={float(np.mean(bag_hits)):.3f}")

        # B-scale gamma sweep
        def rollout_with_gamma(ti, kk, gamma):
            if A2_est is not None and not (ti - 1 >= 0 and E[ti - 1] == E[ti]):
                return None
            z_cur = Z_s[ti].copy()
            z_prev = Z_s[ti - 1].copy() if A2_est is not None else None
            for s in range(kk):
                a = A[ti + s]
                u = np.eye(K)[a]
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + gamma * (B_est @ u) + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = (A_est @ z_cur + gamma * (B_est @ u) + b_est)
            return z_cur
        for gamma in [0.0, 0.25, 0.5, 1.0, 2.0]:
            hits_g = []
            for i in range(len(ok_idx)):
                ti, tpi = int(t[ok_idx[i]]), int(tp[ok_idx[i]])
                zc = rollout_with_gamma(ti, k, gamma)
                if zc is None:
                    continue
                hits_g.append(int(dec.predict(zc[None])[0] == A[tpi]))
            if hits_g:
                print(f"k={k}: B-scale γ={gamma:.2f} acc={float(np.mean(hits_g)):.3f}")

        # 4) TRUE-Z acc on same pairs
        if ok_idx:
            y_true = A[tp_ok]
            true_pred = dec.predict(Z_s[tp_ok])
            print(f"k={k}: TRUE-Z acc={float(np.mean(true_pred == y_true)):.3f}")

        # 5) Distance-to-oracle in scaled space
        if ok_idx:
            from numpy.linalg import norm
            cs, l2 = [], []
            for i in range(len(ok_idx)):
                tp_i = int(tp_ok[i])
                zhat_i = rolled_zs[i]
                ztrue_i = Z_s[tp_i]
                cs.append(float(zhat_i @ ztrue_i / (norm(zhat_i) * norm(ztrue_i) + 1e-8)))
                l2.append(float(norm(zhat_i - ztrue_i)))
            print(f"k={k}: cos(zhat, z_true)={float(np.mean(cs)):.3f} | L2={float(np.mean(l2)):.3f}")

        # 6) ACTION-HISTORY baseline on zig-zag only
        if zig_mask.any():
            zig_idx = np.where(zig_mask)[0]
            AH_z = np.zeros((len(zig_idx), K * k), dtype=float)
            for j, ii in enumerate(zig_idx):
                ti = int(t_ok[ii])
                for s in range(k):
                    AH_z[j, s*K + int(A[ti + s])] = 1.0
            y_z = A[tp_ok[zig_idx]]
            if len(zig_idx) >= 4:
                split_z = len(zig_idx) // 2
                hid = LogisticRegression(max_iter=2000)
                hid.fit(AH_z[:split_z], y_z[:split_z])
                ah_acc_zig = float(hid.score(AH_z[split_z:], y_z[split_z:]))
                print(f"k={k}: ACTION-HISTORY (zigzag only) acc={ah_acc_zig:.3f}")

        # 7) CLOSED-LOOP rollout (optional)
        cl_hits = []
        for i in range(len(ok_idx)):
            ti, tpi = int(t[ok_idx[i]]), int(tp[ok_idx[i]])
            if A2_est is not None and not (ti - 1 >= 0 and E[ti - 1] == E[ti]):
                continue
            z_cur = Z_s[ti].copy()
            z_prev = Z_s[ti - 1].copy() if A2_est is not None else None
            for s in range(k):
                a_hat = dec.predict(z_cur[None])[0]
                if A2_est is not None:
                    z_next = (A_est @ z_cur + A2_est @ z_prev + B_est @ np.eye(K)[a_hat] + b_est)
                    z_prev, z_cur = z_cur, z_next
                else:
                    z_cur = (A_est @ z_cur + B_est @ np.eye(K)[a_hat] + b_est)
            cl_hits.append(int(dec.predict(z_cur[None])[0] == A[tpi]))
        if cl_hits:
            print(f"k={k}: CLOSED-LOOP acc={float(np.mean(cl_hits)):.3f}")

        # Extra target decodes from rolled z at tp
        if ok_idx:
            if pos_dec is not None:
                pred_pos = pos_dec.predict(np.stack(rolled_zs))
                true_pos = pos_id[tp_ok]
                pos_acc = float(np.mean(pred_pos == true_pos))
                print(f"k={k}: POS acc={pos_acc:.3f}")
                # reachable-set baseline on n x n torus
                def reachable_count_torus(n, kk):
                    cnt = 0
                    for dstep in range(kk, -1, -2):
                        if dstep == 0:
                            cnt += 1
                        else:
                            cnt += 4 * dstep
                    return min(cnt, n * n)
                if grid_n is not None and grid_n > 0:
                    rk = reachable_count_torus(grid_n, k)
                    base_pos = 1.0 / rk if rk > 0 else float('nan')
                    print(f"k={k}: POS reachable-baseline ≈ {base_pos:.3f} (n={grid_n}, |R_k|={rk})")
            if onl_dec is not None:
                pred_onl = onl_dec.predict(np.stack(rolled_zs))
                true_onl = onl_all[tp_ok].astype(int)
                print(f"k={k}: ON_LETTER acc={float(np.mean(pred_onl == true_onl)):.3f}")
            if q_dec is not None:
                # Q calibration: accuracy + Brier score + simple ECE with 10 bins
                q_proba = q_dec.predict_proba(np.stack(rolled_zs))
                pred_q = np.argmax(q_proba, axis=1)
                true_q = q_all[tp_ok]
                q_acc = float(np.mean(pred_q == true_q))
                # multi-class Brier score (one-vs-all)
                num_classes = q_proba.shape[1]
                Y_onehot = np.eye(num_classes)[true_q]
                brier = float(np.mean(np.sum((q_proba - Y_onehot) ** 2, axis=1)))
                # ECE
                conf = np.max(q_proba, axis=1)
                correct = (pred_q == true_q).astype(float)
                bins = np.linspace(0.0, 1.0, 11)
                ece = 0.0
                for bi in range(10):
                    m = (conf >= bins[bi]) & (conf < bins[bi + 1])
                    if m.any():
                        acc_bin = float(np.mean(correct[m]))
                        conf_bin = float(np.mean(conf[m]))
                        ece += (np.sum(m) / len(conf)) * abs(acc_bin - conf_bin)
                print(f"k={k}: Q acc={q_acc:.3f} | Brier={brier:.3f} | ECE={ece:.3f}")
                # q-change diagnostics
                t_ok = t[ok_idx]
                q_t = q_all[t_ok]
                change_mask = (true_q != q_t)
                p_change = float(np.mean(change_mask)) if len(change_mask) > 0 else float('nan')
                if change_mask.any():
                    acc_change = float(np.mean(pred_q[change_mask] == true_q[change_mask]))
                else:
                    acc_change = float('nan')
                if (~change_mask).any():
                    acc_same = float(np.mean(pred_q[~change_mask] == true_q[~change_mask]))
                else:
                    acc_same = float('nan')
                print(f"k={k}: Q change rate={p_change:.3f} | acc|change={acc_change:.3f} | acc|same={acc_same:.3f}")
            if val_dec is not None:
                pred_v = val_dec.predict(np.stack(rolled_zs))
                true_v = val_all[tp_ok]
                from sklearn.metrics import r2_score, mean_absolute_error
                r2v = float(r2_score(true_v, pred_v))
                mae_v = float(mean_absolute_error(true_v, pred_v))
                print(f"k={k}: VALUE R2={r2v:.3f} | MAE={mae_v:.3f}")


if __name__ == '__main__':
    main()


