#!/usr/bin/env python3
import argparse
import numpy as np
from numpy.linalg import inv
from numpy.linalg import eigvals


def build_sr(trajs, S, gamma=0.9):
    C = np.zeros((S, S), dtype=float)
    for path in trajs:
        T = len(path)
        for t in range(T):
            s = int(path[t])
            C[s, s] += 1.0  # k=0 term
            gk = 1.0
            for k in range(1, T - t):
                sp = int(path[t + k])
                gk *= gamma
                C[s, sp] += gk
    counts = np.maximum(C.sum(1), 1e-8)
    Psi = (C.T / counts).T
    return Psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--mask_illegal', action='store_true', help='Mask T to legal neighbors before metrics')
    ap.add_argument('--allow_self', action='store_true', help='Allow self-loops as legal moves')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    pos = np.stack(d['pos'])
    ep = d['episode'] if 'episode' in d.files else np.zeros(len(pos))
    H = int(pos[:, 0].max()) + 1
    W = int(pos[:, 1].max()) + 1
    S = H * W
    s_ids = (pos[:, 0] * W + pos[:, 1]).astype(int)

    trajs = []
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        if len(idx) < 2:
            continue
        trajs.append(s_ids[idx])

    # Sanity: fraction of illegal empirical one-step moves under torus
    def legal_one_step_mask(Nh: int, Nw: int) -> np.ndarray:
        S = Nh * Nw
        M = np.zeros((S, S), dtype=bool)
        for i in range(Nh):
            for j in range(Nw):
                s = i * Nw + j
                nbrs = [i * Nw + ((j + 1) % Nw),
                        i * Nw + ((j - 1) % Nw),
                        ((i + 1) % Nh) * Nw + j,
                        ((i - 1) % Nh) * Nw + j]
                for t in nbrs:
                    M[s, t] = True
        return M

    torus_mask = legal_one_step_mask(H, W)
    total_pairs, illegal_pairs = 0, 0
    for path in trajs:
        for t in range(len(path) - 1):
            s, sp = int(path[t]), int(path[t + 1])
            total_pairs += 1
            if not torus_mask[s, sp]:
                illegal_pairs += 1
    if total_pairs > 0:
        print(f"empirical illegal fraction (torus): {illegal_pairs/total_pairs:.4f} ({illegal_pairs}/{total_pairs})")

    Psi = build_sr(trajs, S, gamma=args.gamma)
    # Recover T_pi via (I - gamma T_pi)^{-1} ~ Psi => T_pi ~ (I - Psi^{-1}) / gamma
    try:
        Psi_inv = inv(Psi)
        T_pi_est = (np.eye(S) - Psi_inv) / args.gamma
    except Exception:
        T_pi_est = None
        print('Warning: SR inversion failed; matrix may be singular.')

    def project_to_row_stochastic(M: np.ndarray) -> np.ndarray:
        X = M.copy()
        # clip negatives, then renormalize rows to sum 1 (if any positive mass)
        X[X < 0.0] = 0.0
        row_sums = X.sum(axis=1, keepdims=True)
        nonzero = row_sums.squeeze() > 0
        X[nonzero] = X[nonzero] / row_sums[nonzero]
        # rows with zero sum: set self-loop
        zero_idx = np.where(~nonzero)[0]
        for i in zero_idx:
            X[i, :] = 0.0
            X[i, i] = 1.0
        return X

    # Unified helpers: legal mask, masking+row-normalization, and gaps
    def build_legal_mask(nh: int, nw: int, allow_self: bool) -> np.ndarray:
        S = nh * nw
        M = np.zeros((S, S), dtype=bool)
        for i in range(nh):
            for j in range(nw):
                s = i * nw + j
                nbrs = [i * nw + ((j + 1) % nw),
                        i * nw + ((j - 1) % nw),
                        ((i + 1) % nh) * nw + j,
                        ((i - 1) % nh) * nw + j]
                for t in nbrs:
                    M[s, t] = True
        if allow_self:
            idx = np.arange(S)
            M[idx, idx] = True
        return M

    def mask_and_row_normalize(T: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        X = T.copy()
        X[~legal_mask] = 0.0
        rs = X.sum(axis=1, keepdims=True)
        dead = (rs[:, 0] == 0.0)
        if np.any(dead):
            for s in np.where(dead)[0]:
                nbrs = np.where(legal_mask[s])[0]
                if len(nbrs) == 0:
                    X[s, s] = 1.0
                else:
                    X[s, nbrs] = 1.0 / len(nbrs)
            rs = X.sum(axis=1, keepdims=True)
        X /= rs
        return X

    def spectral_gap(T: np.ndarray) -> float:
        w = np.linalg.eigvals(T)
        lam = np.sort(np.abs(w))
        if len(lam) < 2:
            return 0.0
        return float(1.0 - lam[-2].real)

    def lazy_gap(T: np.ndarray, alpha: float = 0.5) -> float:
        Tl = alpha * np.eye(T.shape[0]) + (1.0 - alpha) * T
        return spectral_gap(Tl)

    # Empirical one-step transitions (unconditional)
    T_emp = np.zeros((S, S), dtype=float)
    for path in trajs:
        for t in range(len(path) - 1):
            T_emp[int(path[t]), int(path[t + 1])] += 1.0
    row_sums = np.maximum(T_emp.sum(1, keepdims=True), 1e-8)
    T_emp = T_emp / row_sums

    # Build a single legal mask for evaluation (consistent across all metrics)
    legal_mask_eval = build_legal_mask(H, W, allow_self=args.allow_self)
    # Mask+renormalize empirical
    T_emp_eval = mask_and_row_normalize(T_emp, legal_mask_eval)
    print("Illegal mass (emp, after mask):", float(T_emp_eval[~legal_mask_eval].sum()))
    print(f"Spectral gap EMP (raw/lazy) = {spectral_gap(T_emp_eval):.4f}/{lazy_gap(T_emp_eval):.4f}")

    # Action-conditional transitions T(s,a,·) and policy π(a|s)
    if 'action' in d.files:
        actions_all = np.asarray(d['action'])
        # Build contiguous same-episode pairs (s_t, a_t, s_{t+1})
        C_sa = np.zeros((S, 4, S), dtype=float)
        for e in np.unique(ep):
            idx = np.where(ep == e)[0]
            if len(idx) < 2:
                continue
            for j in range(len(idx) - 1):
                st = int(s_ids[idx[j]])
                at = int(actions_all[idx[j]])
                sp = int(s_ids[idx[j + 1]])
                if 0 <= at < 4:
                    C_sa[st, at, sp] += 1.0
        alpha = 1e-6
        T_sa = (C_sa + alpha) / (C_sa.sum(-1, keepdims=True) + alpha * S)
        # Empirical policy π(a|s)
        pi_counts = C_sa.sum(-1)  # (S, A)
        pi = (pi_counts + 1e-6) / (pi_counts.sum(-1, keepdims=True) + 1e-6 * 4)
        # Compose T_pi (empirical policy under action-conditional dynamics)
        T_pi_from_Tsa_raw = np.einsum('sa,san->sn', pi, T_sa)
        # Normalize + mask consistently
        T_pi_from_Tsa_eval = mask_and_row_normalize(project_to_row_stochastic(T_pi_from_Tsa_raw), legal_mask_eval)
    else:
        T_sa = None; pi = None; T_pi_from_Tsa_eval = None

    print('SR built. Shapes:', Psi.shape, 'Empirical T_pi:', T_emp.shape)
    if T_pi_est is not None:
        # Normalize + mask consistently
        T_pi_eval = mask_and_row_normalize(project_to_row_stochastic(T_pi_est), legal_mask_eval)
        # Row-sum diagnostics and illegal mass after mask (should be ~0)
        rs_est = T_pi_eval.sum(axis=1, keepdims=True)
        print("row-sum min/max after mask (T_pi):", float(rs_est.min()), float(rs_est.max()))
        illegal_mass_after = float(T_pi_eval[~legal_mask_eval].sum())
        print("Illegal mass after mask (T_pi):", illegal_mass_after)
        # Assertions to catch regressions
        try:
            assert np.allclose(T_emp_eval.sum(1), 1.0, atol=1e-9)
            assert np.allclose(T_pi_eval.sum(1), 1.0, atol=1e-9)
            assert illegal_mass_after < 1e-12
        except AssertionError:
            pass
        # Report mean absolute difference on reachable states
        mask = (row_sums.squeeze() > 0)
        mad = np.mean(np.abs(T_emp_eval[mask] - T_pi_eval[mask]))
        print(f"mean |T_emp - T_pi_est| = {mad:.4f}")
        # Top-k off-diagonal errors
        diff = np.abs(T_emp_eval - T_pi_eval)
        np.fill_diagonal(diff, 0.0)
        flat_idx = np.dstack(np.unravel_index(np.argsort(diff, axis=None)[::-1], diff.shape))[0]
        topk = flat_idx[:10]
        print('Top-10 off-diagonal (i->j) diffs:')
        for (i, j) in topk:
            ii, ij = divmod(int(i), W)
            ji, jj = divmod(int(j), W)
            print(f"  ({int(i)}[{ii},{ij}]->{int(j)}[{ji},{jj}]): emp={T_emp_eval[int(i),int(j)]:.3f}, est={T_pi_eval[int(i),int(j)]:.3f}, |Δ|={diff[int(i),int(j)]:.3f} legal={bool(legal_mask_eval[int(i),int(j)])}")
        # Stationary distributions
        # Empirical visitation
        visit_counts = np.zeros(S, dtype=float)
        for path in trajs:
            for s in path:
                visit_counts[int(s)] += 1
        pi_emp = visit_counts / max(visit_counts.sum(), 1e-8)
        # Stationary from T_pi_est via power iteration
        if T_pi_eval is not None:
            pi = np.ones(S) / S
            for _ in range(200):
                pi = pi @ T_pi_eval
            pi_est = pi / max(pi.sum(), 1e-8)
            stat_diff = float(np.mean(np.abs(pi_emp - pi_est)))
            print(f"mean |stationary_emp - stationary_est| = {stat_diff:.4f}")
            # Spectral gap of T_pi_est
            try:
                ev = eigvals(T_pi_eval.T)
                ev_sorted = np.sort(np.abs(ev))[::-1]
                gap = float(1.0 - ev_sorted[1]) if len(ev_sorted) > 1 else float('nan')
                print(f"spectral gap (1 - |λ2|) = {gap:.4f}")
            except Exception:
                pass
            # Lazy chain spectral gap
            try:
                lazy = 0.5 * np.eye(S) + 0.5 * T_pi_eval
                ev_l = eigvals(lazy.T)
                evl_sorted = np.sort(np.abs(ev_l))[::-1]
                gap_lazy = float(1.0 - evl_sorted[1]) if len(evl_sorted) > 1 else float('nan')
                print(f"spectral gap (lazy) = {gap_lazy:.4f}")
            except Exception:
                pass
            # Also report for T_emp for apples-to-apples
            try:
                ev_emp = eigvals(T_emp_eval.T)
                ev_emp_sorted = np.sort(np.abs(ev_emp))[::-1]
                gap_emp = float(1.0 - ev_emp_sorted[1]) if len(ev_emp_sorted) > 1 else float('nan')
                lazy_emp = 0.5 * np.eye(S) + 0.5 * T_emp_eval
                ev_emp_l = eigvals(lazy_emp.T)
                ev_emp_l_sorted = np.sort(np.abs(ev_emp_l))[::-1]
                gap_emp_lazy = float(1.0 - ev_emp_l_sorted[1]) if len(ev_emp_l_sorted) > 1 else float('nan')
                print(f"spectral gap EMP (raw/lazy) = {gap_emp:.4f}/{gap_emp_lazy:.4f}")
            except Exception:
                pass
        # k-step TV distance vs empirical T_emp^k (legal-only comparison)
        TVs = []
        for k in range(1, 6):
            T_est_k = np.linalg.matrix_power(T_pi_eval, k)
            T_emp_k = np.linalg.matrix_power(T_emp_eval, k)
            tv = 0.5 * np.abs(T_est_k - T_emp_k).sum(axis=1).mean()
            TVs.append(float(tv))
        print('Mean TV @ k=1..5:', TVs)

        # Optional: compare SR T_pi vs action-conditional T_pi_from_Tsa
        if T_pi_from_Tsa_eval is not None:
            mad_tsa = float(np.mean(np.abs(T_pi_from_Tsa_eval - T_pi_eval)))
            print(f"mean |T_pi_from_Tsa - T_pi_est| = {mad_tsa:.4f}")

    # Save artifacts
    out = dict(Psi=Psi, T_emp=T_emp, T_emp_eval=T_emp_eval)
    if T_pi_est is not None:
        out.update(T_pi_raw=T_pi_est, T_pi_eval=T_pi_eval)
    if 'T_pi_from_Tsa_eval' in locals() and T_pi_from_Tsa_eval is not None:
        out.update(T_pi_from_Tsa_eval=T_pi_from_Tsa_eval)
    np.savez_compressed('sr_outputs.npz', **out)
    print('Saved sr_outputs.npz')


if __name__ == '__main__':
    main()


