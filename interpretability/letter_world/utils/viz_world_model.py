#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmx
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import warnings as _warnings
from pathlib import Path

_warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


def is_arr(x):
    return isinstance(x, np.ndarray)


def load_lds(path):
    q = np.load(path, allow_pickle=True)
    # backward-compat keys
    A1 = q['A1'] if 'A1' in q.files else q['A']
    A2 = q['A2'] if 'A2' in q.files else None
    B = q['B']; b = q['b']
    mu = q['mu'] if 'mu' in q.files else None
    sigma = q['sigma'] if 'sigma' in q.files else None
    order = int(q['order']) if 'order' in q.files else (2 if A2 is not None else 1)
    return A1, A2, B, b, mu, sigma, order


def scl(z, mu, sigma):
    if mu is None or sigma is None:
        return z
    sigma_safe = np.maximum(sigma, 1e-8)
    return (z - mu) / sigma_safe


def roll_one_step_scaled(z_cur_s, z_prev_s, a, A1, A2, B, b, order):
    u = np.eye(B.shape[1])[int(a)]
    if order == 2 and A2 is not None and z_prev_s is not None:
        z_next_s = A1 @ z_cur_s + A2 @ z_prev_s + B @ u + b
        return z_next_s, z_cur_s
    else:
        z_next_s = A1 @ z_cur_s + B @ u + b
        return z_next_s, None


def roll_k_scaled(z0_s, a_seq, A1, A2, B, b, order, zprev0_s=None):
    z_s = z0_s.copy()
    zp_s = zprev0_s.copy() if zprev0_s is not None else None
    traj = [z_s.copy()]
    for a in a_seq:
        z_s, zp_s = roll_one_step_scaled(z_s, zp_s, a, A1, A2, B, b, order)
        traj.append(z_s.copy())
    return np.stack(traj, 0)


# ---- Field helpers in scaled space (then project with PCA) ----
def step_total(zs, a, A1, B, b):
    u = np.eye(B.shape[1])[int(a)]
    return (A1 @ zs + B @ u + b) - zs


def step_drift(zs, A1, b):
    return ((A1 - np.eye(A1.shape[0])) @ zs + b)


def step_ctrl(a, B):
    u = np.eye(B.shape[1])[int(a)]
    return B @ u


def to_pca2(vec_s, pca):
    return vec_s @ pca.components_.T


def make_grid_around(points2, nx=25, ny=25, pad=0.2):
    xlo, xhi = points2[:, 0].min(), points2[:, 0].max()
    ylo, yhi = points2[:, 1].min(), points2[:, 1].max()
    dx, dy = xhi - xlo, yhi - ylo
    x = np.linspace(xlo - pad * max(dx, 1e-6), xhi + pad * max(dx, 1e-6), nx)
    y = np.linspace(ylo - pad * max(dy, 1e-6), yhi + pad * max(dy, 1e-6), ny)
    X, Y = np.meshgrid(x, y)
    grid2 = np.stack([X.ravel(), Y.ravel()], 1)
    return X, Y, grid2


def normalize(UV2, scale=0.18):
    n = np.linalg.norm(UV2, axis=1, keepdims=True) + 1e-12
    return (UV2 / n) * scale, n.squeeze()


def pca_2d_fit(Z):
    p = PCA(n_components=2)
    Z2 = p.fit_transform(Z)
    return p, Z2


def neighbors_torus(p, n):
    i, j = int(p[0]), int(p[1])
    return [((i) % n, (j + 1) % n),
            ((i) % n, (j - 1) % n),
            ((i + 1) % n, (j) % n),
            ((i - 1) % n, (j) % n)]


def reachable_mask(n, p0, k):
    frontier = {tuple(p0)}
    for _ in range(k):
        nxt = set()
        for p in frontier:
            for q in neighbors_torus(p, n):
                nxt.add(q)
        frontier = nxt
    mask = np.zeros((n, n), dtype=bool)
    for i, j in frontier:
        mask[i, j] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--feature', default='actor_prelogits')
    ap.add_argument('--lds', required=True)
    ap.add_argument('--episode', type=int, default=None)
    ap.add_argument('--t0', type=int, default=None)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--viz', type=str, default='both', choices=['latent', 'grid', 'both'])
    ap.add_argument('--teacher_forced', action='store_true')
    ap.add_argument('--save', action='store_true')
    ap.add_argument('--out_dir', type=str, default='interpretability/letter_world/figs')
    ap.add_argument('--prefix', type=str, default='viz')
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Z_all = d[args.feature]
    A_all = d['action']
    E_all = d['episode'] if 'episode' in d.files else np.zeros(len(Z_all))
    POS_all = d['pos'] if 'pos' in d.files else None
    mask = np.array([is_arr(z) for z in Z_all])
    Z = np.stack(Z_all[mask])
    act = np.asarray(A_all[mask], dtype=int)
    eps = E_all[mask]
    pos = np.stack(POS_all[mask]) if POS_all is not None else None

    # Train same-step action decoder in scaled space
    A1, A2, B, b, mu, sigma, order = load_lds(args.lds)
    Z_s = scl(Z, mu, sigma)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    tr, te = next(gss.split(Z_s, groups=eps))
    dec = LogisticRegression(max_iter=2000)
    dec.fit(Z_s[tr], act[tr])

    # Drift vs control magnitudes and alignment diagnostics (scaled space, train split)
    try:
        drift_mat = (A1 - np.eye(A1.shape[0])) @ Z_s[tr].T + b[:, None]
        mean_drift = float(np.linalg.norm(drift_mat, axis=0).mean())
        ctrl_norms = [float(np.linalg.norm(B[:, i])) for i in range(B.shape[1])]
        drift_unit = drift_mat.T / (np.linalg.norm(drift_mat.T, axis=1, keepdims=True) + 1e-12)
        cos_list = []
        for i in range(B.shape[1]):
            bcol = B[:, i] / (np.linalg.norm(B[:, i]) + 1e-12)
            cos_list.append(float(np.mean(drift_unit @ bcol)))
        print(f"mean ||(A-I)z + b|| = {mean_drift:.3f} | mean ||B[:,i]|| = {float(np.mean(ctrl_norms)):.3f}")
        print(f"mean cos(drift, B[:,i]) per action: {np.round(np.array(cos_list),3)}")
    except Exception:
        pass

    # pick episode and t0 robustly (auto-select longest episode if needed)
    unique_eps = np.unique(eps)
    ep_lengths = {int(e): int((eps == e).sum()) for e in unique_eps}
    # default to requested or longest
    ep_id = int(args.episode) if args.episode is not None else int(max(ep_lengths, key=ep_lengths.get))
    ep_idx = np.where(eps == ep_id)[0]
    max_len = int(len(ep_idx))
    K_eff = int(args.k)
    if max_len < 3:
        print('No episode long enough to visualize.')
        return
    if max_len < (K_eff + 2):
        K_eff = max_len - 2
        print(f"Requested K={args.k} reduced to K={K_eff} for episode {ep_id} (len={max_len}).")
    # choose t0 so that t0+K_eff within episode
    if args.t0 is None:
        # place t0 roughly in first third to give room
        t0 = int(ep_idx[0] + max(0, min(max_len - (K_eff + 2), max_len // 3)))
    else:
        t0 = int(args.t0)
        if (t0 < ep_idx[0]) or (t0 + K_eff > ep_idx[-1]):
            t0 = int(ep_idx[0] + max(0, min(max_len - (K_eff + 2), max_len // 3)))

    # latent viz
    if args.viz in ('latent', 'both'):
        # PCA on a local slice for clarity
        lo = max(ep_idx.min(), t0 - 20)
        hi = min(ep_idx.max(), t0 + args.k + 20)
        Z_slice_s = Z_s[lo:hi + 1]
        pca, _ = pca_2d_fit(Z_slice_s)

        z0_s = Z_s[t0]
        a_seq = act[t0:t0 + args.k]
        zprev0_s = Z_s[t0 - 1] if (order == 2 and (t0 - 1) in ep_idx) else None
        Z_roll_s = roll_k_scaled(z0_s, a_seq, A1, A2, B, b, order, zprev0_s)
        Z_true_s = Z_s[t0:t0 + K_eff + 1]
        Zr2 = pca.transform(Z_roll_s)
        Zt2 = pca.transform(Z_true_s)

        # Color/time encoding, error vectors, step labels
        Kplot = Zt2.shape[0] - 1
        cmap_t = cmx.get_cmap('Blues')
        cmap_r = cmx.get_cmap('Oranges')
        plt.figure(figsize=(6, 6))
        for kk in range(Kplot):
            frac = kk / max(Kplot, 1)
            plt.plot(Zt2[kk:kk+2, 0], Zt2[kk:kk+2, 1], '-', color=cmap_t(frac), lw=3, alpha=0.9)
            plt.plot(Zr2[kk:kk+2, 0], Zr2[kk:kk+2, 1], '-', color=cmap_r(frac), lw=2, alpha=0.9)
            # error vector at step kk+1
            plt.arrow(Zt2[kk+1, 0], Zt2[kk+1, 1],
                      Zr2[kk+1, 0] - Zt2[kk+1, 0], Zr2[kk+1, 1] - Zt2[kk+1, 1],
                      length_includes_head=True, head_width=0.02, color='k', alpha=0.35)
            plt.text(Zt2[kk+1, 0], Zt2[kk+1, 1], f'{kk+1}', fontsize=8, color='#1f77b4')
        plt.scatter(Zt2[0, 0], Zt2[0, 1], s=100, marker='*', color='#1f77b4', label='start (true)')
        plt.scatter(Zr2[-1, 0], Zr2[-1, 1], s=80, marker='X', color='#ff7f0e', label=f'rolled @k={K_eff}')
        plt.title(f'Latent trajectories (ep={ep_id}, t0={t0}, K={K_eff})')
        plt.legend(); plt.axis('equal'); plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_latent_traj_ep{ep_id}_t{t0}_K{K_eff}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

        # Local vector fields in SAME PCA coords
        take = min(200, len(Z_slice_s))
        Z_ref = Z_slice_s[np.random.choice(len(Z_slice_s), size=take, replace=False)]
        Z2_ref = pca.transform(Z_ref)
        X, Y, Zgrid2 = make_grid_around(Z2_ref, nx=25, ny=25, pad=0.2)
        Zgrid_s = pca.inverse_transform(Zgrid2)

        # Total field per action
        plt.figure(figsize=(10, 5))
        for a in range(B.shape[1]):
            Vtot_s = np.vstack([step_total(z, a, A1, B, b) for z in Zgrid_s])
            Vtot2 = to_pca2(Vtot_s, pca)
            Vtot2_scaled, _ = normalize(Vtot2, scale=0.18)
            ax = plt.subplot(1, B.shape[1], a + 1)
            ax.quiver(X, Y, Vtot2_scaled[:, 0].reshape(X.shape), Vtot2_scaled[:, 1].reshape(Y.shape),
                      angles='xy', scale_units='xy', scale=1, width=0.003, pivot='mid')
            ax.set_title(f'Total (a={a})')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle('Total field $(A-I)z + b + B\,a$ (same PCA)')
        plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_vector_field_total_ep{ep_id}_t{t0}_K{K_eff}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

        # Drift-only and control-only fields (approximate)
        Vdrift_s = np.vstack([step_drift(z, A1, b) for z in Zgrid_s])
        Vd2 = to_pca2(Vdrift_s, pca)
        Vd2_scaled, _ = normalize(Vd2, scale=0.18)
        plt.figure(figsize=(4, 4))
        plt.quiver(X, Y, Vd2_scaled[:, 0].reshape(X.shape), Vd2_scaled[:, 1].reshape(Y.shape),
                   angles='xy', scale_units='xy', scale=1, width=0.003, pivot='mid')
        plt.title('Drift-only (A1 - I)z + b'); plt.axis('equal'); plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_drift_only_ep{ep_id}_t{t0}_K{K_eff}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 5))
        for a in range(B.shape[1]):
            Vc_s = np.repeat(step_ctrl(a, B)[None, :], len(Zgrid_s), axis=0)
            Vc2 = to_pca2(Vc_s, pca)
            Vc2_scaled, _ = normalize(Vc2, scale=0.18)
            ax = plt.subplot(1, B.shape[1], a + 1)
            ax.quiver(X, Y, Vc2_scaled[:, 0].reshape(X.shape), Vc2_scaled[:, 1].reshape(Y.shape),
                      angles='xy', scale_units='xy', scale=1, width=0.003, pivot='mid')
            ax.set_title(f'Control-only Ba (a={a})')
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_control_only_ep{ep_id}_t{t0}_K{K_eff}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

        # Decision regions in PCA plane
        Z_pred_actions = dec.predict(pca.inverse_transform(Zgrid2))
        plt.figure(figsize=(5, 5))
        plt.contourf(X, Y, Z_pred_actions.reshape(X.shape), alpha=0.25,
                     levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap='tab10')
        plt.plot(Zt2[:, 0], Zt2[:, 1], 'o-', color='#1f77b4', lw=3)
        plt.plot(Zr2[:, 0], Zr2[:, 1], 'o--', color='#ff7f0e', lw=2)
        plt.title('Action decision regions + trajectories'); plt.axis('equal'); plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_decision_regions_ep{ep_id}_t{t0}_K{K_eff}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()

    # grid overlay viz
    if args.viz in ('grid', 'both') and pos is not None:
        n = int(max(pos[:, 0].max(), pos[:, 1].max())) + 1
        p0 = tuple(pos[t0])
        z_cur_s = Z_s[t0].copy()
        z_prev_s = Z_s[t0 - 1].copy() if (order == 2 and (t0 - 1) in ep_idx) else None
        pos_pred = [p0]
        for s in range(K_eff):
            a_hat = int(act[t0 + s]) if args.teacher_forced else int(dec.predict(z_cur_s[None])[0])
            p0 = neighbors_torus(p0, n)[a_hat]
            pos_pred.append(p0)
            z_cur_s, z_prev_s = roll_one_step_scaled(z_cur_s, z_prev_s, a_hat, A1, A2, B, b, order)
        pos_true = [tuple(p) for p in pos[t0:t0 + K_eff + 1]]

        plt.figure(figsize=(4, 4))
        # exact-k rings
        def exact_k_ring(nside, start, ksteps):
            if ksteps == 0:
                m = np.zeros((nside, nside), bool); m[start] = True; return m
            prev = reachable_mask(nside, start, ksteps - 1)
            cur = reachable_mask(nside, start, ksteps)
            return np.logical_and(cur, ~prev)
        for kk in range(1, K_eff + 1):
            ring = exact_k_ring(n, pos_true[0], kk)
            plt.contour(ring.T, levels=[0.5], colors='lightgrey', linewidths=0.8, alpha=0.6)
        xs_t, ys_t = zip(*pos_true)
        xs_p, ys_p = zip(*pos_pred)
        plt.plot(xs_t, ys_t, 'o-', lw=2, label='actual', color='#1f77b4')
        plt.plot(xs_p, ys_p, 'x--', lw=2, label=('rolled+decoded' if not args.teacher_forced else 'rolled+TF'), color='#ff7f0e')
        # annotate steps and (pred, true) actions
        pred_actions = []
        z_tmp = Z_s[t0].copy(); z_prev_tmp = Z_s[t0 - 1].copy() if (order == 2 and (t0 - 1) in ep_idx) else None
        for s in range(K_eff):
            a_hat = int(act[t0 + s]) if args.teacher_forced else int(dec.predict(z_tmp[None])[0])
            pred_actions.append(a_hat)
            z_tmp, z_prev_tmp = roll_one_step_scaled(z_tmp, z_prev_tmp, a_hat, A1, A2, B, b, order)
        for s, (xp, yp) in enumerate(pos_pred):
            plt.text(xp, yp, str(s), fontsize=8)
        for s in range(1, len(pos_pred)):
            tp_idx = t0 + s
            a_hat = pred_actions[s - 1] if s - 1 < len(pred_actions) else -1
            a_true = int(act[tp_idx - 1])
            plt.text(xs_p[s], ys_p[s], f"({a_hat}/{a_true})", fontsize=7, color='gray')
        # grid metrics: mean L1, % action matches, first divergence
        true_actions_seq = list(act[t0:t0 + K_eff])
        matches = [int(ah == at) for ah, at in zip(pred_actions, true_actions_seq)]
        acc_actions = float(np.mean(matches)) if matches else float('nan')
        first_diff = next((i + 1 for i, (ah, at) in enumerate(zip(pred_actions, true_actions_seq)) if ah != at), None)
        l1 = [abs(px - x) + abs(py - y) for (px, py), (x, y) in zip(pos_pred, pos_true)]
        title_suffix = f"mean L1={np.mean(l1):.2f} | act_acc={acc_actions:.2f}"
        if first_diff is not None:
            title_suffix += f" | first_diff={first_diff}"
        plt.title(f'Episode {ep_id}, t0={t0}, K={K_eff} | {title_suffix}')
        plt.scatter([pos_true[0][0]], [pos_true[0][1]], c='gold', s=80, marker='*', label='start')
        l1 = [abs(px - x) + abs(py - y) for (px, py), (x, y) in zip(pos_pred, pos_true)]
        plt.title(f'Episode {ep_id}, t0={t0}, K={K_eff} | mean L1={np.mean(l1):.2f}')
        plt.legend(loc='upper right'); plt.xlim(-0.5, n - 0.5); plt.ylim(-0.5, n - 0.5)
        plt.gca().set_aspect('equal'); plt.grid(True, ls=':', alpha=0.3)
        plt.tight_layout()
        if args.save:
            out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{args.prefix}_grid_ep{ep_id}_t{t0}_K{K_eff}{'_TF' if args.teacher_forced else ''}.png"
            plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()


