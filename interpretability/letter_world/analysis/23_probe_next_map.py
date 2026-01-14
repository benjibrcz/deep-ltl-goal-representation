#!/usr/bin/env python3
import argparse
import numpy as np
import csv
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def build_targets(next_obs_raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    N, H, W, C = next_obs_raw.shape
    ch_sums = next_obs_raw.reshape(N, -1, C).sum(axis=1).mean(axis=0)
    agent_ch = int(np.argmin(ch_sums))
    letter_chs = [c for c in range(C) if c != agent_ch]
    Ys = []
    names = []
    for i in range(H):
        for j in range(W):
            for c in letter_chs:
                Ys.append((next_obs_raw[:, i, j, c] > 0.5).astype(int))
                names.append(f"cell({i},{j})_ch{c}")
    Y = np.stack(Ys, axis=1)
    return Y, names


def build_agent_next_targets(next_obs_raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    N, H, W, C = next_obs_raw.shape
    # detect agent channel as sparsest channel on average (one-hot agent)
    ch_sums = next_obs_raw.reshape(N, -1, C).sum(axis=1).mean(axis=0)
    agent_ch = int(np.argmin(ch_sums))
    # positions of agent in next frame
    pos = np.argmax(next_obs_raw[..., agent_ch].reshape(N, -1), axis=1)  # [N]
    # one-vs-rest over grid cells (H*W classes as independent binary targets)
    Y = np.zeros((N, H * W), dtype=int)
    Y[np.arange(N), pos] = 1
    names = [f"agent_next_cell_{i}_{j}" for i in range(H) for j in range(W)]
    return Y, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='branched_one_step.npz')
    ap.add_argument('--feature', default='feature')
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/next_map_auc.csv')
    ap.add_argument('--with_action', action='store_true')
    ap.add_argument('--target', type=str, default='map_next', choices=['map_next', 'agent_next', 'agent_next_offset'], help='Which target to probe')
    ap.add_argument('--permute_labels', action='store_true')
    ap.add_argument('--min_pos', type=int, default=1)
    ap.add_argument('--min_neg', type=int, default=1)
    ap.add_argument('--group_by_state', action='store_true', help='Use StratifiedGroupKFold by state_id if available')
    ap.add_argument('--max_targets', type=int, default=128)
    ap.add_argument('--egocentric', action='store_true')
    ap.add_argument('--only_local', type=int, default=0, help='World-centric: restrict to r-radius around next agent pos')
    ap.add_argument('--include_agent', action='store_true')
    ap.add_argument('--use_kronecker', action='store_true', help='Include x⊗a interactions to allow linear action-specific effects')
    ap.add_argument('--multi_only', action='store_true', help='Keep only bases with >=2 distinct actions')
    ap.add_argument('--actions_filter', type=int, default=None, help='If set, restrict to rows with this action only')
    args = ap.parse_args()

    # Suppress ROC AUC warnings for degenerate folds; we explicitly skip those cases
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Robust loader for branched dataset
    def load_branched(path):
        D = np.load(path, allow_pickle=True)
        # Feature at time t
        X = None
        for k in ["feature_t", "feature", "ego_now"]:
            if k in D.files:
                X = D[k]
                break
        if X is None:
            raise ValueError("No feature key found (tried: feature_t, feature, ego_now)")
        X = np.array(X, dtype=object)
        if X.dtype == object:
            X = np.vstack([np.asarray(r).ravel() for r in X]).astype(np.float32)
        else:
            X = np.asarray(X)
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1).astype(np.float32)
            else:
                X = X.astype(np.float32)
        # Action at time t
        A = None
        for k in ["action", "branched_action"]:
            if k in D.files:
                A = np.asarray(D[k]).astype(np.int64)
                break
        if A is None:
            raise ValueError("No action key found (tried: action, branched_action)")
        # Next map supervision
        Y_next = None
        for k in ["obs_next_raw", "next_obs_raw", "ego_next"]:
            if k in D.files:
                Y_next = np.asarray(D[k])
                break
        if Y_next is None:
            raise ValueError("No next map key found (tried: obs_next_raw, next_obs_raw, ego_next)")
        # base_id (optional; reconstruct if missing)
        if "base_id" in D.files:
            base_id = np.asarray(D["base_id"]) 
        elif "source_id" in D.files:
            base_id = np.asarray(D["source_id"]) 
        else:
            r = np.ascontiguousarray(np.round(X, 6))
            h = r.view(np.dtype((np.void, r.shape[1]*r.dtype.itemsize)))
            _, base_id = np.unique(h, return_inverse=True)
        # Align lengths
        N = min(len(X), len(A), len(Y_next), len(base_id))
        X, A, Y_next, base_id = X[:N], A[:N], Y_next[:N], base_id[:N]
        return X, A, Y_next, base_id, D

    F, A, ORn, source_id, d = load_branched(args.data)
    # Optional: restrict to a single action
    if args.actions_filter is not None:
        a_val = int(args.actions_filter)
        m = (A == a_val)
        F, A, ORn = F[m], A[m], ORn[m]
        source_id = source_id[m] if source_id is not None else None
        print(f"Filtered to action={a_val}: rows={len(F)}")

    # Optional: restrict to bases with >=2 distinct actions
    if args.multi_only and source_id is not None:
        from collections import defaultdict
        acts_by_base = defaultdict(set)
        for i, b in enumerate(source_id):
            acts_by_base[int(b)].add(int(A[i]))
        keep_bases = {b for b, acts in acts_by_base.items() if len(acts) >= 2}
        m = np.array([int(b) in keep_bases for b in source_id])
        if m.sum() == 0:
            raise SystemExit('multi_only filter removed all rows (no bases with >=2 actions)')
        F, A, ORn, source_id = F[m], A[m], ORn[m], source_id[m]
    E = np.zeros(len(F))
    agent_now = d['agent_pos'] if 'agent_pos' in d.files else None
    agent_next = d['agent_pos_next'] if 'agent_pos_next' in d.files else None
    ego_now = d['ego_now'] if 'ego_now' in d.files else None
    ego_next = d['ego_next'] if 'ego_next' in d.files else None

    if args.target == 'agent_next':
        Y, names = build_agent_next_targets(ORn)
    elif args.target == 'agent_next_offset':
        # Build 4-class offset labels from agent_pos and agent_pos_next if available, else from ORn and OR (current)
        if agent_now is not None and agent_next is not None:
            P = np.asarray(agent_now); Q = np.asarray(agent_next)
            dx = (Q[:,0] - P[:,0]) % ORn.shape[1]; dy = (Q[:,1] - P[:,1]) % ORn.shape[2]
        else:
            # derive from current/next agent channel
            ch_sums = ORn.reshape(len(ORn), -1, ORn.shape[-1]).mean(axis=1)
            agent_ch = int(np.argmin(ch_sums))
            # current observation approximated by shifting next back by action is messy; require provided positions
            raise SystemExit('agent_next_offset requires agent_pos and agent_pos_next in dataset')
        # Map wrapped deltas to 4 actions: right(0,1), left(0,6), down(1,0), up(6,0)
        def to_cls(dx, dy):
            if dx==0 and dy==1: return 0
            if dx==0 and dy==ORn.shape[2]-1: return 1
            if dx==1 and dy==0: return 2
            if dx==ORn.shape[1]-1 and dy==0: return 3
            return -1
        y_cls = np.array([to_cls(int(x), int(y)) for x,y in zip(dx,dy)], dtype=int)
        ok = y_cls >= 0
        F, A, ORn, source_id = F[ok], A[ok], ORn[ok], (source_id[ok] if source_id is not None else None)
        Y = np.zeros((len(F), 4), dtype=int)
        for i,c in enumerate(y_cls[ok]):
            Y[i, c] = 1
        names = [f'agent_next_offset_{c}' for c in range(4)]
    else:
        if args.egocentric and ego_next is not None:
            Y, names = build_targets(ego_next)
        else:
            Y, names = build_targets(ORn)
    # Build design matrices
    rows = []
    K = int(A.max()) + 1
    assert A.min() >= 0 and A.max() < K, "Action values must be in [0..K-1]"
    Aoh = np.eye(K, dtype=np.float32)[A]
    X_base = F
    if args.use_kronecker:
        # Build x⊗a: [N, D*4], place x in block of action, zeros elsewhere
        N, D = F.shape
        kron = np.zeros((N, D * K), dtype=F.dtype)
        for a in range(K):
            m = (A == a)
            kron[m, a*D:(a+1)*D] = F[m]
        X_plus = np.concatenate([F, Aoh, kron], axis=1)
    else:
        X_plus = np.concatenate([F, Aoh], axis=1)

    # Diagnostics: dataset sanity
    import numpy as _np
    n_rows = len(F)
    n_bases = len(_np.unique(source_id)) if source_id is not None else n_rows
    # count multi-action bases (post-filter)
    if source_id is not None:
        ids, cnts = _np.unique(source_id, return_counts=True)
        acts_per = {}
        for bid in ids:
            acts_per[int(bid)] = _np.unique(A[source_id==bid]).size
        n_multi_bases = int(sum(v >= 2 for v in acts_per.values()))
    else:
        n_multi_bases = 0
    counts_all = {int(k): int((A==k).sum()) for k in range(K)}
    print(f"Rows={n_rows} | Bases={n_bases} | Multi-bases={n_multi_bases} | action counts={counts_all}")

    # Prepare a single grouped split by base_id
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    groups = source_id if source_id is not None else _np.arange(len(F))
    tr_idx, te_idx = next(gss.split(_np.arange(len(F)), groups=groups))
    counts_tr = {int(k): int((A[tr_idx]==k).sum()) for k in range(K)}
    counts_te = {int(k): int((A[te_idx]==k).sum()) for k in range(K)}
    print(f"Train action counts={counts_tr} | Test action counts={counts_te}")
    test_bases = _np.unique(groups[te_idx])
    test_bases_str = ';'.join(str(int(b)) for b in test_bases)

    # Diagnostics: target prevalence summary
    pos_per_target = Y.sum(axis=0)
    print("Targets summary:", "N_total=", Y.shape[1], "pos>=", args.min_pos, int((pos_per_target >= args.min_pos).sum()))

    # Fit per-target with stratified split to guarantee positives in both folds
    au_base_all = []; au_plus_all = []; au_act_all = []
    au_base_by_a = {a: [] for a in range(K)}
    au_plus_by_a = {a: [] for a in range(K)}
    au_act_by_a = {a: [] for a in range(K)}
    used = 0

    # Special path: multiclass agent_next (absolute cell) for counterfactual per-base accuracy
    if args.target == 'agent_next':
        # Build multiclass labels (cell index 0..H*W-1)
        N, H, W, C = ORn.shape
        ch_sums = ORn.reshape(N, -1, C).sum(axis=1).mean(axis=0)
        agent_ch = int(np.argmin(ch_sums))
        y_cls = np.argmax(ORn[..., agent_ch].reshape(N, -1), axis=1)
        # use module-level sklearn imports
        clf_base = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        clf_plus = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        clf_act  = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000, multi_class='multinomial'))
        clf_base.fit(X_base[tr_idx], y_cls[tr_idx])
        clf_plus.fit(X_plus[tr_idx], y_cls[tr_idx])
        clf_act.fit(Aoh[tr_idx], y_cls[tr_idx])
        acc_base = float((clf_base.predict(X_base[te_idx]) == y_cls[te_idx]).mean())
        acc_plus = float((clf_plus.predict(X_plus[te_idx]) == y_cls[te_idx]).mean())
        acc_act  = float((clf_act.predict(Aoh[te_idx])   == y_cls[te_idx]).mean())
        rows.append(dict(target='__summary__', variant='agent_next_multiclass_base', action='all', auroc=acc_base, delta=0.0, metric='acc'))
        rows.append(dict(target='__summary__', variant='agent_next_multiclass_plus', action='all', auroc=acc_plus, delta=acc_plus-acc_base, metric='acc'))
        rows.append(dict(target='__summary__', variant='agent_next_multiclass_action_only', action='all', auroc=acc_act,  delta=acc_act-acc_base, metric='acc'))
        # Counterfactual per-base accuracy: same base_id, different actions
        bases_te = _np.unique(groups[te_idx])
        base_acc_base = []; base_acc_plus = []; base_acc_act = []
        for b in bases_te:
            idx = _np.where(groups == b)[0]
            idx = idx[_np.isin(idx, te_idx)]
            if len(idx) < 2:
                continue
            true = y_cls[idx]
            base_acc_base.append(float((clf_base.predict(X_base[idx]) == true).mean()))
            base_acc_plus.append(float((clf_plus.predict(X_plus[idx]) == true).mean()))
            base_acc_act.append(float((clf_act.predict(Aoh[idx])     == true).mean()))
        if base_acc_plus:
            rows.append(dict(target='__summary__', variant='agent_next_cf_acc_base', action='all', auroc=float(_np.mean(base_acc_base) if base_acc_base else 0.0), delta=0.0, metric='acc'))
            rows.append(dict(target='__summary__', variant='agent_next_cf_acc_plus', action='all', auroc=float(_np.mean(base_acc_plus)), delta=0.0, metric='acc'))
            rows.append(dict(target='__summary__', variant='agent_next_cf_acc_action_only', action='all', auroc=float(_np.mean(base_acc_act) if base_acc_act else 0.0), delta=0.0, metric='acc'))
        # Continue with binary per-target loop as well (optional)

    for k_idx, name in enumerate(names[:args.max_targets]):
        y = Y[:, k_idx].astype(int)
        # global inclusion
        if (y.sum() < (args.min_pos + 1)) or ((len(y) - y.sum()) < (args.min_neg + 1)):
            continue
        # use the precomputed grouped split (by base_id) to avoid leakage
        y_tr = y[tr_idx]; y_te = y[te_idx]
        # ensure both classes present in test
        if (y_te.sum() < args.min_pos) or ((len(y_te) - y_te.sum()) < args.min_neg) or (np.unique(y_te).size < 2):
            continue
        # Permute test labels if requested
        if args.permute_labels:
            rng = np.random.RandomState(0)
            y_te = y_te.copy(); rng.shuffle(y_te)
        pipe_base = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
        pipe_plus = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
        pipe_act  = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
        pipe_base.fit(X_base[tr_idx], y_tr)
        pipe_plus.fit(X_plus[tr_idx], y_tr)
        pipe_act.fit(Aoh[tr_idx], y_tr)
        p_base = pipe_base.predict_proba(X_base[te_idx])[:, 1]
        p_plus = pipe_plus.predict_proba(X_plus[te_idx])[:, 1]
        p_act  = pipe_act.predict_proba(Aoh[te_idx])[:, 1]
        au_b = roc_auc_score(y_te, p_base)
        au_p = roc_auc_score(y_te, p_plus)
        au_a = roc_auc_score(y_te, p_act)
        au_base_all.append(au_b); au_plus_all.append(au_p); au_act_all.append(au_a)

        # per-target rows (all actions)
        rows.append(dict(target=name, variant='base', action='all', auroc=float(au_b), delta=0.0, test_bases=test_bases_str))
        rows.append(dict(target=name, variant='plus', action='all', auroc=float(au_p), delta=float(au_p - au_b), test_bases=test_bases_str))
        rows.append(dict(target=name, variant='action_only', action='all', auroc=float(au_a), delta=float(au_a - au_b), test_bases=test_bases_str))
        # COPY baseline (only for map_next targets in world-centric coords)
        if args.target == 'map_next' and not args.egocentric:
            try:
                ij = name.split(')')[0].split('(')[-1]
                i, j = [int(t) for t in ij.split(',')]
                ch = int(name.split('ch')[-1])
                y_te_world = y_te
                copy_score = ORn[:, i, j, ch][te_idx]
                au_copy = roc_auc_score(y_te_world, copy_score)
            except Exception:
                au_copy = float('nan')
        else:
            au_copy = float('nan')

        # Per-action AUROCs
        for a in range(K):
            m = (A[te_idx] == a)
            if m.sum() < args.min_pos: continue
            if np.unique(y_te[m]).size < 2: continue
            au_base_by_a[a].append(roc_auc_score(y_te[m], p_base[m]))
            au_plus_by_a[a].append(roc_auc_score(y_te[m], p_plus[m]))
            au_act_by_a[a].append(roc_auc_score(y_te[m], p_act[m]))
            rows.append(dict(target=name, variant='base', action=int(a), auroc=float(au_base_by_a[a][-1]), delta=0.0, test_bases=test_bases_str))
            rows.append(dict(target=name, variant='plus', action=int(a), auroc=float(au_plus_by_a[a][-1]), delta=float(au_plus_by_a[a][-1] - au_base_by_a[a][-1]), test_bases=test_bases_str))
            rows.append(dict(target=name, variant='action_only', action=int(a), auroc=float(au_act_by_a[a][-1]), delta=float(au_act_by_a[a][-1] - au_base_by_a[a][-1]), test_bases=test_bases_str))
        used += 1

    # Summary rows
    if au_base_all and au_plus_all:
        rows.append(dict(target='__summary__', variant='base', action='all', auroc=float(np.mean(au_base_all)), delta=0.0))
        rows.append(dict(target='__summary__', variant='plus', action='all', auroc=float(np.mean(au_plus_all)), delta=float(np.mean(au_plus_all) - np.mean(au_base_all))))
        rows.append(dict(target='__summary__', variant='action_only', action='all', auroc=float(np.mean(au_act_all)), delta=float(np.mean(au_act_all) - np.mean(au_base_all))))
        for a in range(K):
            if au_base_by_a[a] and au_plus_by_a[a]:
                base_m = float(np.mean(au_base_by_a[a])); plus_m = float(np.mean(au_plus_by_a[a]))
                rows.append(dict(target='__summary__', variant='base', action=int(a), auroc=base_m, delta=0.0))
                rows.append(dict(target='__summary__', variant='plus', action=int(a), auroc=plus_m, delta=float(plus_m - base_m)))
            if au_act_by_a[a]:
                act_m = float(np.mean(au_act_by_a[a]))
                rows.append(dict(target='__summary__', variant='action_only', action=int(a), auroc=act_m, delta=float(act_m - (float(np.mean(au_base_by_a[a])) if au_base_by_a[a] else 0.0))))
        # also add how many targets used
        rows.append(dict(target='__summary__', variant='n_targets', action='used', auroc=float(used), delta=0.0))

    # Optional quick visualization for one letter/channel: heatmap of predicted probs vs actual
    try:
        import matplotlib.pyplot as plt
        import os
        vis_out = Path(args.out_csv).with_name('next_map_vis.png')
        # pick first target index with valid model
        if au_plus_all:
            # Refit one target with plus model
            for k_idx, name in enumerate(names[:1]):
                y = Y[:, k_idx]
                pipe = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=2000))
                # simple stratified split for vis
                sss_vis = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
                (tr_v, te_v), = sss_vis.split(X_plus, y)
                pipe.fit(X_plus[tr_v], y[tr_v])
                p = pipe.predict_proba(X_plus[te_v])[:, 1]
                # pick first test example; reshape probs to HxWxL then select this channel
                Ntest = len(te_v)
                # Aggregate by mean over test set as a proxy visual
                # Reconstruct per-cell probabilities for this channel
                # names encodes (i,j)_chC; map back indices by parsing
                H_guess = int(np.sqrt(len(names) // (ORn.shape[-1]-1)))
                W_guess = H_guess
                grid = np.zeros((H_guess, W_guess))
                for idx_name, nm in enumerate(names):
                    if idx_name >= len(p): break
                    if f"ch" in nm and nm.endswith(name.split('ch')[-1]):
                        ij = nm.split(')')[0].split('(')[-1]
                        i, j = [int(t) for t in ij.split(',')]
                        grid[i, j] = float(p.mean())  # coarse proxy
                plt.figure()
                plt.title('Predicted prob map (mean over test)')
                plt.imshow(grid, cmap='viridis'); plt.colorbar()
                plt.tight_layout(); plt.savefig(vis_out, dpi=150); plt.close()
                break
    except Exception:
        pass

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        if rows:
            # Accumulate all keys across rows to avoid missing fields
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        else:
            w = csv.writer(f)
            w.writerow(['message'])
            w.writerow(['No valid targets passed inclusion criteria'])
    print(f"Saved next-map AUCs to {out}")


if __name__ == '__main__':
    main()


