#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BranchDataset(Dataset):
    def __init__(self, F_base, A, C_next, base_id, variant='plus', min_k: int = 3, cos_div_thresh: float = 0.99, k: int = 4):
        self.F_base = F_base
        self.A = A
        self.C_next = C_next
        self.base_id = base_id
        self.variant = variant
        self.k = int(k)
        # group indices by base
        from collections import defaultdict
        self.by_base = defaultdict(list)
        for i, b in enumerate(base_id):
            self.by_base[int(b)].append(i)
        # filter to bases with >=min_k candidates and cosine diversity
        self.bases = []
        for b, idxs in self.by_base.items():
            if len(idxs) < max(min_k, self.k):
                continue
            X = self.C_next[idxs]
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            ok = False
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    ai = X[i] / (np.linalg.norm(X[i]) + 1e-8)
                    aj = X[j] / (np.linalg.norm(X[j]) + 1e-8)
                    cos = float(np.dot(ai, aj))
                    if cos < cos_div_thresh:
                        ok = True; break
                if ok: break
            if not ok:
                continue
            self.bases.append(int(b))

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        b = self.bases[idx]
        idxs = self.by_base[b]
        # query is the common base feature (they share the same base feature across branches); take first
        z_t = self.F_base[idxs[0]].astype(np.float32)
        a_all = self.A[idxs].astype(int)
        z_next = self.C_next[idxs].astype(np.float32)
        # choose a positive index uniformly and sample k-1 negatives
        pos_global = np.random.randint(0, len(idxs))
        a_star = a_all[pos_global]
        # select candidate indices
        choices = list(range(len(idxs)))
        choices.remove(pos_global)
        if len(choices) >= (self.k - 1):
            neg_local = np.random.choice(choices, size=self.k - 1, replace=False)
        else:
            neg_local = np.random.choice(choices, size=self.k - 1, replace=True)
        sel_local = np.concatenate([[pos_global], np.array(neg_local, dtype=int)])
        # shuffle selection so positive position is random within [0..k-1]
        perm = np.random.permutation(self.k)
        sel_local = sel_local[perm]
        pos_j = int(np.where(perm == 0)[0][0])
        z_next = z_next[sel_local]
        # build query per variant
        if self.variant == 'plus':
            K = int(self.A.max()) + 1
            a_oh = np.eye(K, dtype=np.float32)[a_star]
            q = np.concatenate([z_t, a_oh], axis=0)
        elif self.variant == 'base':
            q = z_t
        elif self.variant == 'action_only':
            K = int(self.A.max()) + 1
            q = np.eye(K, dtype=np.float32)[a_star]
        else:
            raise ValueError('unknown variant')
        return (
            torch.from_numpy(q),
            torch.from_numpy(z_next),
            torch.tensor(pos_j, dtype=torch.long),
            torch.tensor(int(a_star), dtype=torch.long)
        )


class BilinearScorer(nn.Module):
    def __init__(self, q_dim: int, z_dim: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.q2d = nn.Linear(q_dim, z_dim)
            self.head = nn.Sequential(
                nn.Linear(q_dim + z_dim + z_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            self.mode = 'mlp'
        else:
            self.W = nn.Linear(q_dim, z_dim, bias=False)
            self.mode = 'bilinear'

    def forward(self, q: torch.Tensor, Zc: torch.Tensor, l2norm: bool = True) -> torch.Tensor:
        # Zc: [B, k, z_dim]
        if self.mode == 'mlp':
            B, k, D = Zc.shape
            q_rep = q.unsqueeze(1).expand(B, k, q.shape[-1])
            q_proj = self.q2d(q_rep)
            if l2norm:
                q_proj = q_proj / (torch.norm(q_proj, dim=-1, keepdim=True) + 1e-8)
                Zc = Zc / (torch.norm(Zc, dim=-1, keepdim=True) + 1e-8)
            prod = q_proj * Zc
            x = torch.cat([q_rep, Zc, prod], dim=-1)
            return self.head(x).squeeze(-1)
        else:
            # bilinear: score = (W q) dot z'
            Wq = self.W(q)  # [B, z_dim]
            if l2norm:
                Wq = Wq / (torch.norm(Wq, dim=-1, keepdim=True) + 1e-8)
                Zc = Zc / (torch.norm(Zc, dim=-1, keepdim=True) + 1e-8)
            return torch.einsum('bd,bkd->bk', Wq, Zc)


def collate_fn(batch):
    qs, znexts, pos_idx, a_star = zip(*batch)
    qs = torch.stack(qs, dim=0)
    # pad candidates to same k by sampling if needed (here assume same k within batch)
    # For simplicity, we use per-sample full candidate set as provided
    # stack candidates along dim=1
    max_k = max(z.shape[0] if z.ndim>1 else 1 for z in znexts)
    Zc = []
    for z in znexts:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        Zc.append(z)
    Zc = torch.stack(Zc, dim=0)
    pos = torch.stack(pos_idx, dim=0)
    a_star = torch.stack(a_star, dim=0)
    return qs, Zc, pos, a_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='branched CLEAN npz')
    ap.add_argument('--feature_key', default='feature_t')
    ap.add_argument('--next_key', default='obs_next_raw', help='Allocentric next map key for candidates')
    ap.add_argument('--cand_repr', default='map_flat', choices=['map_flat','next_cell_embed'])
    ap.add_argument('--variant', default='plus', choices=['plus','base','action_only'])
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min_k', type=int, default=3)
    ap.add_argument('--k', type=int, default=4, help='Number of candidates per base for contrastive set')
    ap.add_argument('--cos_div_thresh', type=float, default=0.99)
    ap.add_argument('--split_by_base', type=float, default=0.7)
    ap.add_argument('--out_csv', default='interpretability/letter_world/results/contrastive_branch.csv')
    ap.add_argument('--permute_candidates', action='store_true')
    ap.add_argument('--l2norm', action='store_true')
    ap.add_argument('--plus_map', choices=['delta','ridge'], default=None, help='Apply action-conditioned mapping to plus variant')
    ap.add_argument('--ridge_alpha', type=float, default=1.0, help='Ridge regression alpha for plus_map=ridge')
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
    A = np.asarray(D['action']).astype(int)
    B = np.asarray(D['base_id'] if 'base_id' in D.files else D['source_id']).astype(int)
    # build allocentric candidate representation
    if args.cand_repr == 'map_flat':
        ORn = D[args.next_key]
        ORn = np.asarray(ORn)
        if ORn.dtype == object:
            ORn = np.stack(list(ORn), axis=0)
        C_next = ORn.reshape(ORn.shape[0], -1).astype(np.float32)
        mu = C_next.mean(axis=0, keepdims=True)
        sigma = C_next.std(axis=0, keepdims=True) + 1e-6
        C_next = (C_next - mu) / sigma
    else:
        if 'agent_pos_next' in D.files:
            PosN = np.asarray(D['agent_pos_next'])
            idx = (PosN[:,0].astype(int) * 7 + PosN[:,1].astype(int))
        else:
            ORn = D['obs_next_raw']
            ORn = np.asarray(ORn)
            if ORn.dtype == object:
                ORn = np.stack(list(ORn), axis=0)
            ch_sums = ORn.reshape(ORn.shape[0], -1, ORn.shape[-1]).sum(axis=1).mean(axis=0)
            agent_ch = int(np.argmin(ch_sums))
            idx = np.argmax(ORn[..., agent_ch].reshape(ORn.shape[0], -1), axis=1)
        G = 7*7
        C_next = np.zeros((len(idx), G), dtype=np.float32)
        C_next[np.arange(len(idx)), idx.astype(int)] = 1.0

    if args.permute_candidates:
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(C_next))
        C_next = C_next[perm]

    ds_full = BranchDataset(F_base=F, A=A, C_next=C_next, base_id=B, variant=args.variant,
                            min_k=args.min_k, cos_div_thresh=args.cos_div_thresh, k=args.k)
    if len(ds_full) == 0:
        raise SystemExit('No multi-branch bases with diverse nexts found')
    bases = np.array(ds_full.bases)
    rng = np.random.RandomState(0)
    rng.shuffle(bases)
    n_tr = max(1, int(args.split_by_base * len(bases)))
    train_bases = set(bases[:n_tr].tolist())
    test_bases = set(bases[n_tr:].tolist())
    def build_subset(bset):
        sub = BranchDataset(F_base=F, A=A, C_next=C_next, base_id=B, variant=args.variant,
                            min_k=args.min_k, cos_div_thresh=args.cos_div_thresh, k=args.k)
        sub.bases = [b for b in sub.bases if b in bset]
        return sub
    ds_tr = build_subset(train_bases)
    ds_te = build_subset(test_bases)
    if len(ds_te) == 0:
        raise SystemExit('Empty test set after split; adjust split_by_base')
    q_dim = (F.shape[1] + (A.max()+1)) if args.variant == 'plus' else (F.shape[1] if args.variant=='base' else (A.max()+1))
    model = BilinearScorer(q_dim, C_next.shape[1], hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    # Precompute action-conditioned mapping on train if requested
    d_a = None
    W_a = None; b_a = None
    if args.variant == 'plus' and hasattr(args, 'plus_map') and args.plus_map in ('delta','ridge'):
        train_mask = np.array([b in train_bases for b in B])
        Z_tr = F[train_mask]
        C_tr = C_next[train_mask]
        A_tr = A[train_mask]
        K = int(A.max()) + 1
        if args.plus_map == 'delta':
            d_a = {}
            for a in range(K):
                m = (A_tr == a)
                if not np.any(m):
                    d_a[a] = np.zeros(C_tr.shape[1], dtype=np.float32)
                else:
                    Zd = Z_tr[m]
                    Cd = C_tr[m]
                    if Zd.shape[1] != Cd.shape[1]:
                        Dd = Cd.shape[1]
                        Zp = np.zeros((Zd.shape[0], Dd), dtype=np.float32)
                        dmin = min(Dd, Zd.shape[1])
                        Zp[:, :dmin] = Zd[:, :dmin]
                        delta = Cd - Zp
                    else:
                        delta = Cd - Zd
                    d_a[a] = delta.mean(axis=0)
        else:
            W_a = {}
            b_a = {}
            F_dim = F.shape[1]
            for a in range(K):
                m = (A_tr == a)
                if not np.any(m):
                    W_a[a] = np.zeros((C_tr.shape[1], F_dim), dtype=np.float32)
                    b_a[a] = np.zeros((C_tr.shape[1],), dtype=np.float32)
                else:
                    Zm = Z_tr[m]
                    Cm = C_tr[m]
                    Z1 = np.concatenate([Zm, np.ones((Zm.shape[0], 1), dtype=np.float32)], axis=1)
                    lam = getattr(args, 'ridge_alpha', 1.0)
                    A_mat = Z1.T @ Z1 + lam * np.eye(Z1.shape[1], dtype=np.float32)
                    Wb = np.linalg.solve(A_mat, Z1.T @ Cm)
                    W_a[a] = Wb[:-1].T
                    b_a[a] = Wb[-1]

    model.train()
    for _ in range(args.epochs):
        for q, Zc, pos, a_star in loader_tr:
            # Zc: [B, k, D]
            if args.variant == 'plus' and hasattr(args, 'plus_map') and args.plus_map in ('delta','ridge'):
                F_dim = F.shape[1]
                zt = q[..., :F_dim]
                if args.plus_map == 'delta':
                    da = torch.stack([torch.from_numpy(d_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    q_proj = zt + da
                else:
                    WW = torch.stack([torch.from_numpy(W_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    bb = torch.stack([torch.from_numpy(b_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    q_proj = torch.einsum('bcf,bf->bc', WW, zt) + bb
                if args.l2norm:
                    qn = q_proj / (torch.norm(q_proj, dim=-1, keepdim=True) + 1e-8)
                    Zn = Zc / (torch.norm(Zc, dim=-1, keepdim=True) + 1e-8)
                else:
                    qn, Zn = q_proj, Zc
                logits = torch.einsum('bd,bkd->bk', qn, Zn)
                # no trainable params in delta/ridge mapping; skip optimizer
                continue
            else:
                logits = model(q, Zc, l2norm=args.l2norm)
                loss = ce(logits, pos)
                opt.zero_grad(); loss.backward(); opt.step()

    # Eval top-1/top-2
    model.eval()
    top1 = []; top2 = []; k_list = []; a_list = []; rank_hist = []
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for q, Zc, pos, a_star in loader_te:
            if args.variant == 'plus' and hasattr(args, 'plus_map') and args.plus_map in ('delta','ridge'):
                F_dim = F.shape[1]
                zt = q[..., :F_dim]
                if args.plus_map == 'delta':
                    da = torch.stack([torch.from_numpy(d_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    q_proj = zt + da
                else:
                    WW = torch.stack([torch.from_numpy(W_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    bb = torch.stack([torch.from_numpy(b_a[int(ai.item())]) for ai in a_star], dim=0).to(zt.dtype)
                    q_proj = torch.einsum('bcf,bf->bc', WW, zt) + bb
                if args.l2norm:
                    qn = q_proj / (torch.norm(q_proj, dim=-1, keepdim=True) + 1e-8)
                    Zn = Zc / (torch.norm(Zc, dim=-1, keepdim=True) + 1e-8)
                else:
                    qn, Zn = q_proj, Zc
                logits = torch.einsum('bd,bkd->bk', qn, Zn)
            else:
                logits = model(q, Zc, l2norm=args.l2norm)
            pred = torch.argmax(logits, dim=1)
            top1.extend((pred == pos).cpu().numpy().tolist())
            # top-2
            top2_pred = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).indices
            top2.extend([(pos[i].item() in top2_pred[i].cpu().numpy().tolist()) for i in range(len(pos))])
            k_list.extend([Zc.shape[1]] * q.shape[0])
            a_list.extend(a_star.cpu().numpy().tolist())
            # rank histogram
            ranks = torch.argsort(logits, dim=1, descending=True)
            for i in range(len(pos)):
                r = (ranks[i] == pos[i]).nonzero(as_tuple=False).item()
                rank_hist.append(int(r)+1)

    acc1 = float(np.mean(top1))
    acc2 = float(np.mean(top2))
    rand = float(np.mean([1.0/kk for kk in k_list])) if k_list else float('nan')
    # per-action acc1
    acc1_by_a = {}
    if a_list:
        a_arr = np.array(a_list)
        top1_arr = np.array(top1)
        for a in np.unique(a_arr):
            m = (a_arr == a)
            acc1_by_a[int(a)] = float(top1_arr[m].mean()) if m.any() else float('nan')
    # rank histogram fractions (for k up to 4)
    rh = {}
    if rank_hist:
        import collections
        cnt = collections.Counter(rank_hist)
        tot = sum(cnt.values())
        for r in sorted(cnt):
            rh[f'rank{r}_frac'] = float(cnt[r]/tot)
    from collections import defaultdict
    by_k = defaultdict(list)
    for i, kk in enumerate(k_list):
        by_k[int(kk)].append(int(top1[i]))
    # write CSV
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open('w', newline='') as f:
        keys = ['variant','acc1','acc2','rand_acc','k_mean','n_bases_test'] + [f'acc1_k{k}' for k in sorted(by_k.keys())] + [f'acc1_a{a}' for a in sorted(acc1_by_a.keys())] + list(rh.keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        row = dict(variant=args.variant, acc1=acc1, acc2=acc2, rand_acc=rand, k_mean=float(np.mean(list(by_k.keys())) if by_k else float('nan')), n_bases_test=len(ds_te))
        for k,v in by_k.items():
            row[f'acc1_k{k}'] = float(np.mean(v))
        for a,v in acc1_by_a.items():
            row[f'acc1_a{a}'] = v
        row.update(rh)
        w.writerow(row)
    print('Saved contrastive branch results to', out)


if __name__ == '__main__':
    main()


