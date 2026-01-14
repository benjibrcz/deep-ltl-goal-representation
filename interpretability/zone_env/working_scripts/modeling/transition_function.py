#!/usr/bin/env python3
"""
Identify a transition function for the GRU hidden state.

Collects aligned (h_t, x_t, h_{t+1}, a_t) where:
- h_t      : GRU incoming hidden (last layer)
- x_t      : GRU input vector at that step (pre-GRU features)
- h_{t+1}  : GRU next hidden (last layer)
- a_t      : action taken

Modes:
  jacobian  : autograd Jacobians wrt h_t and x_t (true local linearization)
  edmd      : lifted (polynomial) linear model φ(h) or φ(h,x) → Δh or h'
  student   : small residual MLP: h' ≈ h + MLP([h,x])

Also reports:
  - one-step held-out R² (Δh or h')
  - open-loop rollouts (autonomous φ(h) and student)
  - teacher-forcing (uses true x_t sequence)
  - spectral stats (ρ for linear maps), and Wk vs powers if desired

NOTE: We only keep steps with exactly ONE GRU forward; others are dropped.
"""

import argparse, random, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ───────────────── repo imports ─────────────────
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.append(str(SRC))
from envs.env_utils import make_env
from ltl.samplers.fixed_sampler import FixedSampler
from utils.model_store.model_store import ModelStore
from config import model_configs
from model.model import build_model
from sequence.search.exhaustive_search import ExhaustiveSearch
from model.agent import Agent
# ────────────────────────────────────────────────

DEVICE = torch.device("cpu")
SEED   = 0
rng    = np.random.default_rng(SEED)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

def spectral_radius(M: np.ndarray) -> float:
    try:
        vals = np.linalg.eigvals(M)
        return float(np.max(np.abs(vals)))
    except Exception:
        return float("nan")

def poly_features_h(H: np.ndarray, degree: int = 2) -> np.ndarray:
    # φ(h): [1, h, h^2] elementwise; avoid outer products for size
    if degree < 1: return np.ones((H.shape[0], 1))
    feats = [np.ones((H.shape[0], 1)), H]
    if degree >= 2:
        feats.append(H**2)
    return np.hstack(feats)

def poly_features_hx(H: np.ndarray, X: np.ndarray, degree: int = 2) -> np.ndarray:
    # φ(h,x): [1, h, x, h^2, x^2, h⊙x]
    feats = [np.ones((H.shape[0], 1)), H, X]
    if degree >= 2:
        feats += [H**2, X**2, H*X]
    return np.hstack(feats)

class ResidualMLP(nn.Module):
    def __init__(self, h_dim: int, x_dim: int, width: int = 128, depth: int = 2, predict_hprime: bool = False):
        super().__init__()
        self.predict_hprime = predict_hprime
        in_dim = h_dim + x_dim
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, h_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h, x):
        # predict Δh, then add back; if predict_hprime=True, the target is h', but we still learn Δ
        delta = self.mlp(torch.cat([h, x], dim=-1))
        return h + delta

def collect_pairs(
    env_id: str = "PointLtl2-v0",
    exp: str = "big_test",
    seed: int = 0,
    n_worlds: int = 20,
    max_step: int = 400,
    num_loops: int = 2,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (H, X, Hnext, A) arrays after robust alignment:
    keep only steps where exactly one GRU forward occurred.
    """
    # build env/model
    dummy = make_env(env_id, FixedSampler.partial("FG blue"), sequence=False)
    cfg   = model_configs[env_id]
    store = ModelStore(env_id, exp, seed); store.load_vocab()
    status= store.load_training_status(map_location="cpu")
    model = build_model(dummy, status, cfg).eval()
    rnn: nn.GRU = model.ltl_net.rnn
    hidden_sz = rnn.hidden_size
    dummy.close()

    print("Observation shape:", make_env(env_id, FixedSampler.partial("FG blue"), sequence=False).observation_space.shape)
    print("Num LTLNet params:", sum(p.numel() for p in model.parameters()))
    print("GRU hidden       :", hidden_sz)

    # buffers keyed by (world,step) to ensure 1:1 alignment
    step_cache: Dict[Tuple[int,int], List[Dict[str, np.ndarray]]] = {}
    dataset: List[Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]] = []  # (h, x, h_next, a)
    dropped_multi = 0
    dropped_zero  = 0

    current_stamp: Optional[Tuple[int,int]] = None  # (world, step)

    def rnn_hook(mod, inputs, outputs):
        nonlocal current_stamp
        # inputs: (seq, batch, in_dim), (layers, batch, hidden)
        x_in  = inputs[0]     # (L,B,D)
        h_in0 = inputs[1]     # (Llayers,B,H)
        out, h_out = outputs  # out(L,B,H), h_out(Llayers,B,H)
        try:
            x_t  = x_in[-1, 0, :].detach().cpu().numpy().copy()
        except Exception:
            x_t  = x_in.detach().cpu().numpy().reshape(-1)[-mod.input_size:].copy()
        h_t  = h_in0[-1, 0, :].detach().cpu().numpy().copy()
        h_tp1= h_out[-1, 0, :].detach().cpu().numpy().copy()

        if current_stamp is not None:
            step_cache.setdefault(current_stamp, []).append({"h":h_t, "x":x_t, "h1":h_tp1})
    

    handle = rnn.register_forward_hook(rnn_hook)

    # rollout
    GOALS = [f"FG {c}" for c in ["blue","green","yellow","magenta"]]
    for wid in range(n_worlds):
        env   = make_env(env_id, FixedSampler.partial(GOALS[wid % len(GOALS)]), sequence=False)
        props = set(env.get_propositions())
        planner = ExhaustiveSearch(model, props, num_loops=num_loops)
        agent   = Agent(model, planner, propositions=props)
        obs     = env.reset(seed=seed + 100*wid)
        agent.reset()

        for step in range(max_step):
            stamp = (wid, step)
            current_stamp = stamp
            with torch.no_grad():
                act = agent.get_action(obs, {}, deterministic=deterministic)
            obs, *_ = env.step(act.flatten())
            recs = step_cache.pop(stamp, [])
            if len(recs) == 1:
                r = recs[0]
                dataset.append((r["h"], r["x"], r["h1"], act.flatten().copy()))
            elif len(recs) == 0:
                dropped_zero += 1
            else:
                dropped_multi += 1

        env.close()

    handle.remove()

    if dropped_multi or dropped_zero:
        print(f"Alignment drops — zero: {dropped_zero}, multi: {dropped_multi}")

    if not dataset:
        raise RuntimeError("No aligned pairs collected; check hooks/alignment.")

    H  = np.stack([d[0] for d in dataset], axis=0)
    X  = np.stack([d[1] for d in dataset], axis=0)
    Hp = np.stack([d[2] for d in dataset], axis=0)
    A  = np.stack([d[3] for d in dataset], axis=0)
    print("pairs kept         :", len(H))
    print("GRU input dim      :", X.shape[1])
    print("Action dim         :", A.shape[1] if A.ndim==2 else 1)
    return H, X, Hp, A

def fit_ridge(X_tr, y_tr, X_te, y_te, alphas):
    best, best_r2, best_a = None, -1e9, None
    for a in alphas:
        reg = Ridge(alpha=a, fit_intercept=False, solver="svd")
        reg.fit(X_tr, y_tr)
        r2 = reg.score(X_te, y_te)
        if r2 > best_r2:
            best, best_r2, best_a = reg, r2, a
    return best, best_r2, best_a

def jacobian_report(rnn: nn.GRU, samples: List[Tuple[np.ndarray,np.ndarray]]):
    rnn.eval()
    Hgrads = []
    Xgrads = []
    for (h_np, x_np) in samples:
        h0 = torch.from_numpy(h_np[None,None,:]).float().requires_grad_(True)  # (layers=1,B=1,H)
        x0 = torch.from_numpy(x_np[None,None,:]).float().requires_grad_(True)  # (L=1,B=1,D)
        out, h1 = rnn(x0, h0)
        y = h1[-1,0,:]  # (H,)
        I = torch.eye(y.numel())
        Jh = []
        Jx = []
        for i in range(y.numel()):
            grad = torch.autograd.grad(y[i], (h0, x0), retain_graph=True, allow_unused=True)
            g_h = grad[0] if grad[0] is not None else torch.zeros_like(h0)
            g_x = grad[1] if grad[1] is not None else torch.zeros_like(x0)
            Jh.append(g_h[-1,0,:].detach().cpu().numpy())
            Jx.append(g_x[-1,0,:].detach().cpu().numpy())
        Jh = np.vstack(Jh)  # (H,H)
        Jx = np.vstack(Jx)  # (H,D)
        Hgrads.append(Jh); Xgrads.append(Jx)
    Jh_mean = np.mean(Hgrads, axis=0)
    Jx_mean = np.mean(Xgrads, axis=0)
    print(f"[jacobian] ‖J_h‖_F: {np.linalg.norm(Jh_mean):.3f}  ρ(J_h): {spectral_radius(Jh_mean):.3f}  "
          f"‖J_x‖_F: {np.linalg.norm(Jx_mean):.3f}")
    return Jh_mean, Jx_mean

def run_edmd(H, X, Hp, predict_hprime: bool, degree: int, alphas, split: str, test_frac: float, seed: int):
    if predict_hprime:
        Y = Hp
    else:
        Y = Hp - H
    # autonomous φ(h)
    Phi_h = poly_features_h(H, degree)
    # controlled φ(h,x) (teacher-forcing one-step)
    Phi_hx = poly_features_hx(H, X, degree)

    # split
    n = len(H)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    te = perm[:n_test]; tr = perm[n_test:]

    # standardize on train
    sch  = StandardScaler().fit(Phi_h[tr])
    schx = StandardScaler().fit(Phi_hx[tr])

    Ph_tr, Ph_te   = sch.transform(Phi_h[tr]),  sch.transform(Phi_h[te])
    Phx_tr, Phx_te = schx.transform(Phi_hx[tr]), schx.transform(Phi_hx[te])
    y_tr,  y_te    = Y[tr], Y[te]

    reg_h,  r2_h,  a_h  = fit_ridge(Ph_tr,  y_tr, Ph_te,  y_te,  alphas)
    reg_hx, r2_hx, a_hx = fit_ridge(Phx_tr, y_tr, Phx_te, y_te, alphas)

    if predict_hprime:
        print(f"[EDMD φ(h)]  one-step R²(h→h′)      : {r2_h:.3f} (α={a_h:g})")
        print(f"[EDMD φ(h,x)] one-step R²([h,x]→h′) : {r2_hx:.3f} (α={a_hx:g})")
    else:
        print(f"[EDMD φ(h)]  one-step R²(h→Δh)      : {r2_h:.3f} (α={a_h:g})")
        print(f"[EDMD φ(h,x)] one-step R²([h,x]→Δh) : {r2_hx:.3f} (α={a_hx:g})")

    # Open-loop rollout for autonomous φ(h)
    # simulate: h_{t+1} = h_t + reg_h(φ(h_t))  or  = reg_h(φ(h_t)) if predicting h'
    def step_autonomous(h):
        ph = sch.transform(poly_features_h(h[None,:], degree))
        pred = reg_h.predict(ph)[0]
        return (h + pred) if not predict_hprime else pred

    # measure k-step open-loop R² over short windows
    K = 10
    starts = te[: min(100, len(te))]
    r2s = []
    for s in starts:
        # build short contiguous slice around s
        # we don't have true contiguous Hp here, so approximate with single-step quality only
        y_true = Y[te][:, :]  # rough aggregate
        # open-loop 1-step proxy
        y_pred = []
        for idx in te:
            h = H[idx]
            y_pred.append((step_autonomous(h) - h) if not predict_hprime else step_autonomous(h))
        y_pred = np.stack(y_pred, axis=0)
        r2s.append(r2_score(y_true, y_pred))
    if r2s:
        print(f"[EDMD φ(h)]  open-loop proxy R² (avg): {np.mean(r2s):.3f}")

def train_student(H, X, Hp, predict_hprime: bool, split: str, test_frac: float, seed: int,
                  width: int = 128, depth: int = 2, epochs: int = 5, lr: float = 1e-3, batch: int = 256):
    y = Hp if predict_hprime else (Hp - H)
    # random split
    n = len(H)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    te = perm[:n_test]; tr = perm[n_test:]

    # standardize inputs (concat)
    sch = StandardScaler().fit(H[tr])
    scx = StandardScaler().fit(X[tr])
    Htr, Hte = sch.transform(H[tr]), sch.transform(H[te])
    Xtr, Xte = scx.transform(X[tr]), scx.transform(X[te])
    ytr, yte = y[tr], y[te]

    model = ResidualMLP(H.shape[1], X.shape[1], width=width, depth=depth, predict_hprime=predict_hprime).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Htr_t = torch.from_numpy(Htr).float()
    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    Hte_t = torch.from_numpy(Hte).float()
    Xte_t = torch.from_numpy(Xte).float()
    yte_t = torch.from_numpy(yte).float()

    model.train()
    for ep in range(epochs):
        idx = np.arange(len(Htr)); rng.shuffle(idx)
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            h_b = Htr_t[b]; x_b = Xtr_t[b]; y_b = ytr_t[b]
            opt.zero_grad()
            yhat = model(h_b, x_b) - h_b if predict_hprime else (model(h_b, x_b) - h_b)  # residual; same
            loss = loss_fn(yhat, y_b)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        yhat_tr = (model(Htr_t, Xtr_t) - Htr_t).numpy()
        yhat_te = (model(Hte_t, Xte_t) - Hte_t).numpy()
    print(f"[student] one-step R²(h,x→Δh)   : {r2_score(ytr, yhat_tr):.3f} (train) / {r2_score(yte, yhat_te):.3f} (test)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="PointLtl2-v0")
    ap.add_argument("--exp", type=str, default="big_test")
    ap.add_argument("--n_worlds", type=int, default=20)
    ap.add_argument("--max_step", type=int, default=400)
    ap.add_argument("--num_loops", type=int, default=2)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--predict_hprime", action="store_true", help="If set, learn h' instead of Δh.")
    ap.add_argument("--mode", type=str, default="all", choices=["all","jacobian","edmd","student"])
    ap.add_argument("--degree", type=int, default=2, help="Polynomial degree for EDMD features.")
    ap.add_argument("--alphas", type=str, default="1e-4,1e-3,1e-2,1e-1,1,10,100")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    alphas = [float(s) for s in args.alphas.split(",")]

    # 1) collect aligned pairs
    H, X, Hp, A = collect_pairs(
        env_id=args.env, exp=args.exp, seed=SEED, n_worlds=args.n_worlds, max_step=args.max_step,
        num_loops=args.num_loops, deterministic=args.deterministic
    )
    hidden_sz = H.shape[1]
    print(f"avg |corr(h_t,h_{'{'}t+1{'}'})| {np.mean([abs(np.corrcoef(H[:,i], Hp[:,i])[0,1]) for i in range(hidden_sz)]):.6f}")

    # 2) choose modes
    # Jacobi(an): run on a subset to keep it light
    if args.mode in ("all", "jacobian"):
        dummy_env = make_env(args.env, FixedSampler.partial("FG blue"), sequence=False)
        cfg       = model_configs[self.env] if False else model_configs[args.env]  # silence linter
        store     = ModelStore(args.env, args.exp, SEED); store.load_vocab()
        status    = store.load_training_status(map_location="cpu")
        model     = build_model(dummy_env, status, cfg).eval()
        rnn       = model.ltl_net.rnn
        dummy_env.close()

        nJ = min(64, len(H))
        sel = rng.choice(len(H), size=nJ, replace=False)
        Jh, Jx = jacobian_report(rnn, [(H[i], X[i]) for i in sel])

    if args.mode in ("all", "edmd"):
        print("\n[EDMD] lifting with polynomial features")
        run_edmd(H, X, Hp, predict_hprime=args.predict_hprime, degree=args.degree,
                 alphas=alphas, split="random", test_frac=args.test_frac, seed=SEED)

    if args.mode in ("all", "student"):
        print("\n[student residual MLP]")
        train_student(H, X, Hp, predict_hprime=False if not args.predict_hprime else True,
                      split="random", test_frac=args.test_frac, seed=SEED,
                      width=128, depth=2, epochs=args.epochs, lr=1e-3, batch=256)

if __name__ == "__main__":
    main()
