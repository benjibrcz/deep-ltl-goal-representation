#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def is_vec(x):
    try: a=np.asarray(x); return a.ndim==1 and a.size>0
    except: return False

def parse_goal(s):
    if not isinstance(s,str): return None
    m = re.findall(r"[A-Za-z]+", s)
    return m[-1].lower() if m else None

def group_split(groups, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq)<2:
        idx = np.arange(len(groups)); rng.shuffle(idx)
        nte = max(1, int(round(len(idx)*test_frac)))
        return idx[nte:], idx[:nte]
    rng.shuffle(uniq)
    nte = max(1, int(round(len(uniq)*test_frac)))
    test_g = set(uniq[:nte])
    te = np.array([g in test_g for g in groups])
    return np.where(~te)[0], np.where(te)[0]

def build_pairs(df):
    if "chain_id" not in df.columns: df["chain_id"]=0
    if "t" in df.columns: step_col="t"
    elif "step" in df.columns: step_col="step"
    else:
        df["_tmp_idx"]=df.groupby("chain_id").cumcount(); step_col="_tmp_idx"
    df = df[df["h"].apply(is_vec)].copy()
    goal_col = next((c for c in ["goal_text","goal","goal_str","goal_now"] if c in df.columns), None)
    rows=[]
    for g,gdf in df.groupby("chain_id",sort=False):
        gdf = gdf.sort_values(step_col)
        hs  = gdf["h"].apply(np.asarray).tolist()
        acts= gdf["a"].apply(np.asarray).tolist() if "a" in gdf.columns else [None]*len(gdf)
        goals = gdf[goal_col].tolist() if goal_col else [None]*len(gdf)
        for i in range(len(gdf)-1):
            rows.append({"chain_id":g,
                         "h_t":hs[i],
                         "a_t":acts[i],
                         "goal_t":goals[i],
                         "goal_prev":goals[i-1] if i>0 else None,
                         "h_tp1":hs[i+1]})
    pairs = pd.DataFrame(rows)
    pairs["goal_t"] = pairs["goal_t"].map(parse_goal)
    pairs["goal_prev"] = pairs["goal_prev"].map(parse_goal)
    pairs["is_switch"] = (pairs["goal_t"]!=pairs["goal_prev"]) & pairs["goal_prev"].notna()
    return pairs

def make_design(pairs, use_actions=False, use_goal=False, goal_order=None):
    H = len(pairs.iloc[0]["h_t"])
    Xs = [np.stack(pairs["h_t"].values, axis=0).astype(np.float32)]
    names = ["h"]
    if use_actions and not pairs["a_t"].isna().all():
        dim_a=None; A=[]
        for a in pairs["a_t"].values:
            if a is None: A.append(None)
            else:
                v=np.asarray(a,dtype=np.float32).ravel()
                dim_a = dim_a or v.size; A.append(v)
        if dim_a:
            A = np.stack([np.zeros(dim_a,dtype=np.float32) if v is None else v for v in A], axis=0)
            Xs.append(A); names.append("a")
    if use_goal:
        colors = goal_order or sorted({g for g in pairs["goal_t"].values if g is not None})
        idx = {c:i for i,c in enumerate(colors)}
        G = np.zeros((len(pairs), len(colors)), dtype=np.float32)
        for i,g in enumerate(pairs["goal_t"].values):
            if g in idx: G[i,idx[g]]=1.0
        if G.shape[1]>0:
            Xs.append(G); names.append("goal")
    X = np.concatenate(Xs, axis=1)
    Y = np.stack(pairs["h_tp1"].values, axis=0).astype(np.float32)
    return X,Y,names

def fit_eval(X,Y,groups,tr_idx,te_idx,alphas):
    Xsc = StandardScaler().fit(X[tr_idx])
    Ysc = StandardScaler().fit(Y[tr_idx])
    Xtr,Xte = Xsc.transform(X[tr_idx]), Xsc.transform(X[te_idx])
    Ytr,Yte = Ysc.transform(Y[tr_idx]), Ysc.transform(Y[te_idx])
    best=-1; r2=-1
    for a in alphas:
        reg = Ridge(alpha=a).fit(Xtr,Ytr)
        r = reg.score(Xte,Yte)
        if r>r2: r2=r; best=reg
    return r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--colors", type=str, default="green,blue,yellow,magenta")
    ap.add_argument("--alphas", type=str, default="1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = build_pairs(pd.read_parquet(args.parquet))
    print(f"pairs: {len(pairs)}; switches: {int(pairs['is_switch'].sum())}")

    groups = pairs["chain_id"].to_numpy()
    tr_idx, te_idx = group_split(groups, args.test_frac, args.seed)
    alphas = [float(s) for s in args.alphas.split(",")]
    goal_order = [s.strip().lower() for s in args.colors.split(",") if s.strip()]

    for subset_name, mask in [
        ("during-link", ~pairs["is_switch"].to_numpy()),
        ("switch-rows", pairs["is_switch"].to_numpy())
    ]:
        if mask.sum()<10:
            print(f"\n{subset_name}: too few rows ({mask.sum()})")
            continue
        print(f"\n{subset_name}: N={mask.sum()}")
        idx_map = np.where(mask)[0]
        # remap tr/te to subset
        tr_m = np.intersect1d(tr_idx, idx_map)
        te_m = np.intersect1d(te_idx, idx_map)
        if len(tr_m)<10 or len(te_m)<10:
            print("  (not enough rows in split)")
            continue
        for name,(ua,ug) in [
            ("h-only", (False, False)),
            ("h+act", (True,  False)),
            ("h+act+goal", (True,  True)),
        ]:
            X,Y,_ = make_design(pairs.iloc[mask], use_actions=ua, use_goal=ug, goal_order=goal_order)
            # tr_m/te_m are absolute indices; convert to local (mask) indices:
            loc = {g:i for i,g in enumerate(idx_map)}
            tr_local = np.array([loc[i] for i in tr_m])
            te_local = np.array([loc[i] for i in te_m])
            r2 = fit_eval(X,Y,groups=None,tr_idx=tr_local,te_idx=te_local,alphas=alphas)
            print(f"  {name:12s} | R^2={r2:.3f}")

if __name__=="__main__":
    main()
