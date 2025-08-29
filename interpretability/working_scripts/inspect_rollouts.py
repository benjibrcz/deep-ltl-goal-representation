import pandas as pd, numpy as np, re, sys
df = pd.read_parquet("interpretability/working_scripts/rollouts_stateful.parquet")
print("columns:", list(df.columns))
print("rows:", len(df))
has_goal = any(c in df.columns for c in ["goal_text","goal","goal_str","goal_now"])
print("has_goal_col:", has_goal)

def parse_goal(s):
    if not isinstance(s, str): return None
    m = re.findall(r"[A-Za-z]+", s)
    return m[-1].lower() if m else None

goal_col = next((c for c in ["goal_text","goal","goal_str","goal_now"] if c in df.columns), None)
if goal_col:
    g = df[goal_col].map(parse_goal)
    print("goal value counts (top):")
    print(g.value_counts().head(10))
    # detect switches per chain
    if "chain_id" not in df.columns:
        df["chain_id"] = 0
    df = df[df["h"].notna()].copy()
    df["_goal"] = g
    df["_goal_prev"] = df.groupby("chain_id")["_goal"].shift(1)
    switches = (df["_goal"] != df["_goal_prev"]) & df["_goal_prev"].notna()
    print("switch rows:", int(switches.sum()))
else:
    print("No goal_* column found.")
