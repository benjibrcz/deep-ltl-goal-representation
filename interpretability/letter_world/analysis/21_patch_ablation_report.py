#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', type=str, default='interpretability/letter_world/results/patch_live.jsonl')
    p.add_argument('--offline', type=str, default='interpretability/letter_world/results/patch_offline.csv')
    p.add_argument('--steer', type=str, default='interpretability/letter_world/results/steer_grid.csv')
    p.add_argument('--out', type=str, default='interpretability/letter_world/results/patch_report.csv')
    p.add_argument('--make_plots', action='store_true')
    p.add_argument('--make_target_panels', action='store_true')
    args = p.parse_args()

    rows = []

    # Live
    live_path = Path(args.live)
    live_fr = {}
    if live_path.exists():
        flips_by_alpha = {}
        total_by_alpha = {}
        with live_path.open('r') as f:
            for line in f:
                rec = json.loads(line)
                a = float(rec.get('alpha', 0.0))
                flip = int(rec.get('base_action', -1) != rec.get('patched_action', -1))
                flips_by_alpha[a] = flips_by_alpha.get(a, 0) + flip
                total_by_alpha[a] = total_by_alpha.get(a, 0) + 1
        for a, n in total_by_alpha.items():
            fr = flips_by_alpha.get(a, 0) / max(n, 1)
            live_fr[a] = fr
            rows.append(dict(source='live', key=f'alpha={a}', metric='flip_rate', value=fr))

    # Offline
    offline_path = Path(args.offline)
    offline_df = None
    if offline_path.exists():
        import pandas as pd
        offline_df = pd.read_csv(offline_path)
        g = offline_df.groupby(['mode', 'alpha']).agg({'dce':'mean', 'flip_rate':'mean'}).reset_index()
        for _, r in g.iterrows():
            rows.append(dict(source='offline', key=f"{r['mode']},alpha={r['alpha']}", metric='dce', value=float(r['dce'])))
            rows.append(dict(source='offline', key=f"{r['mode']},alpha={r['alpha']}", metric='flip_rate', value=float(r['flip_rate'])))

    # Steering grid
    steer_path = Path(args.steer)
    steer_df = None
    if steer_path.exists():
        import pandas as pd
        steer_df = pd.read_csv(steer_path)
        g = steer_df.groupby(['dir', 'alpha']).agg({'dce':'mean', 'flip_rate':'mean'}).reset_index()
        for _, r in g.iterrows():
            rows.append(dict(source='steer', key=f"{r['dir']},alpha={r['alpha']}", metric='dce', value=float(r['dce'])))
            rows.append(dict(source='steer', key=f"{r['dir']},alpha={r['alpha']}", metric='flip_rate', value=float(r['flip_rate'])))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved patch ablation report to {out}")

    # Optional simple plots
    if args.make_plots and steer_df is not None:
        import matplotlib.pyplot as plt
        # Flip/ΔCE vs alpha for PC1 and random (mean±sd)
        pc1 = steer_df[steer_df['dir'] == 'PC1']
        rand = steer_df[steer_df['dir'].str.startswith('rand')]
        def agg(df, col):
            gg = df.groupby('alpha')[col].agg(['mean','std']).reset_index()
            return gg['alpha'].values, gg['mean'].values, gg['std'].values
        for col in ['flip_rate','dce']:
            a1, m1, s1 = agg(pc1, col)
            a2, m2, s2 = agg(rand, col)
            plt.figure()
            plt.errorbar(a1, m1, yerr=s1, label='PC1', marker='o')
            plt.errorbar(a2, m2, yerr=s2, label='Random (mean±sd)', marker='o')
            if offline_df is not None:
                off = offline_df[offline_df['mode']=='steer']
                ao = off['alpha'].values; mo = off[col].values
                plt.plot(ao, mo, label='Steer (offline)', marker='o')
            if live_fr:
                al = sorted(live_fr.keys()); ml = [live_fr[a] for a in al]
                if col == 'flip_rate':
                    plt.plot(al, ml, label='Live flip', marker='x')
            plt.xlabel('alpha'); plt.ylabel(col); plt.legend()
            plt.tight_layout()
            png = out.parent / f'dose_{col}.png'
            plt.savefig(png, dpi=150)
            plt.close()
        print(f"Saved dose-response plots to {out.parent}")

    # Targeted panels: discover offline targeted CSVs and live summary
    if args.make_target_panels:
        import pandas as pd
        import matplotlib.pyplot as plt
        res_dir = out.parent
        offline_target_files = sorted(res_dir.glob('patch_offline_target_*.csv'))
        live_summary = res_dir / 'patch_live_summary.csv'
        # Panel 1: ΔCE vs alpha (offline targeted)
        if offline_target_files:
            plt.figure()
            for f in offline_target_files:
                df = pd.read_csv(f)
                name = f.stem.replace('patch_offline_target_', '')
                if 'dce' in df.columns:
                    g = df.groupby('alpha')['dce'].mean().reset_index()
                    plt.plot(g['alpha'], g['dce'], marker='o', label=name)
            plt.xlabel('alpha'); plt.ylabel('ΔCE (offline targeted)'); plt.legend()
            plt.tight_layout()
            plt.savefig(res_dir / 'targeted_dce.png', dpi=150)
            plt.close()
        # Panel 2: targeted vs collateral flip vs alpha (offline + live)
        plt.figure()
        # offline targeted
        for f in offline_target_files:
            df = pd.read_csv(f)
            name = f.stem.replace('patch_offline_target_', '')
            if 'targeted_flip_rate' in df.columns:
                g = df.groupby('alpha')['targeted_flip_rate'].mean().reset_index()
                plt.plot(g['alpha'], g['targeted_flip_rate'], marker='o', label=f'{name} (offline targeted)')
        # live targeted/collateral if present
        if live_summary.exists():
            ldf = pd.read_csv(live_summary)
            if {'alpha','targeted_rate'}.issubset(ldf.columns):
                plt.plot(ldf['alpha'], ldf['targeted_rate'], marker='x', linestyle='--', label='live targeted')
            if {'alpha','collateral_rate'}.issubset(ldf.columns):
                plt.plot(ldf['alpha'], ldf['collateral_rate'], marker='x', linestyle=':', label='live collateral')
        plt.xlabel('alpha'); plt.ylabel('Targeted / Collateral flip rate'); plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / 'targeted_flip.png', dpi=150)
        plt.close()
        print(f"Saved targeted panels to {res_dir}")

    # Layer table (extend with actor_mid and critic_mid if present)
    table_path = out.parent / 'layer_table.csv'
    with table_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer','alpha','dce','flip_rate'])
        import pandas as pd
        # actor_prelogits from default offline file
        if offline_df is not None:
            off = offline_df[offline_df['mode']=='steer']
            for a in [0.25, 0.5, 1.0]:
                r = off[np.isclose(off['alpha'], a)]
                if len(r):
                    w.writerow(['actor_prelogits', a, float(r['dce'].mean()), float(r['flip_rate'].mean())])
        # actor_mid and critic_mid if csvs exist
        for layer_name, csv_name in [('actor_mid','patch_offline_actor_mid.csv'), ('critic_mid','patch_offline_critic_mid.csv')]:
            pth = out.parent / csv_name
            if pth.exists():
                df = pd.read_csv(pth)
                df = df[df['mode']=='steer'] if 'mode' in df.columns else df
                for a in [0.25, 0.5, 1.0]:
                    r = df[np.isclose(df['alpha'], a)] if 'alpha' in df.columns else None
                    if r is not None and len(r):
                        w.writerow([layer_name, a, float(r['dce'].mean()), float(r['flip_rate'].mean())])
    print(f"Saved layer table to {table_path}")


if __name__ == '__main__':
    main()
