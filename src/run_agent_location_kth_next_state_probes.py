#!/usr/bin/env python3
import subprocess
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--k-step', type=str, default=None, help='Comma-separated list of k values (e.g., 1,5,10) or a single integer step size (e.g., 5)')
parser.add_argument('--k-max', type=int, default=None, help='Maximum k value (used with integer k-step)')
args = parser.parse_args()

# Determine K_RANGE
if args.k_step:
    if ',' in args.k_step:
        # Comma-separated list
        K_RANGE = [int(k) for k in args.k_step.split(',')]
    else:
        # Single integer step size
        step = int(args.k_step)
        k_max = args.k_max if args.k_max is not None else 50
        K_RANGE = list(range(1, k_max+1, step))
else:
    K_RANGE = range(1, 51)

LAYER = 'env_net'
SCRIPT = 'src/probe_agent_location_kth_next_state_trajectories.py'

results = []
for k in K_RANGE:
    print(f'Running probe for k={k}...')
    proc = subprocess.run([
        'python', SCRIPT,
        '--layer', LAYER,
        '--k', str(k),
        '--n-worlds', '20',
        '--max-steps', '500',
        '--out', f'/dev/null'  # Don't save plots
    ], capture_output=True, text=True)
    output = proc.stdout
    # Parse MSE and R2 from output
    match = re.search(r'Test  MSE: ([0-9.]+)  Test  R\^2: ([\-0-9.]+)', output)
    if match:
        mse = float(match.group(1))
        r2 = float(match.group(2))
        results.append((k, mse, r2))
        print(f'k={k}: Test MSE={mse:.4f}, Test R^2={r2:.4f}')
    else:
        print(f'k={k}: Failed to parse results.')

print('\nSummary:')
print(f'k\tTest MSE\tTest R^2')
for k, mse, r2 in results:
    print(f'{k}\t{mse:.4f}\t{r2:.4f}') 