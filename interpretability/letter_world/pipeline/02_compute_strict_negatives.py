import argparse
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--zeta', type=str, required=True, help='path to sequences zeta_q.pkl')
    p.add_argument('--lambda_', type=float, default=0.05, help='lambda for strict negatives')
    p.add_argument('--out', type=str, required=True)
    # Placeholder for state buffer if/when needed
    return p.parse_args()


def compute_strict_negatives(seqs, lambda_):
    # Placeholder implementation: pass-through (no-op) until value_oracle integration per-step
    # The structure is preserved; caller can later replace with true lambda-strict computation.
    return seqs


def main():
    args = parse_args()
    with open(args.zeta, 'rb') as f:
        payload = pickle.load(f)
    seqs = payload['sequences']
    strict_seqs = compute_strict_negatives(seqs, args.lambda_)

    out_payload = dict(payload)
    out_payload['strict_lambda'] = args.lambda_
    out_payload['strict_sequences'] = strict_seqs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out_payload, f)
    print(f"Saved strict sequences to {out_path}")


if __name__ == '__main__':
    main()


