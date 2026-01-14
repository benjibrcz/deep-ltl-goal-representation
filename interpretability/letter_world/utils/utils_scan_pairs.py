#!/usr/bin/env python3
import argparse
import numpy as np
from collections import Counter


def hits_before(seq: np.ndarray, A: int, B: int, H: int):
    for k in range(1, H + 1):
        x = int(seq[k])
        if x == A:
            return 1
        if x == B:
            return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--letters_key', default='letter_id')
    ap.add_argument('--episode_key', default='episode')
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--min_per_class', type=int, default=50)
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    L = np.asarray(D[args.letters_key])
    E = np.asarray(D[args.episode_key]) if args.episode_key in D.files else np.zeros(len(L))

    uniq_letters = sorted([int(x) for x in np.unique(L) if int(x) >= 0])
    pairs_stats = {}
    for A in uniq_letters:
        for B in uniq_letters:
            if A == B:
                continue
            pos = 0; neg = 0
            for ep in np.unique(E):
                idx = np.where(E == ep)[0]
                if idx.size < args.horizon + 1:
                    continue
                seq = L[idx]
                for t in range(0, len(seq) - args.horizon):
                    if int(seq[t]) in (A, B):
                        continue
                    y = hits_before(seq[t:t + args.horizon + 1], A, B, args.horizon)
                    if y is None:
                        continue
                    if y == 1:
                        pos += 1
                    else:
                        neg += 1
            if pos + neg > 0:
                pairs_stats[(A, B)] = (pos, neg)

    good = []
    for (A, B), (pos, neg) in sorted(pairs_stats.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        if pos >= args.min_per_class and neg >= args.min_per_class:
            good.append((A, B, pos, neg))

    print(f"H={args.horizon}")
    if good:
        print("Pairs with both classes >= min_per_class:")
        for A, B, pos, neg in good[:50]:
            print(f"pair=({A},{B}) pos={pos} neg={neg} frac_pos={pos/(pos+neg):.2f}")
    else:
        print("No robust pairs found; try increasing --horizon or lowering --min_per_class.")


if __name__ == '__main__':
    main()


