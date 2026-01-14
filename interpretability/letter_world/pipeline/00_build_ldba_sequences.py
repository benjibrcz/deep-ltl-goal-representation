import argparse
import pickle
from pathlib import Path

import numpy as np

import sys
SRC = Path(__file__).resolve().parents[2] / 'src'
sys.path.append(str(SRC))

from envs.env_utils import make_env
from ltl.automata import ltl2ldba, LDBA
from sequence.search.exhaustive_search import ExhaustiveSearch


def build_ldba(env_name: str, formula: str) -> LDBA:
    env = make_env(env_name, lambda _: (lambda __: None), render_mode=None)  # sampler not used
    propositions = env.get_propositions()
    possible_assignments = env.get_possible_assignments()
    ldba = ltl2ldba(formula, propositions, simplify_labels=False)
    ldba.prune(possible_assignments)
    ldba.complete_sink_state()
    ldba.compute_sccs()
    return ldba


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='LetterEnv-v0')
    p.add_argument('--formula', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--num_loops', type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    ldba = build_ldba(args.env, args.formula)

    # Collect all simple paths to accepting cycles from the initial state
    dummy_model = object()  # not used by all_sequences when obs=None
    search = ExhaustiveSearch(dummy_model, propositions=None, num_loops=args.num_loops)
    seqs = search.all_sequences(ldba, ldba.initial_state, obs=None, num_loops=args.num_loops)

    payload = {
        'env': args.env,
        'formula': args.formula,
        'num_loops': args.num_loops,
        'initial_state': ldba.initial_state,
        'sequences': seqs,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"Saved {out_path} with {len(seqs)} sequences")


if __name__ == '__main__':
    main()


