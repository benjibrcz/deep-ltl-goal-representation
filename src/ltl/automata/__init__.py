from contextlib import nullcontext

from utils import memory, timeit
from .rabinizer import run_rabinizer
from .ldba import LDBA, LDBATransition
from .ldba_sequence import LDBASequence


@memory.cache
def ltl2ldba(formula: str, propositions: list[str] = None, simplify_labels=True, print_time=False) -> LDBA:
    """Converts an LTL formula to an LDBA using the rabinizer tool."""
    if formula == 'fixedparity':
        # Construct an LDBA matching a DFA for a parity task (see Yalcinkaya et al. 2024)
        return LDBA.parity_automaton()
    from ltl.hoa import HOAParser
    with timeit(f'Converting LTL formula "{formula}" to LDBA') if print_time else nullcontext():
        hoa = run_rabinizer(formula)
    return HOAParser(formula, hoa, propositions, simplify_labels=simplify_labels).parse_hoa()


__all__ = ['LDBASequence', 'LDBA', 'LDBATransition', 'ltl2ldba']
