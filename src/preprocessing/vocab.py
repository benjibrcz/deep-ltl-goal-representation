from ltl.logic import Assignment

FIXED = ['PAD', 'EPSILON', 'NULL']
VOCAB = {k: i for i, k in enumerate(FIXED)}


def reset_vocab():
    global VOCAB
    VOCAB.clear()
    VOCAB.update({k: i for i, k in enumerate(FIXED)})


def init_vocab(assignments: list[Assignment]):
    for a in assignments:
        VOCAB[a.to_frozen()] = len(VOCAB)
