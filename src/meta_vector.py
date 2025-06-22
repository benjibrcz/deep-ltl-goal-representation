import os, sys, torch

sys.path.insert(0, os.path.abspath(os.path.join(__file__,"..","src")))

from utils.model_store    import ModelStore
from model.model          import build_model
from config               import model_configs
from ltl                  import FixedSampler
from envs                 import make_env
from sequence.search      import ExhaustiveSearch
from model.agent          import Agent
from steer_and_plot       import collect_acts  # reuse your existing helper

def load_model(env, exp, seed, formula):
    """Same loader as before."""
    store = ModelStore(env, exp, seed)
    store.load_vocab()
    status = store.load_training_status(map_location='cpu')
    cfg    = model_configs[env]
    sampler= FixedSampler.partial(formula)
    dummy  = make_env(env, sampler)
    model  = build_model(dummy, status, cfg)
    model.eval()
    return model

def main():
    ENV, EXP, SEED = "PointLtl2-v0", "big_test", 0
    LAYER = "actor.enc.3"
    EPS   = 50

    # 1) Load the generalist model
    model = load_model(ENV, EXP, SEED, "F yellow")
    layer = dict(model.named_modules())[LAYER]

    # 2) Gather all propositions except 'yellow'
    props = make_env(ENV, FixedSampler.partial("F yellow")).get_propositions()
    extras = [p for p in props if p != "yellow"]

    # 3) Build the list of meta‐formulas
    formulas = []
    for p in extras:
        formulas.append(f"F yellow & F {p}")    # find yellow + find p
        formulas.append(f"F yellow & G !{p}")   # find yellow + avoid p

    # 4) Collect activation vectors for each meta‐formula
    vs = []
    for f in formulas:
        print("Collecting for:", f)
        v = collect_acts(model, layer, ENV, f, EPS, SEED)
        vs.append(v)

    # 5) Meta‐average
    meta_vec = torch.stack(vs, dim=1).mean(dim=1)
    out_path = "v_meta_yellow.pt"
    torch.save(meta_vec, out_path)
    print("Saved meta‐averaged vector to", out_path)

if __name__ == "__main__":
    main()
