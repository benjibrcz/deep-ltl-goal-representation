#!/usr/bin/env bash
export PYTHONPATH=src/

LAYERS=(
  env_net
  env_net.mlp.0
  env_net.mlp.1
  env_net.mlp.2
  env_net.mlp.3
  actor.enc.0
  actor.enc.3
  actor.mu.0
  ltl_net.rnn
)

for layer in "${LAYERS[@]}"; do
  out="dynamics_${layer//./_}.gif"
  echo "=== Rendering $layer → $out ==="
  python src/plot_single_world_dynamics.py \
    --formula 'GF blue & GF green' \
    --layer "$layer" \
    --world-idx 37 \
    --warmup 10 \
    --snaps 100 200 500 700 \
    --show-colors blue green \
    --gif \
    --fps 5 \
    --gif-dt 1 \
    --out "$out"
done
