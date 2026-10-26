#!/usr/bin/env bash
# Utility experiment: Alpaca-GPT4, response perplexity, trigger forces a fixed response.
#   bash run_utility.sh 0                      # one seed (about 10 combinations of 30 rounds)
#   for s in 0 1 2; do nohup bash run_utility.sh $s > util_s$s.log 2>&1 & done
# Other model:  MODEL_ARGS="--model microsoft/Phi-3-mini-4k-instruct --targets qkv_proj --dtype bfloat16" bash run_utility.sh 0
# Finished runs are skipped. Outputs: out/U_*.csv (column "ppl" instead of "acc"); then: python analyze_utility.py
set -e
cd "$(dirname "$0")"
mkdir -p out
S=${1:?usage: bash run_utility.sh <seed>}
PY=hetlora_poison_v4.py
COMMON="--task alpaca --seed $S --rounds 30 --bs 8 --n_test 500 --gamma 0.5 --no_stats ${MODEL_ARGS:-}"

run() {  # run <name> <args...>
  local name=$1; shift
  if [ -f "out/$name.csv" ]; then echo "skip $name"; return; fi
  echo "=== $name  ($(date '+%a %H:%M'))"
  python "$PY" --out "out/$name.csv" "$@" 2>&1 | tee "out/$name.log"
}

# quality (no attack) and security (band attack), baselines with and without median
run U_base_s$S $COMMON --agg zeropad,coverage --attack none,band --defense none,median
run U_ours_s$S $COMMON --agg covguard         --attack none,band
echo "seed $S finished"
