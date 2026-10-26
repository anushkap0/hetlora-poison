#!/usr/bin/env bash
# Runs the experiments in priority order. Finished runs are skipped, so you can
# stop (Ctrl+C) and restart at any time; only the run in progress is lost.
#   bash run_all.sh            # everything, in order
#   bash run_all.sh A          # one stage only (A, B, C, D, E or F)
# Order: A main table -> B coverage sweep -> C CovGuard ablation ->
#        E robustness settings -> D extra seeds -> F Phi-3 Mini (slow)
set -e
cd "$(dirname "$0")"
mkdir -p out
STAGE=${1:-ALL}
PY=hetlora_poison_v2.py

run() {  # run <name> <args...>
  local name=$1; shift
  if [ -f "out/$name.csv" ]; then echo "skip $name"; return; fi
  echo "=== $name  ($(date '+%a %H:%M'))"
  python "$PY" --out "out/$name.csv" "$@" 2>&1 | tee "out/$name.log"
}

ranks_for() {  # 20 clients, $1 of them at rank 32
  case $1 in
    2) echo 4:8,8:6,16:4,32:2 ;;
    4) echo 4:8,8:6,16:2,32:4 ;;
    6) echo 4:8,8:4,16:2,32:6 ;;
    8) echo 4:6,8:4,16:2,32:8 ;;
  esac
}

# A: main table, 3 seeds (most important)
main_runs() {
  local s=$1
  run A_base_s$s   --seed $s --agg zeropad,coverage --attack none,backdoor,band --defense none,clip,median
  run A_svd_s$s    --seed $s --agg svd              --attack none,backdoor,band --defense none
  run A_ours_s$s   --seed $s --agg covguard         --attack none,backdoor,band,band_adapt
  run A_horus_s$s  --seed $s --agg horus            --attack none,backdoor,band
  run A_adapt_s$s  --seed $s --agg coverage         --attack band_adapt         --defense none,clip,median
}
if [[ $STAGE == ALL || $STAGE == A ]]; then
  for s in 0 1 2; do main_runs $s; done
fi

# B: coverage sweep, 3 seeds (the main figure)
if [[ $STAGE == ALL || $STAGE == B ]]; then
  for s in 0 1 2; do for c in 2 4 6 8; do
    R=$(ranks_for $c)
    run B_cov${c}_base_s$s --seed $s --ranks $R --agg coverage --attack none,band --defense median
    run B_cov${c}_ours_s$s --seed $s --ranks $R --agg covguard --attack none,band
  done; done
fi

# C: CovGuard settings, seed 0 (ablation)
if [[ $STAGE == ALL || $STAGE == C ]]; then
  for t in 2 4 10 20; do run C_tau${t}_s0   --seed 0 --agg covguard --attack none,band,band_adapt --tau $t --no_stats; done
  # gamma 1000 = clipping off, shrinkage only
  for g in 0.5 1.0 2.0 3.0 1000; do run C_gamma${g}_s0 --seed 0 --agg covguard --attack none,band,band_adapt --gamma $g --no_stats; done
fi

# E: robustness settings, 2 seeds, key methods only
#    name pattern E_<setting>_<group>_s<seed>; <setting> must not contain "_"
setting_runs() {  # setting_runs <setting> <seed> <extra args...>
  local st=$1 s=$2; shift 2
  run E_${st}_zp_s$s    --seed $s "$@" --agg zeropad  --attack none,backdoor,band --defense none   --no_stats
  run E_${st}_cov_s$s   --seed $s "$@" --agg coverage --attack none,backdoor,band --defense median --no_stats
  run E_${st}_ours_s$s  --seed $s "$@" --agg covguard --attack none,backdoor,band,band_adapt --no_stats
  run E_${st}_horus_s$s --seed $s "$@" --agg horus --attack none,backdoor,band --no_stats
}
if [[ $STAGE == ALL || $STAGE == E ]]; then
  for s in 0 1; do
    setting_runs noniid0.5 $s --alpha 0.5                 # moderate label skew
    setting_runs noniid0.1 $s --alpha 0.1                 # strong label skew
    setting_runs twomal    $s --n_mal 2                   # both top-band clients malicious
    setting_runs partial   $s --clients_per_round 10      # half the clients per round
    setting_runs qwen      $s --model Qwen/Qwen2.5-0.5B   # second model
    setting_runs agnews    $s --task agnews               # second task (4 classes)
  done
fi

# D: two more seeds for the main table, if time allows
if [[ $STAGE == ALL || $STAGE == D ]]; then
  for s in 3 4; do main_runs $s; done
fi
# F: Phi-3 Mini (the CovRank-FL paper's model). Slow: about 10x a SmolLM2 run.
#    Seed 0, 20 rounds, key methods only. Lower --bs to 4 if you run out of memory.
if [[ $STAGE == ALL || $STAGE == F ]]; then
  P3="--model microsoft/Phi-3-mini-4k-instruct --targets qkv_proj --dtype bfloat16 --bs 8 --rounds 20 --no_stats"
  run E_phi3_zp_s0   --seed 0 $P3 --agg zeropad  --attack none,band --defense none
  run E_phi3_cov_s0  --seed 0 $P3 --agg coverage --attack none,band --defense median
  run E_phi3_ours_s0 --seed 0 $P3 --agg covguard --attack none,band,band_adapt
fi
echo "all requested runs finished"
