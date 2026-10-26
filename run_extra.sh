#!/usr/bin/env bash
# Extra stages for the revised paper framing ("why coverage is attackable, and what
# a coverage-aware defense has to give up"). Same conventions as run_all.sh:
# finished runs are skipped, so you can stop and restart at any time.
#   bash run_extra.sh U        # one stage (U, H, I, T or K)
#   bash run_extra.sh          # all, in the order below
# Order of value for the paper:  U (does coverage buy utility?)  ->  H (does ASR fall
# as more clients cover the top band?)  ->  I (is ASR still rising?)  ->  T, K (CovGuard).
# New file prefixes (U_, H_, I_, T_, K_) do not collide with analyze_tables.py (A_, B_, C_, E_).
set -e
cd "$(dirname "$0")"
mkdir -p out
STAGE=${1:-ALL}
PY=hetlora_poison_v2.py
UT=hetlora_utility.py

run() {  # run <script> <name> <args...>
  local script=$1 name=$2; shift 2
  if [ -f "out/$name.csv" ]; then echo "skip $name"; return; fi
  echo "=== $name  ($(date '+%a %H:%M'))"
  python "$script" --out "out/$name.csv" "$@" 2>&1 | tee "out/$name.log"
}

# U: utility. Instruction tuning, clean, perplexity (lower is better). 3 seeds.
#    zeropad/median is a control: with the default ranks its median is exactly zero in every
#    band covered by fewer than half the clients (ranks above 8), so it behaves like rank-8 truncation.
#    Run `python hetlora_utility.py --smoke --rounds 5 --out out/u_smoke.csv` first.
if [[ $STAGE == ALL || $STAGE == U ]]; then
  for s in 0 1 2; do
    run $UT U_base_s$s   --seed $s --agg zeropad,coverage,svd --defense none,median --rounds 30
    run $UT U_ours_s$s   --seed $s --agg covguard            --gamma 0.5 --gamma 0.5 --gamma 0.5 --rounds 30
  done
  # a layout where high ranks should matter more: most clients are small, so zero-padding dilutes the top bands
  for s in 0 1 2; do
    run $UT U_skew_base_s$s --seed $s --ranks 4:12,8:4,16:2,32:2 --agg zeropad,coverage --defense none,median --rounds 30
    run $UT U_skew_ours_s$s --seed $s --ranks 4:12,8:4,16:2,32:2 --agg covguard --gamma 0.5 --gamma 0.5 --gamma 0.5 --rounds 30
  done
fi

# H: top-band coverage sweep, 5 seeds, 60 rounds. Only the number of top-rank clients (c) and
#    the rank-4 tier change (4:(10-c)); the 8 and 16 tiers stay fixed. c=2 is the default layout.
#    (run_all.sh stage B also moves clients out of the rank-16 tier, which this avoids.)
ranks_clean() { echo "4:$((10 - $1)),8:6,16:4,32:$1"; }
if [[ $STAGE == ALL || $STAGE == H ]]; then
  for s in 0 1 2 3 4; do for c in 1 2 4 6 8; do
    R=$(ranks_clean $c)
    run $PY H_cov${c}_base_s$s --seed $s --ranks $R --rounds 60 --agg coverage,zeropad --attack none,band --defense none,median --no_stats
    run $PY H_cov${c}_ours_s$s --seed $s --ranks $R --rounds 60 --agg covguard --gamma ${GAMMA_STAR:-0.5} --tau ${TAU_STAR:-6}         --attack none,band --no_stats
  done; done
fi

# I: longer runs of the main comparison (band attack), 3 seeds, 60 rounds: is ASR still rising?
if [[ $STAGE == ALL || $STAGE == I ]]; then
  for s in 0 1 2; do
    run $PY I_base_s$s --seed $s --rounds 60 --agg zeropad,coverage --attack none,band --defense none,median --no_stats
    run $PY I_ours_s$s --seed $s --rounds 60 --agg covguard --gamma ${GAMMA_STAR:-0.5} --tau ${TAU_STAR:-6}         --attack none,band,band_adapt --no_stats
  done
fi

# T: tune CovGuard on held-out seeds 100-102 only (never on the reporting seeds 0-2).
if [[ $STAGE == ALL || $STAGE == T ]]; then
  for s in 100 101 102; do for g in 0.25 0.5 1.0 1.5; do for t in 2 6 20; do
    run $PY T_g${g}_t${t}_s$s --seed $s --agg covguard --gamma $g --tau $t --attack none,band,band_adapt,backdoor --no_stats
  done; done; done
fi

# K: CovGuard ablation on the reporting seeds 0-2 with FROZEN settings from stage T.
#    Set GAMMA_STAR and TAU_STAR from the T results before running, e.g.
#      GAMMA_STAR=0.5 TAU_STAR=6 bash run_extra.sh K
#    tau 2 = shrinkage off at 2 top-band clients; gamma 1000 = clipping off; tau 20 = zero-padding weights.
if [[ $STAGE == K ]]; then
  : "${GAMMA_STAR:?set GAMMA_STAR from stage T}"; : "${TAU_STAR:?set TAU_STAR from stage T}"
  for s in 0 1 2; do
    run $PY K_both_s$s     --seed $s --agg covguard --gamma $GAMMA_STAR --tau $TAU_STAR --attack none,band,band_adapt,backdoor --no_stats
    run $PY K_noclip_s$s   --seed $s --agg covguard --gamma 1000        --tau $TAU_STAR --attack none,band,band_adapt,backdoor --no_stats
    run $PY K_noshrink_s$s --seed $s --agg covguard --gamma $GAMMA_STAR --tau 2         --attack none,band,band_adapt,backdoor --no_stats
    run $PY K_zpweight_s$s --seed $s --agg covguard --gamma $GAMMA_STAR --tau 20        --attack none,band,band_adapt,backdoor --no_stats
    run $PY K_zpnoclip_s$s --seed $s --agg covguard --gamma 1000        --tau 20        --attack none,band,band_adapt,backdoor --no_stats
  done
fi
echo "all requested runs finished"
