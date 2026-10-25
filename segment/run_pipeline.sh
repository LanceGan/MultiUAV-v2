#!/bin/bash
# Full downstream re-run after Bug 3 (region scaling) + Bug 4 (noise) fixes.
#
# Training: TD3 for N=2/3/4 (core single-segment policy) + DDPG/SAC for N=2 only
# (single-segment algorithm comparison).  5 processes in parallel.
# Downstream: edge weights -> ALNS -> scheduler -> eps scan (per N); compare+plot
# for N=2.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="F:/anaconda3/envs/pytorch/python.exe"
export OMP_NUM_THREADS=5
mkdir -p results/logs

run() { local name="$1"; shift; echo "==== $name ===="; "$@" 2>&1 | tee "results/logs/retrain_${name}.log"; echo "---- exit ${PIPESTATUS[0]} ----"; }

echo "############ [1/4] train (5 procs parallel) ############"
$PY segment/train.py --uav_num 2 --total_episode 3000 --algo TD3  --model_tag TD3  --c_lr 3e-4 --policy_delay 3 --no_plot > results/logs/retrain_TD3_uav2.log  2>&1 &
P1=$!
$PY segment/train.py --uav_num 2 --total_episode 3000 --algo DDPG --model_tag DDPG --no_plot                                   > results/logs/retrain_DDPG_uav2.log 2>&1 &
P2=$!
$PY segment/train.py --uav_num 2 --total_episode 3000 --algo SAC  --model_tag SAC  --no_plot                                   > results/logs/retrain_SAC_uav2.log  2>&1 &
P3=$!
$PY segment/train.py --uav_num 3 --total_episode 3000 --algo TD3  --model_tag TD3  --c_lr 3e-4 --policy_delay 3 --no_plot > results/logs/retrain_TD3_uav3.log  2>&1 &
P4=$!
$PY segment/train.py --uav_num 4 --total_episode 3000 --algo TD3  --model_tag TD3  --c_lr 3e-4 --policy_delay 3 --no_plot > results/logs/retrain_TD3_uav4.log  2>&1 &
P5=$!
wait $P1 $P2 $P3 $P4 $P5
echo "train exit codes: TD3-uav2=$? (see logs); done"

echo "############ [2/4] copy TD3 -> allpairs ############"
for n in 2 3 4; do
  cp "results/models/segment/uav${n}_TD3_best.pth" "results/models/segment/uav${n}_allpairs_best.pth"
  echo "copied uav${n}"
done

echo "############ [3/4] per-N downstream ############"
for n in 2 3 4; do
  run precompute_uav${n}  $PY segment/precompute.py --uav_num $n
  run alns_uav${n}        $PY segment/alns.py --uav_num $n --iters 2000 --iterative
  run scheduler_uav${n}   $PY segment/scheduler.py --uav_num $n
  run scan_eps_uav${n}    $PY segment/scan_eps.py --uav_num $n
done

echo "############ [4/4] compare + plot (N=2) ############"
run compare       $PY segment/compare.py --uav_num 2
run plot_compare  $PY segment/plot_compare.py --uav_num 2

echo "############ ALL DONE ############"
