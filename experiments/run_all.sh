#!/usr/bin/env bash
# Runs every experiment one after another on the GPU. Logs go to ../results/logs/.
# About 2 hours 15 minutes in total on an RTX 3050 Ti laptop GPU.
set -u
cd "$(dirname "$0")"
mkdir -p ../results/logs
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

run() {
  local name=$1; shift
  echo "=== $(date '+%H:%M:%S') START $name"
  python "$@" > "../results/logs/$name.log" 2>&1
  echo "=== $(date '+%H:%M:%S') END   $name (exit $?)"
}

run 1_eval_original      eval_original.py
run 2_baselines_cifar10  baselines.py cifar10
run 3_baselines_cifake   baselines.py cifake
run 4_sidenet_cifake     train_sidenet.py cifake --epochs 1 --seed 0
run 5_sidenet_cifar10    train_sidenet.py cifar10 --epochs 5 --seed 0   # about 1 hour on the laptop GPU
echo "=== ALL DONE"
