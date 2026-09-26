#!/usr/bin/env bash
# 고도화 v4 전체 — 단계마다 GPU 가 빌 때까지 기다린다(다른 세션 측정 보호). 학습은 run_queue.sh 가 한 번 더 GPU 를 본다.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
step() { bash scripts/wait_gpu.sh; echo "== $1 $(date +%H:%M)"; }
step "D ood v2 seed0";   $PY scripts/ood_eval_v2.py --seed 0 > runs/logs/ood_v2_s0.log 2>&1 || { echo FAIL D; tail -5 runs/logs/ood_v2_s0.log; exit 1; }
step "B catchup seed0";  $PY scripts/catchup.py --budget 0.01 --seed 0 > runs/logs/catchup_s0.log 2>&1 || { echo FAIL B; tail -5 runs/logs/catchup_s0.log; exit 1; }
for s in 1 2; do
  step "A main seed$s"; bash scripts/run_queue.sh "--split lot --seed $s" "--split wafer --seed $s" "--split official --seed $s" || { echo FAIL A main $s; exit 1; }
done
for s in 1 2; do
  step "A exclude seed$s"
  args=(); for c in Center Donut Edge-Loc Edge-Ring Loc Near-full Random Scratch; do args+=("--split lot --exclude $c --epochs 12 --seed $s"); done
  bash scripts/run_queue.sh "${args[@]}" || { echo FAIL A ex $s; exit 1; }
  step "A ood v2 seed$s"; $PY scripts/ood_eval_v2.py --seed $s > runs/logs/ood_v2_s$s.log 2>&1 || { echo FAIL A ood $s; exit 1; }
done
step "C resnet18"; bash scripts/run_queue.sh "--split lot --arch resnet18 --lr 1e-3" "--split wafer --arch resnet18 --lr 1e-3" || { echo FAIL C; exit 1; }
echo "V4 ALL DONE $(date +%H:%M)"
