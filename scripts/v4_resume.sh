#!/usr/bin/env bash
# 고도화 재개 (09-26 12:15 다른 세션과 GPU 가 겹쳐 멈춘 뒤) — B v2 → A → C. 사용자가 다른 세션이 끝났다고 알려 준 뒤 실행.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
step() { bash scripts/wait_gpu.sh; echo "== $1 $(date +%H:%M)"; }
step "B v2 catchup gentle"; $PY scripts/catchup.py --budget 0.01 --seed 0 --lr 1e-4 --train-last --tag _v2 > runs/logs/catchup_s0_v2.log 2>&1 || { echo FAIL B; tail -5 runs/logs/catchup_s0_v2.log; exit 1; }
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
