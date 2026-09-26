#!/usr/bin/env bash
# 학습을 한 번에 하나씩. 시작 전에 다른 프로세스가 GPU 를 쓰고 있으면(메모리 1.5 GB 초과) 멈춘다.
# 사용: bash scripts/run_queue.sh "--split lot" "--split wafer" ...
cd "$(dirname "$0")/.."
mkdir -p runs/logs
for args in "$@"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -gt 1500 ]; then
    echo "STOP gpu busy ${used}MiB before: $args"
    exit 3
  fi
  name=$(echo "$args" | tr ' ' '_' | tr -d '-')
  echo "START $args"
  PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/train_cls.py $args > "runs/logs/$name.log" 2>&1
  code=$?
  echo "DONE $args exit=$code $(grep -o '"test_macro_f1": [0-9.]*\|"val_macro_f1": [0-9.]*' "runs/logs/$name.log" | tr '\n' ' ')"
  [ $code -ne 0 ] && tail -5 "runs/logs/$name.log" && exit $code
done
echo "QUEUE FINISHED"
