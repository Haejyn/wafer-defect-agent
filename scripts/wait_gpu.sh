#!/usr/bin/env bash
# GPU 가 빌 때까지 기다린다 — Ollama 에 올라간 모델이 없고 GPU 메모리가 1.5 GB 미만인 상태가 1분 간격 3번 연속.
# 다른 세션의 측정을 오염시키지 않으려고 쓴다. 사용: bash scripts/wait_gpu.sh && 다음 작업
ok=0
while [ $ok -lt 3 ]; do
  models=$(ollama ps 2>/dev/null | tail -n +2 | grep -c .)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$models" -eq 0 ] && [ "$used" -lt 1500 ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60
done
echo "GPU READY $(date +%H:%M)"
