"""조사 에이전트로 웨이퍼 한 장 조사. 도구 호출 기록(trace)과 최종 카드를 출력한다.
  python scripts/investigate.py --row 243725
  python scripts/investigate.py --row 26672 --exclude Donut     # Donut 을 빼고 학습한 모델로"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))
from wafer.agent import read_card  # noqa: E402
from wafer.investigate import Toolbox, investigate, load_data, needs_agent, ollama_llm  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402
from wafer.reader import Reader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--row", type=int, required=True, help="WM-811K 원본 행 번호 (라벨 없는 것도 된다)")
ap.add_argument("--model", default="qwen3.5:4b")
ap.add_argument("--exclude", choices=CLASSES[1:], help="이 패턴을 빼고 학습한 모델(lot_s0_ex-*)을 쓴다")
ap.add_argument("--hybrid", action="store_true", help="경고 · 맵 이상 · 1위 확률 < 0.7 일 때만 조사 에이전트, 나머지는 판독 파이프라인이 판정")
ap.add_argument("--json", action="store_true", help="전체 결과를 JSON 으로 출력")
args = ap.parse_args()

data = load_data()
ex = CLASSES.index(args.exclude) if args.exclude else None
reader = Reader("lot_s0" if ex is None else f"lot_s0_ex-{args.exclude}", ex, data["X"], data["y"], data["split"], "cuda", np.random.default_rng(0))
box = Toolbox(reader, data)
if args.hybrid:
    why = needs_agent(box.look(args.row))
    if not why:
        r = read_card(box.case_for_check(args.row), model=args.model)
        card = {**r["final"], "route": "판정"}
        print(json.dumps({"mode": "hybrid", "handled_by": "pipeline", "route": "판정", "seconds": r["seconds"], "pass_final": r["pass_final"], "card": card}, ensure_ascii=False, indent=2))
        sys.exit()
    print("혼합 모드: 조사 에이전트로 보낸다 —", "; ".join(why))
res = investigate(args.row, box.tools, ollama_llm(args.model), box.case_for_check(args.row))

if args.json:
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    sys.exit()
print(f'row {args.row}  정답 {box.case(args.row)["true"]}  모델 {args.model}  가까움 기준 {box.close_t:.3f}')
for s in res["trace"]:
    print(f'[{s["step"]}] {s["tool"]}({json.dumps({k: v for k, v in s["args"].items() if k != "card"}, ensure_ascii=False)}) {s["ms"]}ms')
    print("    " + s["result_summary"])
print()
print(f'종료 {res["end"]} · 도구 호출 {res["n_tool_calls"]} · submit {res["n_submits"]} · {res["seconds"]}초 (LLM {res["llm_seconds"]}초)')
for s in res["submits"]:
    print(f'  submit {s["n"]}: ' + ("통과" if s["ok"] else "거절 " + "; ".join(s["problems"])))
print(json.dumps(res["card"], ensure_ascii=False, indent=2))
