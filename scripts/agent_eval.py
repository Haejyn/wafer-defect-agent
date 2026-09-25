"""5단계 — 판독 에이전트 평가. 로트 test 에서 사례를 뽑아 판독 카드를 받고 자동 검사 통과율을 잰다.
사례 A: 전체 모델(lot_s0)이 아는 패턴을 판정한 웨이퍼 — 패턴마다 N장
사례 B: 한 종류를 뺀 모델이 그 '뺀 패턴' 웨이퍼를 받은 경우 — 처음 보는 패턴 경고가 켜져야 한다
사용: python scripts/agent_eval.py --per-class 6 --model qwen3.5:4b"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.agent import read_card  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=6)
ap.add_argument("--unknown-per-class", type=int, default=3)
ap.add_argument("--model", default="qwen3.5:4b")
ap.add_argument("--out", default="agent_eval")
args = ap.parse_args()

device = "cuda"
d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split_lot"]
rng = np.random.default_rng(0)

from wafer.reader import Reader  # noqa: E402

te = np.flatnonzero(split == 2)
results = []
main = Reader("lot_s0", None, X, y, split, device, rng)
for c in range(1, 9):
    pick = rng.choice(te[y[te] == c], min(args.per_class, (y[te] == c).sum()), replace=False)
    for i in pick:
        r = read_card(main.case(int(i)), model=args.model)
        r["set"] = "known"
        results.append(r)
        print(f'[known] {CLASSES[c]:9s} -> {r["case"]["pattern"]:10s} loc={r["case"]["location"]} first={r["pass_first"]} final={r["pass_final"]} {r["seconds"]}s', flush=True)
del main
torch.cuda.empty_cache()

for c in range(1, 9):
    name = CLASSES[c]
    if not (ROOT / f"runs/lot_s0_ex-{name}.pt").exists():
        continue
    rd = Reader(f"lot_s0_ex-{name}", c, X, y, split, device, rng)
    pick = rng.choice(te[y[te] == c], min(args.unknown_per_class, (y[te] == c).sum()), replace=False)
    for i in pick:
        r = read_card(rd.case(int(i)), model=args.model)
        r["set"] = f"unseen:{name}"
        results.append(r)
        print(f'[unseen {name}] -> {r["case"]["pattern"]:10s} flag={r["case"]["unknown_flag"]} first={r["pass_first"]} final={r["pass_final"]}', flush=True)
    del rd
    torch.cuda.empty_cache()


def rate(rs, key):
    return round(float(np.mean([r[key] for r in rs])), 4) if rs else None


known = [r for r in results if r["set"] == "known"]
unseen = [r for r in results if r["set"] != "known"]
summary = {
    "date": str(date.today()), "model": args.model, "n": len(results),
    "pass_first": rate(results, "pass_first"), "pass_final_after_1_retry": rate(results, "pass_final"),
    "known": {"n": len(known), "pass_first": rate(known, "pass_first"), "pass_final": rate(known, "pass_final"),
              "classifier_correct": rate([{"x": r["case"]["classifier_pred"] == r["case"]["true"]} for r in known], "x")},
    "unseen": {"n": len(unseen), "pass_first": rate(unseen, "pass_first"), "pass_final": rate(unseen, "pass_final"),
               "flagged_unknown": rate([{"x": r["case"]["unknown_flag"]} for r in unseen], "x")},
    "problem_counts_first_try": {},
    "mean_seconds": round(float(np.mean([r["seconds"] for r in results])), 1),
}
for r in results:
    for p in r["attempts"][0]["problems"]:
        key = p.split(":")[0].split("(")[0].strip()
        summary["problem_counts_first_try"][key] = summary["problem_counts_first_try"].get(key, 0) + 1
out = ROOT / "reports" / f"{args.out}.json"
out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
