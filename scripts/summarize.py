"""결과 모으기 → reports/summary.json(화면 수치) · reports/leakage.json · reports/RESULTS.md"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.metrics import macro_f1  # noqa: E402

R = ROOT / "reports"


def load(p):
    q = R / p
    return json.loads(q.read_text(encoding="utf-8")) if q.exists() else None


d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
y, lot, sw = d["y"], d["lot"], d["split_wafer"]
cls = {s: load(f"cls/{s}_s0.json") for s in ["lot", "wafer", "official"]}

leak = {"date": str(date.today())}
for s, r in cls.items():
    if r:
        leak[f"{s}_test_macro_f1"] = r["test_macro_f1"]
if cls["lot"] and cls["wafer"]:
    leak["gap_wafer_minus_lot"] = round(cls["wafer"]["test_macro_f1"] - cls["lot"]["test_macro_f1"], 4)
# 웨이퍼 무작위 분할 안에서: 같은 로트 이웃이 학습에 있는 시험 웨이퍼 vs 없는 시험 웨이퍼
t = ROOT / "runs/wafer_s0_test.npz"
if t.exists():
    z = np.load(t)
    te, pred = z["idx"], z["logits"].argmax(1)
    train_lots = set(lot[sw == 0])
    has_mate = np.array([l in train_lots for l in lot[te]])
    for name, m in [("with_lotmate_in_train", has_mate), ("without_lotmate_in_train", ~has_mate)]:
        leak[f"wafer_split_{name}"] = {"n": int(m.sum()), "macro_f1": round(macro_f1(y[te][m], pred[m], 9), 4) if m.sum() else None}
(R / "leakage.json").write_text(json.dumps(leak, ensure_ascii=False, indent=2), encoding="utf-8")

ood = load("ood_leave_one_out.json")
ms = load("map_search.json")
ag = load("agent_eval.json")
best_ood = max(ood["summary"].items(), key=lambda kv: kv[1]["mean_auroc"]) if ood and ood.get("summary") else None

summary = {}
if cls["lot"]:
    summary["아는 패턴 macro-F1 (로트 분할 test)"] = f'{cls["lot"]["test_macro_f1"]:.3f}'
if cls["official"]:
    summary["나중 로트에서 하락 (같은 시기 검증 → 나중 로트)"] = f'{cls["official"]["val_macro_f1"]:.3f} → {cls["official"]["test_macro_f1"]:.3f}'
if "gap_wafer_minus_lot" in leak:
    summary["웨이퍼 무작위 분할의 부풀림 (시드 1)"] = f'{leak["gap_wafer_minus_lot"]:+.3f} F1'
if best_ood:
    summary["처음 보는 패턴 AUROC (8종 평균)"] = f'{best_ood[1]["mean_auroc"]:.3f}'
    summary["오경보 5%에서 처음 보는 패턴 재현율"] = f'{best_ood[1]["mean_recall_at_fpr5"] * 100:.0f}%'
if ms:
    summary["유사 사례 정밀도@5"] = f'{ms["cnn_embedding"]["precision_at_5_macro"]:.3f}'
if ag:
    s = ag["summary"]
    summary["판독 검사 통과 (첫 답 → 재질문 1회)"] = f'{s["pass_first"] * 100:.0f}% → {s["pass_final_after_1_retry"] * 100:.0f}%'
(R / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"leakage": leak, "summary": summary, "best_ood": best_ood}, ensure_ascii=False, indent=2))
