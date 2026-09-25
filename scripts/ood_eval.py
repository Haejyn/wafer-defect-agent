"""3단계 — 한 종류씩 학습에서 뺀 모델로, 뺀 패턴을 '처음 보는 것'으로 잡는지 잰다.
시험 세트: 아는 패턴(뺀 것 외 전부, none 포함) = 음성, 뺀 패턴 = 양성.
점수 4가지: MSP · 에너지 · 임베딩 kNN · 패치 kNN."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.metrics import auroc, recall_at_fpr  # noqa: E402
from wafer.model import WaferCNN  # noqa: E402
from wafer.ood import energy_score, feature_maps, knn_distance, msp_score, patch_bank, patch_scores  # noqa: E402
from wafer.prep import CLASS_TO_ID, CLASSES  # noqa: E402

DEFECTS = CLASSES[1:]
device = "cuda"
d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split_lot"]
rng = np.random.default_rng(0)

names = sys.argv[1:] or DEFECTS
rows = {}
out_path = ROOT / "reports/ood_leave_one_out.json"
if out_path.exists():
    rows = json.loads(out_path.read_text(encoding="utf-8")).get("rows", {})

for name in names:
    c = CLASS_TO_ID[name]
    tag = f"lot_s0_ex-{name}"
    ck = ROOT / f"runs/{tag}.pt"
    if not ck.exists():
        print(f"skip {name}: {ck.name} 없음")
        continue
    model = WaferCNN(len(CLASSES) - 1).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    t = np.load(ROOT / f"runs/{tag}_test.npz")
    te, logits = t["idx"], t["logits"]
    is_pos = y[te] == c

    # 학습 쪽 표본 — 클래스마다 최대 3,000장 (none 포함, 뺀 클래스 제외)
    tr = np.flatnonzero((split == 0) & (y != c))
    bank_idx = np.concatenate([rng.choice(tr[y[tr] == k], min(3000, (y[tr] == k).sum()), replace=False)
                               for k in range(len(CLASSES)) if k != c and (y[tr] == k).any()])
    tr_maps, tr_emb = feature_maps(model, X[bank_idx], device)
    te_maps, te_emb = feature_maps(model, X[te], device)

    scores = {
        "msp": msp_score(logits),
        "energy": energy_score(logits),
        "knn_emb": knn_distance(te_emb, tr_emb, k=5).numpy(),
        "knn_patch": patch_scores(te_maps, patch_bank(tr_maps, 400_000))[1],
    }
    row = {"n_pos": int(is_pos.sum()), "n_neg": int((~is_pos).sum())}
    for s_name, s in scores.items():
        r5, _ = recall_at_fpr(s[is_pos], s[~is_pos], 0.05)
        row[s_name] = {"auroc": round(auroc(s[is_pos], s[~is_pos]), 4), "recall_at_fpr5": round(r5, 4)}
    # 분류기가 뺀 패턴을 어느 아는 패턴으로 우겨 넣었나
    kept = [k for k in range(len(CLASSES)) if k != c]
    pred = np.array(kept)[logits[is_pos].argmax(1)]
    row["forced_into"] = {CLASSES[k]: int((pred == k).sum()) for k in np.unique(pred)}
    rows[name] = row
    print(name, json.dumps(row, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()

summary = {}
for s_name in ["msp", "energy", "knn_emb", "knn_patch"]:
    vals = [r[s_name]["auroc"] for r in rows.values()]
    rec = [r[s_name]["recall_at_fpr5"] for r in rows.values()]
    if vals:
        summary[s_name] = {"mean_auroc": round(float(np.mean(vals)), 4), "mean_recall_at_fpr5": round(float(np.mean(rec)), 4)}
out_path.write_text(json.dumps({"date": str(date.today()), "split": "lot", "negatives": "아는 패턴 전부(none 포함) 시험 웨이퍼",
                                "rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
