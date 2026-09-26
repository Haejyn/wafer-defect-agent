"""처음 보는 패턴 v2 (고도화 D·A) — 한 종류씩 뺀 모델에서
점수: MSP · 에너지 · 임베딩 kNN · 마할라노비스
음성 두 가지: (all) 아는 패턴 전부(none 포함) · (defect) 아는 불량만 — none 이 쉬운 음성이라 부풀리는지 본다
웨이퍼별 점수를 runs/ood_scores_s{seed}_ex-{name}.npz 로 남겨 부트스트랩·시드 합치기에 쓴다.
사용: python scripts/ood_eval_v2.py --seed 0 [패턴 …]"""
import argparse
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
from wafer.ood import energy_score, feature_maps, knn_distance, mahalanobis_score, msp_score  # noqa: E402
from wafer.prep import CLASS_TO_ID, CLASSES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("names", nargs="*")
args = ap.parse_args()
device = "cuda"
d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split_lot"]
out_path = ROOT / f"reports/ood_v2_s{args.seed}.json"
rows = json.loads(out_path.read_text(encoding="utf-8"))["rows"] if out_path.exists() else {}
SCORES = ["msp", "energy", "knn_emb", "mahalanobis"]

for name in args.names or CLASSES[1:]:
    c = CLASS_TO_ID[name]
    tag = f"lot_s{args.seed}_ex-{name}"
    if not (ROOT / f"runs/{tag}.pt").exists():
        print(f"skip {name}: {tag}.pt 없음")
        continue
    rng = np.random.default_rng(args.seed)
    model = WaferCNN(len(CLASSES) - 1).to(device)
    model.load_state_dict(torch.load(ROOT / f"runs/{tag}.pt", map_location=device))
    t = np.load(ROOT / f"runs/{tag}_test.npz")
    te, logits = t["idx"], t["logits"]
    is_pos = y[te] == c
    tr = np.flatnonzero((split == 0) & (y != c))
    bank_idx = np.concatenate([rng.choice(tr[y[tr] == k], min(3000, (y[tr] == k).sum()), replace=False)
                               for k in range(len(CLASSES)) if k != c and (y[tr] == k).any()])
    _, tr_emb = feature_maps(model, X[bank_idx], device)
    _, te_emb = feature_maps(model, X[te], device)
    scores = {
        "msp": msp_score(logits),
        "energy": energy_score(logits),
        "knn_emb": knn_distance(te_emb, tr_emb, k=5).numpy(),
        "mahalanobis": mahalanobis_score(te_emb, tr_emb, y[bank_idx]),
    }
    np.savez_compressed(ROOT / f"runs/ood_scores_s{args.seed}_ex-{name}.npz", idx=te, is_pos=is_pos, **scores)
    row = {"n_pos": int(is_pos.sum())}
    for neg_name, neg in [("all", ~is_pos), ("defect", (~is_pos) & (y[te] != 0))]:
        row[f"n_neg_{neg_name}"] = int(neg.sum())
        for s in SCORES:
            r5, _ = recall_at_fpr(scores[s][is_pos], scores[s][neg], 0.05)
            row.setdefault(neg_name, {})[s] = {"auroc": round(auroc(scores[s][is_pos], scores[s][neg]), 4), "recall_at_fpr5": round(r5, 4)}
    rows[name] = row
    print(name, json.dumps({k: {s: v["auroc"] for s, v in row[k].items()} for k in ("all", "defect")}), flush=True)
    del model
    torch.cuda.empty_cache()

summary = {}
for neg_name in ("all", "defect"):
    for s in SCORES:
        vals = [r[neg_name][s] for r in rows.values()]
        if vals:
            summary.setdefault(neg_name, {})[s] = {"mean_auroc": round(float(np.mean([v["auroc"] for v in vals])), 4),
                                                   "mean_recall_at_fpr5": round(float(np.mean([v["recall_at_fpr5"] for v in vals])), 4)}
out_path.write_text(json.dumps({"date": str(date.today()), "seed": args.seed, "rows": rows, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
