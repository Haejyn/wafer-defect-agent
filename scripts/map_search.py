"""4단계 — 81만 장 임베딩, 유사 사례 검색 정밀도@k, 라벨 없는 웨이퍼 중 '가장 낯선' 후보. (지도는 umap_map.py)
GPU 는 임베딩에만 쓴다(학습이 끝난 뒤 실행)."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.model import WaferCNN, one_hot  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402

device = "cuda"
X = np.load(ROOT / "data/proc/all64.npy", mmap_mode="r")
meta = np.load(ROOT / "data/proc/all_meta.npz", allow_pickle=True)
y_all = meta["y"]
lab = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
src_index, split = lab["src_index"], lab["split_lot"]  # 라벨 캐시 행 → 원본 행 번호

emb_path = ROOT / "runs/emb_all_lot_s0.npy"
if not emb_path.exists():
    model = WaferCNN(9).to(device)
    model.load_state_dict(torch.load(ROOT / "runs/lot_s0.pt", map_location=device))
    model.eval()
    embs = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(len(X), model.emb_dim))
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = one_hot(torch.from_numpy(np.ascontiguousarray(X[i:i + 4096])).to(device))
            embs[i:i + 4096] = model.embed(xb).half().cpu().numpy()
    embs.flush()
E = np.load(emb_path, mmap_mode="r")


def topk_same_label(q_vec, db_vec, q_y, db_y, k=5, chunk=4096):
    db = F.normalize(torch.from_numpy(np.asarray(db_vec, dtype=np.float32)).to(device), dim=1).half()
    hits = []
    for i in range(0, len(q_vec), chunk):
        q = F.normalize(torch.from_numpy(np.asarray(q_vec[i:i + chunk], dtype=np.float32)).to(device), dim=1).half()
        nn_idx = (q @ db.T).topk(k, dim=1).indices.cpu().numpy()
        hits.append((db_y[nn_idx] == q_y[i:i + chunk, None]).mean(1))
    return np.concatenate(hits)


# 검색 정밀도@5 — 질의: 로트 test 의 불량 웨이퍼, DB: 로트 train 의 라벨 웨이퍼 (로트가 겹치지 않는다)
q_rows = src_index[(split == 2) & (lab["y"] != 0)]
db_rows = src_index[split == 0]
res = {"date": str(date.today()), "k": 5, "queries": int(len(q_rows)), "db": int(len(db_rows))}
for name, qv, dv in [
    ("cnn_embedding", E[q_rows], E[db_rows]),
    ("raw_pixels_16x16", None, None),
]:
    if qv is None:  # 비교 기준: 16×16 으로 줄인 원-핫 픽셀 그대로
        def pix(rows):
            a = np.asarray(X[np.sort(rows)])[:, 2::4, 2::4]
            order = np.argsort(np.argsort(rows))
            return np.eye(3, dtype=np.float32)[a].reshape(len(rows), -1)[order]
        qv, dv = pix(q_rows), pix(db_rows)
    prec = topk_same_label(qv, dv, y_all[q_rows], y_all[db_rows])
    per = {CLASSES[c]: round(float(prec[y_all[q_rows] == c].mean()), 4) for c in range(1, 9) if (y_all[q_rows] == c).any()}
    res[name] = {"precision_at_5_macro": round(float(np.mean(list(per.values()))), 4), "per_class": per}
    print(name, res[name], flush=True)

# 라벨 없는 웨이퍼 중 라벨 있는 학습 웨이퍼와 가장 먼 것 (새 패턴 후보) — 상위 목록만 저장, 판정은 사람이
unl = np.flatnonzero(y_all < 0)
rng = np.random.default_rng(0)
bank = db_rows if len(db_rows) <= 60000 else rng.choice(db_rows, 60000, replace=False)
b = F.normalize(torch.from_numpy(np.asarray(E[bank], dtype=np.float32)).to(device), dim=1).half()
far = np.empty(len(unl), dtype=np.float32)
for i in range(0, len(unl), 8192):
    q = F.normalize(torch.from_numpy(np.asarray(E[unl[i:i + 8192]], dtype=np.float32)).to(device), dim=1).half()
    far[i:i + 8192] = (1 - (q @ b.T).topk(5, dim=1).values.float().mean(1)).cpu().numpy()
order = np.argsort(-far)
res["unlabeled_novelty"] = {"n_unlabeled": int(len(unl)), "top200_rows": unl[order[:200]].tolist(),
                            "top200_scores": np.round(far[order[:200]], 4).tolist(),
                            "score_percentiles": {p: round(float(np.percentile(far, p)), 4) for p in (50, 90, 99, 99.9)}}
np.save(ROOT / "runs/unlabeled_novelty.npy", np.stack([unl, far]).astype(np.float32))

(ROOT / "reports/map_search.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "unlabeled_novelty"}, ensure_ascii=False, indent=2))
# UMAP 지도는 scripts/umap_map.py (09-25 첫 시도: 17만 5천 점 + random_state 고정 → 병렬이 꺼져 90분 넘게 끝나지 않아 중단)
