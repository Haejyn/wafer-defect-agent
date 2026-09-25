"""4단계 지도 — 81만 장 임베딩(runs/emb_all_lot_s0.npy)의 표본을 UMAP 2차원으로. CPU 전용.
표본: 라벨 불량 전부 + none 15,000 + 라벨 없음 40,000 + 라벨 없음 중 가장 낯선 2,000.
random_state 를 주면 병렬이 꺼져 매우 느려서(첫 시도 90분+ 중단) 주지 않는다 — 좌표는 실행마다 조금 다르다."""
import json
import time
from pathlib import Path

import numpy as np
import umap

ROOT = Path(__file__).resolve().parents[1]
E = np.load(ROOT / "runs/emb_all_lot_s0.npy", mmap_mode="r")
y = np.load(ROOT / "data/proc/all_meta.npz", allow_pickle=True)["y"]
unl, far = np.load(ROOT / "runs/unlabeled_novelty.npy")
unl = unl.astype(np.int64)
order = np.argsort(-far)
rng = np.random.default_rng(0)
sample = np.unique(np.concatenate([
    np.flatnonzero(y > 0),
    rng.choice(np.flatnonzero(y == 0), 15000, replace=False),
    rng.choice(unl, 40000, replace=False),
    unl[order[:2000]],
]))
t0 = time.time()
xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", n_jobs=-1).fit_transform(np.asarray(E[sample], dtype=np.float32))
nov = np.full(len(y), np.nan, dtype=np.float32)
nov[unl] = far
np.savez_compressed(ROOT / "runs/umap_sample.npz", rows=sample, xy=xy.astype(np.float32), y=y[sample], novelty=nov[sample])
info = {"points": int(len(sample)), "labeled_defect": int((y > 0).sum()), "none": 15000, "unlabeled_random": 40000,
        "unlabeled_most_novel": 2000, "minutes": round((time.time() - t0) / 60, 1)}
p = ROOT / "reports/map_search.json"
res = json.loads(p.read_text(encoding="utf-8"))
res["umap"] = info
p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(info)
