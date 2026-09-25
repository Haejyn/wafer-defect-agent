"""히트맵이 결함을 제자리에서 짚는지 — 로트 test 불량 웨이퍼(종류마다 최대 100장)에서
pointing game: 히트맵 최댓값 위치(웨이퍼 안)가 계통 불량 덩어리(systematic_mask, 1 다이 여유) 위에 있나.
비교: 패치 kNN · Grad-CAM(b3 16×16 · b4 8×8) · 무작위 점(기준)."""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.heat import gradcam_heat, knn_heat  # noqa: E402
from wafer.locate import systematic_mask  # noqa: E402
from wafer.model import WaferCNN  # noqa: E402
from wafer.ood import feature_maps, patch_bank  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402

device = "cuda"
d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split_lot"]
model = WaferCNN(9).to(device)
model.load_state_dict(torch.load(ROOT / "runs/lot_s0.pt", map_location=device))
rng = np.random.default_rng(0)
normal = rng.choice(np.flatnonzero((split == 0) & (y == 0)), 6000, replace=False)
bank = patch_bank(feature_maps(model, X[normal], device)[0], 300_000)

te = np.flatnonzero(split == 2)
rows = {}
methods = ["knn_patch", "gradcam_b3", "gradcam_b4", "random_point"]
hits = {m: [] for m in methods}
per_class = {}
for c in range(1, 9):
    if CLASSES[c] == "Near-full":
        continue  # 거의 전면이 불량이라 어디를 짚어도 맞는다 — 채점에서 뺀다
    pick = rng.choice(te[y[te] == c], min(100, (y[te] == c).sum()), replace=False)
    ch = {m: [] for m in methods}
    for i in pick:
        m = X[i]
        target = ndimage.binary_dilation(systematic_mask(m), iterations=2)
        if not target.any():
            continue
        inside = m > 0
        heats = {"knn_patch": knn_heat(model, m, bank, device),
                 "gradcam_b3": gradcam_heat(model, m, device, "b3"),
                 "gradcam_b4": gradcam_heat(model, m, device, "b4")}
        for name, h in heats.items():
            h = np.where(inside, h, -1)
            yy, xx = np.unravel_index(np.argmax(h), h.shape)
            ch[name].append(bool(target[yy, xx]))
        iy, ix = np.nonzero(inside)
        k = rng.integers(len(iy))
        ch["random_point"].append(bool(target[iy[k], ix[k]]))
    per_class[CLASSES[c]] = {name: round(float(np.mean(v)), 3) for name, v in ch.items()} | {"n": len(ch["knn_patch"])}
    for name in methods:
        hits[name] += ch[name]
    print(CLASSES[c], per_class[CLASSES[c]], flush=True)

res = {"date": str(date.today()), "metric": "pointing game — 히트맵 최댓값이 계통 불량 덩어리(2픽셀 팽창) 위", "per_class": per_class,
       "overall": {name: round(float(np.mean(v)), 3) for name, v in hits.items()}, "n": len(hits["knn_patch"])}
(ROOT / "reports/heatmap_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res["overall"], ensure_ascii=False))
