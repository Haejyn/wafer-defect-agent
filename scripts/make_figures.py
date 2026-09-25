"""README·PDF 그림 — docs/img/ood.png (처음 보는 패턴 AUROC 비교) · docs/img/umap.png (지도 정적판)"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs/img"
IMG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b9b8b3", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # 기본 팔레트 고정 순서

# 1) 처음 보는 패턴 — 뺀 패턴별 AUROC, 분류기 확신도(MSP) 대 임베딩 kNN
r = json.loads((ROOT / "reports/ood_leave_one_out.json").read_text(encoding="utf-8"))
names = list(r["rows"])
msp = [r["rows"][n]["msp"]["auroc"] for n in names]
knn = [r["rows"][n]["knn_emb"]["auroc"] for n in names]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8, 3.2))
w = 0.36
ax.bar(x - w / 2 - 0.01, msp, w, color=ORANGE, label=f'분류기 확신도 (평균 {r["summary"]["msp"]["mean_auroc"]:.2f})')
ax.bar(x + w / 2 + 0.01, knn, w, color=BLUE, label=f'임베딩 kNN 거리 (평균 {r["summary"]["knn_emb"]["mean_auroc"]:.2f})')
ax.axhline(0.5, color="#52514e", lw=1, ls=(0, (3, 3)), label="우연 수준 (0.5)")
ax.set_xticks(x, names, fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("AUROC (처음 보는 패턴 가려내기)", fontsize=9)
ax.grid(axis="y", color="#ecebe7", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.14))
fig.savefig(IMG / "ood.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 2) 지도 — UMAP 표본, 라벨 없음·none 은 회색, 불량 8종은 고정 순서 색
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from wafer.prep import CLASSES  # noqa: E402

um = np.load(ROOT / "runs/umap_sample.npz")
xy, y = um["xy"], um["y"]
fig, ax = plt.subplots(figsize=(7, 5.2))
bg = y <= 0
ax.scatter(xy[bg, 0], xy[bg, 1], s=0.4, c=GRAY, alpha=0.35, linewidths=0, label=f"none·라벨 없음 ({bg.sum():,})")
for k in range(1, 9):
    m = y == k
    ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=CAT[k - 1], alpha=0.85, linewidths=0, label=f"{CLASSES[k]} ({m.sum():,})")
ax.set_axis_off()
ax.legend(frameon=False, fontsize=8, markerscale=6, loc="center left", bbox_to_anchor=(1, 0.5))
fig.savefig(IMG / "umap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("ok")
