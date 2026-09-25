"""웨이퍼 한 장 판독. 결과: 판독 카드(JSON) + 그림(원본 · Grad-CAM · 유사 사례 · 카드) reports/read/<이름>.png
  python scripts/read_wafer.py --row 123456          # WM-811K 원본 행 번호 (라벨 없는 것도 된다)
  python scripts/read_wafer.py --npy my_wafer.npy    # 값 0 밖 · 1 정상 · 2 불량 2차원 배열, 크기 자유
  python scripts/read_wafer.py --novel 3             # 라벨 없는 웨이퍼 중 '가장 낯선' 3번째"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.agent import CAUSES, read_card  # noqa: E402
from wafer.heat import gradcam_heat  # noqa: E402
from wafer.prep import CLASSES, resize_nearest  # noqa: E402
from wafer.reader import Reader  # noqa: E402

ap = argparse.ArgumentParser()
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--row", type=int)
g.add_argument("--npy")
g.add_argument("--novel", type=int, help="라벨 없는 웨이퍼 중 낯선 순위(1부터)")
ap.add_argument("--model", default="qwen3.5:4b")
args = ap.parse_args()

d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
reader = Reader("lot_s0", None, d["X"], d["y"], d["split_lot"])
if args.npy:
    m, name, true = resize_nearest(np.load(args.npy)), Path(args.npy).stem, "?"
else:
    allX = np.load(ROOT / "data/proc/all64.npy", mmap_mode="r")
    y_all = np.load(ROOT / "data/proc/all_meta.npz", allow_pickle=True)["y"]
    if args.novel:
        unl, far = np.load(ROOT / "runs/unlabeled_novelty.npy")
        row = int(unl[np.argsort(-far)[args.novel - 1]])
        name = f"novel{args.novel}_row{row}"
    else:
        row = args.row
        name = f"row{row}"
    m = np.asarray(allX[row])
    true = CLASSES[y_all[row]] if y_all[row] >= 0 else "라벨 없음"

case = reader.case_from_map(m, true=true)
result = read_card(case, model=args.model)
out = ROOT / "reports/read"
out.mkdir(parents=True, exist_ok=True)
(out / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

# 그림
cmap = ListedColormap(["#ffffff", "#cfd8d3", "#d6453d"])
heat = gradcam_heat(reader.model, m, reader.device, "b3")
fig = plt.figure(figsize=(12, 4.2))
gs = fig.add_gridspec(2, 8, height_ratios=[3, 1])
ax = fig.add_subplot(gs[0, 0:2]); ax.imshow(m, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax.set_title(f"원본 · {true}", fontsize=10); ax.axis("off")
ax = fig.add_subplot(gs[0, 2:4]); ax.imshow(m, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
rgba = plt.get_cmap("plasma")(heat); rgba[..., 3] = np.where(m > 0, np.clip((heat - 0.45) / 0.55, 0, 1) * 0.85, 0)
ax.imshow(rgba, interpolation="bilinear"); ax.set_title("Grad-CAM", fontsize=10); ax.axis("off")
for k, s in enumerate(case["similar"][:4]):
    ax = fig.add_subplot(gs[1, k]); ax.imshow(reader.X[s["row"]], cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax.axis("off")
    ax.set_title(f'{s["pattern"]} {s["similarity"]:.2f}', fontsize=7)
f = result["final"]
cause_names = {c["cause_id"]: c["cause"] for lst in CAUSES["patterns"].values() for c in lst}
lines = [f'{case["pattern"]}' + ("  [처음 보는 패턴]" if case["unknown_flag"] else ""),
         f'확신도 {case["confidence"]:.2f} · 이상 점수 백분위 {case["ood_percentile"]:.0f} · 위치 {case["location"]}',
         "검사 " + ("통과" if result["pass_final"] else "실패") + (" (첫 답)" if result["pass_first"] else " (재질문 뒤)"), "",
         "원인 후보: " + (", ".join(cause_names[c] for c in f["cause_ids"] if c in cause_names) or "표에 없음 — 사람 확인"), ""]
lines += [f"{i + 1}. {s}" for i, s in enumerate(f["check_order"])] + ["", f["summary"]]
ax = fig.add_subplot(gs[:, 4:]); ax.axis("off")
ax.text(0, 1, "\n".join(textwrap.fill(t, 46) if t else "" for t in lines), va="top", fontsize=9.5, family="Malgun Gothic")
plt.rcParams["font.family"] = "Malgun Gothic"
for a in fig.axes:
    a.title.set_fontfamily("Malgun Gothic")
fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")
print(json.dumps({"out": str(out / f"{name}.png"), "pattern": case["pattern"], "unknown": case["unknown_flag"],
                  "location": case["location"], "pass_first": result["pass_first"], "pass_final": result["pass_final"],
                  "card": f}, ensure_ascii=False, indent=2))
