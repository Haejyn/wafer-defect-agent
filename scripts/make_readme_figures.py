"""README 그림 — docs/img/{split,ood,heat,agent,umap}.png. 원자료는 reports/*.json · runs/ (학습·평가를 먼저 돌린다).
Pretendard 가 없으면 build/fonts 에 받는다. 사용: python scripts/make_readme_figures.py"""
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.prep import CLASSES  # noqa: E402

OUT = ROOT / "docs/img"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "reports"
BLACK, GRAY, LIGHT, BLUE, RED = "#111111", "#6b6b6b", "#d9d9d9", "#1f4e79", "#b03a2e"
WAFER = ListedColormap(["#ffffff", "#d9d9d9", RED])
PRETENDARD_URL = "https://github.com/orioncactus/pretendard/releases/download/v1.3.9/Pretendard-1.3.9.zip"


def setup_style():
    fonts = ROOT / "build/fonts"
    files = list(fonts.glob("Pretendard-*.ttf")) if fonts.exists() else []
    if not files:
        fonts.mkdir(parents=True, exist_ok=True)
        archive = fonts / "Pretendard.zip"
        urllib.request.urlretrieve(PRETENDARD_URL, archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith("public/static/alternative/") and name.endswith(".ttf"):
                    (fonts / Path(name).name).write_bytes(z.read(name))
        files = list(fonts.glob("Pretendard-*.ttf"))
    for f in files:
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Pretendard", "font.size": 9, "axes.unicode_minus": False,
        "axes.linewidth": 0.8, "axes.edgecolor": BLACK, "axes.labelcolor": BLACK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.color": BLACK, "ytick.color": BLACK,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "lines.linewidth": 1.0,
    })


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="semibold")


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def split_fig():
    cls = {s: load(f"cls/{s}_s0.json") for s in ["lot", "wafer", "official"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    names = ["로트 단위", "웨이퍼 무작위", "공식(나중 로트)"]
    val = [cls[s]["val_macro_f1"] for s in ["lot", "wafer", "official"]]
    test = [cls[s]["test_macro_f1"] for s in ["lot", "wafer", "official"]]
    x = np.arange(3)
    ax.bar(x - 0.19, val, 0.36, color="white", edgecolor=BLACK, lw=0.8, label="검증")
    ax.bar(x + 0.19, test, 0.36, color=BLUE, edgecolor=BLACK, lw=0.8, label="시험")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.01))
    panel(ax, "a")
    ax = axes[1]
    rec = cls["lot"]["test_recall"]
    k = [c for c in CLASSES]
    ax.barh(np.arange(len(k)), [rec[c] for c in k], color=BLUE, edgecolor=BLACK, lw=0.6, height=0.6)
    ax.set_yticks(np.arange(len(k)), k, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("재현율 (로트 단위 시험)")
    panel(ax, "b")
    fig.tight_layout()
    fig.savefig(OUT / "split.png")
    plt.close(fig)


def ood_fig():
    r = load("ood_leave_one_out.json")
    names = list(r["rows"])
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    for ax, key, label, letter in [(axes[0], "auroc", "AUROC", "a"), (axes[1], "recall_at_fpr5", "재현율 (오경보 5 %)", "b")]:
        for off, s, name, color, hatch in [(-0.27, "msp", "분류기 확신도", "white", "///"), (0, "energy", "에너지", "white", "..."),
                                           (0.27, "knn_emb", "임베딩 kNN", BLUE, None)]:
            ax.bar(x + off, [r["rows"][n][s][key] for n in names], 0.25, color=color, edgecolor=BLACK, lw=0.6, hatch=hatch, label=name)
        if key == "auroc":
            ax.axhline(0.5, color=GRAY, lw=0.8, ls="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        panel(ax, letter)
    axes[0].legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.01), fontsize=7.5)
    axes[1].set_xticks(x, names, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "ood.png")
    plt.close(fig)


def heat_fig():
    import torch
    from wafer.heat import gradcam_heat, knn_heat
    from wafer.model import WaferCNN
    from wafer.ood import feature_maps, patch_bank
    from wafer.locate import systematic_mask

    d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
    X, y, split = d["X"], d["y"], d["split_lot"]
    model = WaferCNN(9).to("cuda")
    model.load_state_dict(torch.load(ROOT / "runs/lot_s0.pt", map_location="cuda"))
    rng = np.random.default_rng(1)
    bank = patch_bank(feature_maps(model, X[rng.choice(np.flatnonzero((split == 0) & (y == 0)), 6000, replace=False)], "cuda")[0], 300_000)
    te = np.flatnonzero(split == 2)
    picks = []
    for name in ["Center", "Edge-Loc", "Scratch", "Donut"]:
        c = CLASSES.index(name)
        cand = [i for i in rng.permutation(te[y[te] == c])[:60] if systematic_mask(X[i]).sum() > 20]
        picks.append((name, cand[0]))
    fig, axes = plt.subplots(3, 4, figsize=(6.4, 5.0))
    rows = ["원본", "패치 kNN", "Grad-CAM"]
    for j, (name, i) in enumerate(picks):
        m = X[i]
        heats = [None, knn_heat(model, m, bank), gradcam_heat(model, m, "cuda", "b3")]
        for r_, h in enumerate(heats):
            ax = axes[r_, j]
            ax.imshow(m, cmap=WAFER, vmin=0, vmax=2, interpolation="nearest")
            if h is not None:
                rgba = plt.get_cmap("viridis")(h)
                rgba[..., 3] = np.where(m > 0, np.clip((h - 0.45) / 0.55, 0, 1) * 0.9, 0)
                ax.imshow(rgba, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if r_ == 0:
                ax.set_title(name, fontsize=9)
            if j == 0:
                ax.set_ylabel(rows[r_], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "heat.png")
    plt.close(fig)


def agent_fig():
    rs = load("agent_rescored_v3_checker.json")
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    x = np.arange(3)
    vs = ["v1", "v2", "v3"]
    ax.bar(x - 0.19, [rs[v]["first"] * 100 for v in vs], 0.36, color="white", edgecolor=BLACK, lw=0.8, label="첫 답")
    ax.bar(x + 0.19, [rs[v]["final"] * 100 for v in vs], 0.36, color=BLUE, edgecolor=BLACK, lw=0.8, label="재질문 1회 뒤")
    for xi, v in zip(x, vs):
        ax.text(xi - 0.19, rs[v]["first"] * 100 + 2, f'{rs[v]["first"] * 100:.0f}', ha="center", fontsize=7.5)
        ax.text(xi + 0.19, rs[v]["final"] * 100 + 2, f'{rs[v]["final"] * 100:.0f}', ha="center", fontsize=7.5)
    ax.set_xticks(x, ["v1", "v2", "v3"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("검사 통과 (%)")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "agent.png")
    plt.close(fig)


def umap_fig():
    um = np.load(ROOT / "runs/umap_sample.npz")
    xy, y = um["xy"], um["y"]
    colors = ["#1f4e79", "#b03a2e", "#2e7d32", "#c77c02", "#6a3d9a", "#00838f", "#ad1457", "#5d4037"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bg = y <= 0
    ax.scatter(xy[bg, 0], xy[bg, 1], s=0.3, c=LIGHT, linewidths=0, label=f"none · 라벨 없음 ({bg.sum():,})")
    for k in range(1, 9):
        m = y == k
        ax.scatter(xy[m, 0], xy[m, 1], s=1.2, c=colors[k - 1], linewidths=0, label=f"{CLASSES[k]} ({m.sum():,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=7, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.5)
    fig.savefig(OUT / "umap.png")
    plt.close(fig)


if __name__ == "__main__":
    setup_style()
    split_fig()
    ood_fig()
    agent_fig()
    umap_fig()
    heat_fig()
    print("ok", sorted(p.name for p in OUT.glob("*.png")))
