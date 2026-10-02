"""README 판독 흐름 애니메이션 — docs/media/agent.gif · agent.mp4.
모델(CNN · LLM)을 다시 돌리지 않는다. 기록된 출력만 재생한다.
  - 판정 · 위치 · 유사 사례 · 판독 카드(첫 답 · 재질문 뒤 답) · 검사 결과: reports/agent_eval.json
  - 사례 1 의 원본 · Grad-CAM · 유사 사례 그림: reports/screen/index.html 에 박힌 PNG 그대로
  - 사례 2 의 웨이퍼 맵: docs/media/cases/14866 기록 이미지 (없으면 data/proc/labeled64.npz)
  - 원인 이름 · 인용: src/wafer/causes.json (카드의 cause_id 로 찾음)
  - 기록된 카드에 wafer.agent.check 를 다시 걸어 기록된 problems 와 같은지 확인한다 (코드 검사만, 모델 아님)
사용: python scripts/make_agent_animation.py"""
import base64
import html
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.agent import CAUSES, check  # noqa: E402

OUT = ROOT / "docs/media"
FRAMES = ROOT / "build/anim_frames"
FPS = 12
W, H = 1440, 900

# 사례 — 사례 1: 첫 답 통과 (화면에 Grad-CAM 이 기록된 Loc) · 사례 2: 첫 답이 검사에 걸리고 재질문 뒤 통과
ROW_PASS, ROW_RETRY = 81489, 14866

BG, PANEL, CARD, INK, MUTED = "#f6f7f8", "#ffffff", "#f1f2f4", "#16181d", "#6b7280"
RED, OK, WARN, LINE = "#d6453d", "#1f8a5b", "#b7791f", "#e5e7eb"
RED_TINT, OK_TINT = "#fbe9e7", "#e6f4ec"
WAFER_CMAP = ListedColormap(["#ffffff00", "#cfd8d3", "#d6453d"])

FONT_DIR = ROOT / "build/fonts"


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / f"Pretendard-{weight}.ttf"
    if not path.exists():
        path = Path("C:/Windows/Fonts/malgunbd.ttf" if weight != "Regular" else "C:/Windows/Fonts/malgun.ttf")
    return ImageFont.truetype(str(path), int(size))


F = {
    "title": font("Bold", 24), "sub": font("Regular", 14), "chip": font("SemiBold", 13),
    "pattern": font("Bold", 26), "kv": font("Regular", 15), "kvb": font("Bold", 15),
    "cap": font("Regular", 12), "capS": font("Regular", 11),
    "head": font("Bold", 15), "body": font("Regular", 14), "bodyB": font("SemiBold", 14),
    "h4": font("SemiBold", 12), "quote": font("Regular", 11.5), "status": font("Bold", 14), "tag": font("SemiBold", 12),
}


# ---------- 기록 읽기 ----------
agent = json.loads((ROOT / "reports/agent_eval.json").read_text(encoding="utf-8"))
by_row = {r["case"]["row"]: r for r in agent["results"]}
cause_text = {c["cause_id"]: c for lst in CAUSES["patterns"].values() for c in lst}


def decode(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGBA")


def screen_images(row: int) -> list[Image.Image]:
    """화면(index.html)에서 이 사례 줄의 그림 7장: 원본 · Grad-CAM · 유사 5장. 요약 문장으로 사례를 찾는다."""
    page = (ROOT / "reports/screen/index.html").read_text(encoding="utf-8")
    want = by_row[row]["final"]["summary"]
    for sec in re.findall(r'<section class="case"(?:\s[^>]*)?>(.*?)</section>', page, re.S):
        if html.unescape(re.search(r"<p>(.*?)</p>", sec, re.S).group(1)) == want:
            return [decode(u) for u in re.findall(r'<img src="(data:image/png;base64,[^"]+)"', sec)]
    raise SystemExit(f"화면에 row {row} 사례가 없다 — make_screen.py 를 먼저 돌린다")


def wafer_img(m, size=2.2) -> Image.Image:
    """make_screen.wafer_png 와 같은 그림 (히트맵 없이)."""
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


def verify(r):
    """기록된 카드에 지금의 검사 코드를 다시 걸어 기록된 problems 와 같은지."""
    for a in r["attempts"]:
        assert check(r["case"], a["card"]) == a["problems"], (r["case"]["row"], a["problems"])


# ---------- 그리기 도구 ----------
def wrap(text: str, fnt, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if fnt.getlength(cur + ch) > width and cur:
            cut = cur.rfind(" ")
            if cut > len(cur) * 0.5:
                lines.append(cur[:cut])
                cur = cur[cut + 1:] + ch
            else:
                lines.append(cur)
                cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def mix(c1: str, c2: str, t: float) -> tuple:
    a = np.array(Image.new("RGB", (1, 1), c1).getpixel((0, 0)), float)
    b = np.array(Image.new("RGB", (1, 1), c2).getpixel((0, 0)), float)
    return tuple(int(v) for v in a + (b - a) * max(0.0, min(1.0, t)))


def paste(canvas: Image.Image, img: Image.Image, xy, size: int, alpha: float = 1.0):
    if alpha <= 0:
        return
    im = img.resize((size, size), Image.LANCZOS)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im.putalpha(a)
    canvas.alpha_composite(im, (int(xy[0]), int(xy[1])))


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


# ---------- 한 장면 ----------
class Case:
    def __init__(self, row: int, idx: int, total: int):
        self.r = by_row[row]
        verify(self.r)
        self.c = self.r["case"]
        self.idx, self.total = idx, total
        self.retry = len(self.r["attempts"]) > 1
        imgs = screen_images(row) if not self.retry else None
        if imgs:  # 기록된 화면 그림 그대로
            self.orig, self.heat, self.sims = imgs[0], imgs[1], imgs[2:7]
        elif (ROOT / f"docs/media/cases/{row}/original.png").exists():
            cache = ROOT / f"docs/media/cases/{row}"
            self.orig = Image.open(cache / "original.png").convert("RGBA")
            self.heat = None
            self.sims = [Image.open(cache / f"similar-{i}.png").convert("RGBA") for i in range(5)]
        else:
            X = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)["X"]
            self.orig, self.heat = wafer_img(X[row]), None
            self.sims = [wafer_img(X[s["row"]], 0.9) for s in self.c["similar"]]
        first, last = self.r["attempts"][0]["card"], self.r["attempts"][-1]["card"]
        self.first, self.last = first, last
        # 재질문 뒤 빠진 줄 = 두 답의 점검 순서 차이 (기록된 두 답을 비교)
        self.dropped = [s for s in first["check_order"] if s not in last["check_order"]]
        self.steps = ["맵", "히트맵", "판정", "유사 사례", "판독 카드", "코드 검사"]
        if self.heat is None:
            self.steps.remove("히트맵")

    # 상태 s: dict — 각 요소의 진행(0~1)과 카드 글자 수 등
    def draw(self, s: dict) -> Image.Image:
        from inspection_animation import draw
        return draw(self, s, cause_text)


def timeline(case: Case) -> list[dict]:
    """장면 상태 목록 (프레임마다 하나)."""
    frames = []
    base = dict(step=0, map=0.0, heat=0.0, verdict=0.0, sims=0.0, card=0.0, typed=0, answer=1,
                check=None, flag_hl=0.0, drop_hl=0.0, strike=0.0, t=0)
    cur = dict(base)

    def hold(n, **kw):
        for _ in range(n):
            cur.update(kw)
            cur["t"] += 1
            frames.append(dict(cur))

    def tween(n, key, a=0.0, b=1.0, **kw):
        for i in range(1, n + 1):
            cur.update(kw)
            cur[key] = a + (b - a) * ease(i / n)
            cur["t"] += 1
            frames.append(dict(cur))

    has_heat = case.heat is not None
    step = 0
    hold(3)
    tween(8, "map", step=step)
    hold(4)
    if has_heat:
        step += 1
        tween(12, "heat", step=step)
        hold(6)
    step += 1
    tween(12, "verdict", step=step)
    hold(6)
    step += 1
    tween(12, "sims", step=step)
    hold(6)
    step += 1
    tween(6, "card", step=step)
    first = case.first
    total = len(first["summary"]) + sum(len(x) for x in first["check_order"])
    per = 7
    for n in range(0, total + per, per):
        cur["typed"] = min(n, total)
        hold(1)
    hold(8)
    step += 1
    hold(10, step=step, check="running")
    if not case.retry:
        hold(26, check="pass")
    else:
        hold(8, check="fail")
        tween(6, "flag_hl")
        tween(6, "drop_hl")
        hold(20)
        hold(12, check="reask")
        tween(10, "strike")
        hold(8)
        # 재질문 뒤 답 — 빠진 줄 없이 다시 그림
        cur.update(answer=2, typed=10 ** 6, flag_hl=0.0, drop_hl=0.0, strike=0.0, check="running")
        hold(10)
        hold(36, check="pass")
    return frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Render one frame at a time: full-resolution frames need not fit in RAM.
    cases = [Case(ROW_PASS, 1, 2), Case(ROW_RETRY, 2, 2)]
    FRAMES.mkdir(parents=True, exist_ok=True)
    count = 0
    mp4 = OUT / "agent.mp4"
    with imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8,
                           pixelformat="yuv420p", macro_block_size=2,
                           ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for case in cases:
            states = timeline(case)
            for state in states:
                im = case.draw(state).convert("RGB")
                writer.append_data(np.asarray(im))
                count += 1
            if case.idx == 1:
                im.save(OUT / "agent-poster.png")
                im.save(ROOT / "docs/img/case_loc.png")
            im.save(FRAMES / f"case-{case.idx}-final.png")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    gif = OUT / "agent.gif"
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else 960
    vf = (f"fps=8,scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4),
                    "-filter_complex", vf, "-loop", "0", str(gif)], check=True)
    print(f"frames {count} · {count / FPS:.1f} s @ {FPS} fps")
    for output in (gif, mp4):
        print(output.relative_to(ROOT), f"{output.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
