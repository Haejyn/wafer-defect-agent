"""README 판독 흐름 애니메이션 — docs/media/agent.gif · agent.mp4.
모델(CNN · LLM)을 다시 돌리지 않는다. 기록된 출력만 재생한다.
  - 판정 · 위치 · 유사 사례 · 판독 카드(첫 답 · 재질문 뒤 답) · 검사 결과: reports/agent_eval.json
  - 사례 1 의 원본 · Grad-CAM · 유사 사례 그림: reports/screen/index.html 에 박힌 PNG 그대로
  - 사례 2 의 웨이퍼 맵: data/proc/labeled64.npz (make_screen.py 와 같은 색 · 같은 크기로 그림만)
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
W, H = 1280, 640

# 사례 — 사례 1: 첫 답 통과 (화면에 Grad-CAM 이 기록된 Loc) · 사례 2: 첫 답이 검사에 걸리고 재질문 뒤 통과
ROW_PASS, ROW_RETRY = 81489, 14866

BG, PANEL, CARD, INK, MUTED = "#f6f7f8", "#ffffff", "#f1f2f4", "#16181d", "#6b7280"
RED, OK, WARN, LINE = "#d6453d", "#1f8a5b", "#b7791f", "#e5e7eb"
RED_TINT, OK_TINT = "#fbe9e7", "#e6f4ec"
WAFER_CMAP = ListedColormap(["#ffffff00", "#cfd8d3", "#d6453d"])

FONT_DIR = ROOT / "build/fonts"


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"Pretendard-{weight}.ttf"), size)


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
    for sec in re.findall(r'<section class="case">(.*?)</section>', page, re.S):
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
        im = Image.new("RGBA", (W, H), BG)
        d = ImageDraw.Draw(im)
        c = self.c
        # 머리
        d.text((32, 24), "웨이퍼 판독 에이전트 · 판독 흐름", font=F["title"], fill=INK)
        d.text((32, 60), "기록된 출력 재생 — 모델 재실행 없음 · WM-811K 로트 시험 웨이퍼", font=F["sub"], fill=MUTED)
        tag = f"사례 {self.idx} / {self.total} · {'첫 답이 검사에 걸림 → 재질문' if self.retry else '첫 답 통과'}"
        tw = F["tag"].getlength(tag)
        d.rounded_rectangle((W - 32 - tw - 24, 26, W - 32, 54), 14, fill=PANEL)
        d.text((W - 32 - tw - 12, 32), tag, font=F["tag"], fill=INK)
        # 단계 칩
        x = 32
        for i, name in enumerate(self.steps):
            label = f"{i + 1}  {name}"
            w = F["chip"].getlength(label) + 26
            active, done = i == s["step"], i < s["step"]
            fill = INK if active else PANEL
            col = "#ffffff" if active else (INK if done else "#a3a8b1")
            d.rounded_rectangle((x, 94, x + w, 122), 14, fill=fill)
            d.text((x + 13, 100), label, font=F["chip"], fill=col)
            x += w + 8
        # 본판
        d.rounded_rectangle((24, 138, W - 24, H - 20), 18, fill=PANEL)

        # 맵
        size = 200
        paste(im, self.orig, (48, 162), size, s["map"])
        if s["map"] > 0:
            cap = f"원본 · 라벨 {c['true']}"
            d.text((48 + size / 2 - F["cap"].getlength(cap) / 2, 368), cap, font=F["cap"], fill=MUTED)
        vx = 280
        if self.heat is not None:
            paste(im, self.orig, (264, 162), size, s["map"] * (1 - s["heat"]))
            paste(im, self.heat, (264, 162), size, s["heat"])
            if s["heat"] > 0:
                cap = "히트맵 (Grad-CAM)"
                d.text((264 + size / 2 - F["cap"].getlength(cap) / 2, 368), cap, font=F["cap"], fill=mix(PANEL, MUTED, s["heat"]))
            vx = 492

        # 판정
        rows = [("분류 확신도", f"{c['confidence']:.2f}"), ("이상 점수 백분위", f"{c['ood_percentile']:.0f}"),
                ("계산된 위치", c["location"]), ("불량 다이", f"{c['fail_ratio'] * 100:.1f}%"),
                ("처음 보는 패턴 경고", "켜짐" if c["unknown_flag"] else "꺼짐")]
        vw = 250 if self.heat is not None else 300
        p = s["verdict"]
        if p > 0:
            d.text((vx, 162), c["pattern"], font=F["pattern"], fill=mix(PANEL, INK, p * 3))
            for i, (k, v) in enumerate(rows):
                a = ease(p * len(rows) - i * 0.7)
                if a <= 0:
                    continue
                y = 208 + i * 30
                if i == 4 and s["flag_hl"] > 0:
                    d.rounded_rectangle((vx - 8, y - 5, vx + vw + 8, y + 23), 8, fill=mix(PANEL, RED_TINT, s["flag_hl"]))
                d.text((vx, y), k, font=F["kv"], fill=mix(PANEL, MUTED, a))
                vcol = RED if (i == 4 and s["flag_hl"] > 0.5) else INK
                d.text((vx + vw - F["kvb"].getlength(v), y), v, font=F["kvb"], fill=mix(PANEL, vcol, a))

        # 유사 사례
        if s["sims"] > 0:
            d.text((48, 410), "유사 과거 웨이퍼 5장 · 임베딩 유사도", font=F["h4"], fill=mix(PANEL, MUTED, s["sims"] * 3))
            for i, (img, sim) in enumerate(zip(self.sims, c["similar"])):
                a = ease(s["sims"] * 2.2 - i * 0.28)
                if a <= 0:
                    continue
                sx = 48 + i * 136 + (1 - a) * 60
                paste(im, img, (sx, 436), 116, a)
                cap = f"{sim['pattern']} · {sim['similarity']:.2f}"
                d.text((sx + 58 - F["capS"].getlength(cap) / 2, 558), cap, font=F["capS"], fill=mix(PANEL, MUTED, a))

        # 판독 카드
        self.draw_card(im, d, s)
        return im

    def draw_card(self, im, d, s):
        if s["card"] <= 0:
            return
        x0, y0, x1, y1 = 748, 158, W - 44, H - 40
        d.rounded_rectangle((x0, y0, x1, y1), 14, fill=mix(PANEL, CARD, s["card"] * 2))
        if s["card"] < 0.5:
            return
        tx, tw = x0 + 18, x1 - x0 - 36
        card = self.last if s["answer"] == 2 else self.first
        head = "판독 카드 · qwen3.5:4b"
        d.text((tx, y0 + 16), head, font=F["head"], fill=INK)
        sub = "재질문 뒤 답" if s["answer"] == 2 else "첫 답"
        d.text((x1 - 18 - F["cap"].getlength(sub), y0 + 19), sub, font=F["cap"], fill=MUTED)
        y = y0 + 48
        budget = s["typed"]

        def typed(text):
            nonlocal budget
            n = max(0, min(len(text), budget))
            budget -= len(text)
            return text[:n]

        # 요약
        t = typed(card["summary"])
        for ln in wrap(t, F["body"], tw):
            d.text((tx, y), ln, font=F["body"], fill=INK)
            y += 21
        y = y0 + 48 + 21 * len(wrap(card["summary"], F["body"], tw)) + 10
        # 원인 후보 — 코드가 원인 표에서 찾아 붙임
        if budget > 0 or s["typed"] >= 10 ** 6:
            d.text((tx, y), "원인 후보 · 원인 표 인용", font=F["h4"], fill=MUTED)
            y += 20
            for cid in card["cause_ids"]:
                ct = cause_text[cid]
                d.text((tx, y), "• " + ct["cause"], font=F["bodyB"], fill=INK)
                y += 21
                for ln in wrap(f"“{ct['quote']}” [{ct['source']}]", F["quote"], tw - 14):
                    d.text((tx + 14, y), ln, font=F["quote"], fill=MUTED)
                    y += 17
            y += 8
            d.text((tx, y), "점검 순서", font=F["h4"], fill=MUTED)
            y += 20
            for i, step in enumerate(card["check_order"]):
                t = typed(step)
                if not t and budget < 0:
                    break
                lines = wrap(step, F["body"], tw - 22)
                shown = wrap(t, F["body"], tw - 22)
                is_drop = step in self.dropped and s["answer"] == 1
                h = 21 * len(lines)
                if is_drop and s["drop_hl"] > 0:
                    d.rounded_rectangle((tx - 8, y - 3, x1 - 10, y + h + 1), 8, fill=mix(CARD, RED_TINT, s["drop_hl"]))
                fill = INK
                if is_drop and s["strike"] > 0:
                    fill = mix(INK, "#b8bcc4", s["strike"])
                d.text((tx, y), f"{i + 1}.", font=F["body"], fill=fill)
                for j, ln in enumerate(shown):
                    d.text((tx + 22, y + 21 * j), ln, font=F["body"], fill=fill)
                    if is_drop and s["strike"] > 0:
                        lw = F["body"].getlength(ln) * min(1.0, s["strike"] * 1.4)
                        d.line((tx + 22, y + 21 * j + 10, tx + 22 + lw, y + 21 * j + 10), fill=RED, width=2)
                y += h + 4

        # 코드 검사 줄
        st = s["check"]
        if st:
            by = y1 - 16 - (78 if st in ("fail", "reask") else 44)
            if st == "running":
                d.rounded_rectangle((tx - 6, by, x1 - 12, y1 - 16), 10, fill=PANEL)
                dots = "·" * (1 + s["t"] // 3 % 3)
                d.text((tx + 8, by + 12), f"코드 검사 중 {dots}", font=F["status"], fill=MUTED)
            elif st == "pass":
                d.rounded_rectangle((tx - 6, by, x1 - 12, y1 - 16), 10, fill=OK_TINT)
                d.ellipse((tx + 8, by + 14, tx + 20, by + 26), fill=OK)
                d.text((tx + 28, by + 11), "검사 통과", font=F["status"], fill=OK)
                note = "첫 답 그대로" if not self.retry else "재질문 1회 뒤"
                note += f" · {self.r['seconds']} 초"
                d.text((x1 - 28 - F["cap"].getlength(note), by + 14), note, font=F["cap"], fill=OK)
            else:  # fail · reask
                prob = self.r["attempts"][0]["problems"][0]
                d.rounded_rectangle((tx - 6, by, x1 - 12, y1 - 16), 10, fill=RED_TINT)
                d.ellipse((tx + 8, by + 14, tx + 20, by + 26), fill=RED)
                d.text((tx + 28, by + 11), "검사에 걸림", font=F["status"], fill=RED)
                d.text((tx + 8, by + 42), f"“{prob}”", font=F["body"], fill=INK)
                if st == "reask":
                    note = "→ 재질문 1회"
                    d.text((x1 - 28 - F["status"].getlength(note), by + 11), note, font=F["status"], fill=INK)


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
    cases = [Case(ROW_PASS, 1, 2), Case(ROW_RETRY, 2, 2)]
    imgs = []
    for k, case in enumerate(cases):
        seq = [case.draw(s).convert("RGB") for s in timeline(case)]
        if k > 0:  # 앞 사례에서 이어지는 짧은 넘김
            prev = imgs[-1]
            for i in range(1, 5):
                imgs.append(Image.blend(prev, seq[0], i / 5))
        imgs += seq
    # 반복될 때 첫 사례로 부드럽게
    for i in range(1, 5):
        imgs.append(Image.blend(imgs[-1], imgs[0], i / 5))

    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)
    for i, im in enumerate(imgs):
        im.save(FRAMES / f"{i:04d}.png")

    mp4 = OUT / "agent.mp4"
    imageio.mimwrite(mp4, [np.asarray(im) for im in imgs], fps=FPS, codec="libx264", quality=8,
                     pixelformat="yuv420p", macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"])
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    gif = OUT / "agent.gif"
    gw = int(sys.argv[1]) if len(sys.argv) > 1 else W
    vf = (f"scale={gw}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FRAMES / "%04d.png"),
                    "-vf", vf, "-loop", "0", str(gif)], check=True)
    print(f"frames {len(imgs)} · {len(imgs) / FPS:.1f} s @ {FPS} fps")
    for p in (gif, mp4):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
