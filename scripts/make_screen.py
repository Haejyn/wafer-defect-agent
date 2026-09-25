"""6단계 — 판독 화면(정적 HTML). reports/*.json 과 runs/ 산출물로 만든다.
사례 줄: 원본 | 이상 히트맵 | 판정·경고·위치 | 유사 사례 | 판독 카드(검사 결과)  +  81만 장 지도(plotly)"""
import base64
import html
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.agent import CAUSES, UNKNOWN  # noqa: E402
from wafer.model import WaferCNN  # noqa: E402
from wafer.ood import feature_maps, patch_bank, patch_scores  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402

device = "cuda"
OUT = ROOT / "reports/screen"
OUT.mkdir(parents=True, exist_ok=True)
d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, split = d["X"], d["y"], d["split_lot"]
agent = json.loads((ROOT / "reports/agent_eval.json").read_text(encoding="utf-8"))
WAFER_CMAP = ListedColormap(["#ffffff00", "#cfd8d3", "#d6453d"])


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def wafer_png(m, size=2.2, heat=None):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(m, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    if heat is not None:
        h = np.ma.masked_where(m == 0, heat)
        ax.imshow(h, cmap="magma", alpha=0.72, interpolation="bilinear")
    ax.axis("off")
    return png(fig)


# 히트맵 — 전체 모델의 패치 특징을 정상(none) 학습 웨이퍼 패치와 비교 (PatchCore 식)
model = WaferCNN(9).to(device)
model.load_state_dict(torch.load(ROOT / "runs/lot_s0.pt", map_location=device))
rng = np.random.default_rng(0)
normal_rows = rng.choice(np.flatnonzero((split == 0) & (y == 0)), 6000, replace=False)
nmaps, _ = feature_maps(model, X[normal_rows], device)
bank = patch_bank(nmaps, 300_000)


def heatmap(i):
    maps, _ = feature_maps(model, X[i:i + 1], device)
    h, _ = patch_scores(maps, bank, device)
    t = torch.from_numpy(h[None]).float()
    up = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].numpy()
    return (up - up.min()) / (up.max() - up.min() + 1e-9)


cause_text = {c["cause_id"]: c for lst in CAUSES["patterns"].values() for c in lst}


def card_html(r):
    c, f = r["case"], r["final"]
    warn = '<span class="warn">처음 보는 패턴</span>' if c["unknown_flag"] else ""
    first = "통과" if r["pass_first"] else "재질문"
    status = f'<span class="ok">검사 통과</span> <span class="muted">첫 답 {first}</span>' if r["pass_final"] else '<span class="bad">검사 실패</span>'
    causes = "".join(
        f'<li><b>{html.escape(cause_text[cid]["cause"])}</b><br><span class="quote">“{html.escape(cause_text[cid]["quote"])}” [{cause_text[cid]["source"]}]</span></li>'
        for cid in f.get("cause_ids", []) if cid in cause_text) or '<li class="muted">원인 표에 없음 — 엔지니어 확인</li>'
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in f.get("check_order", []))
    sims = "".join(f'<figure><img src="{wafer_png(X[s["row"]], 0.9)}"><figcaption>{s["pattern"]} · {s["similarity"]:.2f}</figcaption></figure>' for s in c["similar"])
    return f"""
<section class="case">
  <figure class="big"><img src="{wafer_png(X[c["row"]])}"><figcaption>원본 · 라벨 {c["true"]}</figcaption></figure>
  <figure class="big"><img src="{wafer_png(X[c["row"]], heat=heatmap(c["row"]))}"><figcaption>이상 히트맵</figcaption></figure>
  <div class="verdict">
    <div class="pattern">{html.escape(c["pattern"])} {warn}</div>
    <div class="kv"><span>분류 확신도</span><b>{c["confidence"]:.2f}</b></div>
    <div class="kv"><span>이상 점수 백분위</span><b>{c["ood_percentile"]:.0f}</b></div>
    <div class="kv"><span>계산된 위치</span><b>{c["location"]}</b></div>
    <div class="kv"><span>불량 다이</span><b>{c["fail_ratio"] * 100:.1f}%</b></div>
    <div class="sims">{sims}</div>
  </div>
  <div class="card">
    <div class="card-head">판독 카드 · {status}</div>
    <p>{html.escape(f.get("summary", ""))}</p>
    <h4>원인 후보</h4><ul>{causes}</ul>
    <h4>점검 순서</h4><ol>{steps}</ol>
  </div>
</section>"""


# 보여 줄 사례: 아는 패턴 6종 + 처음 보는 패턴 2장 (검사 통과한 것 우선)
res = agent["results"]
chosen, seen = [], set()
for r in sorted(res, key=lambda r: (not r["pass_final"], not r["pass_first"])):
    key = r["set"] if r["set"] != "known" else r["case"]["true"]
    if key in seen:
        continue
    if r["set"] == "known" and len([x for x in chosen if x["set"] == "known"]) >= 6:
        continue
    if r["set"] != "known" and (len([x for x in chosen if x["set"] != "known"]) >= 2 or not r["case"]["unknown_flag"]):
        continue
    seen.add(key)
    chosen.append(r)
cases = "".join(card_html(r) for r in chosen)

# 지도
import plotly.graph_objects as go  # noqa: E402

um = np.load(ROOT / "runs/umap_sample.npz")
xy, uy, nov = um["xy"], um["y"], um["novelty"]
palette = {-1: "#c9ccd1", 0: "#8fa39a", 1: "#e4572e", 2: "#f3a712", 3: "#29335c", 4: "#669bbc", 5: "#a8c686", 6: "#7b2cbf", 7: "#ff70a6", 8: "#00a6a6"}
traces = []
for k in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]:
    m = uy == k
    name = "라벨 없음" if k < 0 else CLASSES[k]
    traces.append(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{name} ({m.sum():,})",
                               marker=dict(size=2 if k <= 0 else 3, color=palette[k], opacity=0.35 if k <= 0 else 0.8)))
thr = np.nanpercentile(nov, 99.5)
m = np.nan_to_num(nov, nan=-1) >= thr
traces.append(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"라벨 없음 중 가장 낯선 0.5% ({m.sum():,})",
                           marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color="#111", width=1))))
fig = go.Figure(traces)
fig.update_layout(template="plotly_white", height=620, margin=dict(l=10, r=10, t=10, b=10), legend=dict(itemsizing="constant"),
                  xaxis=dict(visible=False), yaxis=dict(visible=False))
map_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

# 수치
def load(p):
    q = ROOT / "reports" / p
    return json.loads(q.read_text(encoding="utf-8")) if q.exists() else {}

kpi = load("summary.json")
kpi_html = "".join(f'<div class="kpi"><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>' for k, v in kpi.items())

page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>웨이퍼 판독 에이전트</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#f6f7f8;--panel:#fff;--ink:#16181d;--muted:#6b7280;--accent:#d6453d;--ok:#1f8a5b;--warn:#b7791f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:Pretendard,"Noto Sans KR","Malgun Gothic",sans-serif}}
header{{padding:28px 32px 8px}} h1{{margin:0;font-size:26px}} .sub{{color:var(--muted);margin-top:6px}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;padding:12px 32px}} .kpi{{background:var(--panel);border-radius:12px;padding:12px 16px;min-width:170px}}
.kpi b{{display:block;font-size:22px}} .kpi span{{color:var(--muted);font-size:13px}}
h2{{padding:18px 32px 0;margin:0;font-size:18px}} .map{{margin:10px 32px;background:var(--panel);border-radius:14px;padding:8px}}
.case{{display:grid;grid-template-columns:190px 190px 250px 1fr;gap:14px;margin:12px 32px;background:var(--panel);border-radius:14px;padding:16px;align-items:start}}
figure{{margin:0;text-align:center}} figure img{{width:100%}} figcaption{{font-size:12px;color:var(--muted)}}
.pattern{{font-size:20px;font-weight:700;margin-bottom:8px}} .kv{{display:flex;justify-content:space-between;font-size:14px;padding:3px 0}} .kv span{{color:var(--muted)}}
.sims{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-top:10px}} .sims figcaption{{font-size:10px}}
.card{{background:var(--bg);border-radius:12px;padding:12px 14px;font-size:14px}} .card-head{{font-weight:700;margin-bottom:6px}}
.card h4{{margin:10px 0 4px;font-size:13px;color:var(--muted)}} .card ul,.card ol{{margin:0;padding-left:18px}} .quote{{color:var(--muted);font-size:12px}}
.ok{{color:var(--ok)}} .bad{{color:var(--accent)}} .warn{{background:var(--warn);color:#fff;border-radius:6px;padding:2px 8px;font-size:13px;margin-left:6px}} .muted{{color:var(--muted)}}
footer{{padding:20px 32px 40px;color:var(--muted);font-size:12px}}
@media (max-width:900px){{.case{{grid-template-columns:1fr 1fr}} .card,.verdict{{grid-column:1/-1}}}}
</style></head><body>
<header><h1>웨이퍼 불량 패턴 탐지·판독 에이전트</h1><div class="sub">실제 팹 웨이퍼 맵 WM-811K · 아는 패턴은 분류, 처음 보는 패턴은 경고, 판독은 코드가 검사</div></header>
<div class="kpis">{kpi_html}</div>
<h2>판독 사례</h2>{cases}
<h2>81만 장 지도 (UMAP 표본 {len(xy):,}점)</h2><div class="map">{map_html}</div>
<footer>데이터: WM-811K (MIR Lab). [1] M.-J. Wu, J.-S. R. Jang, J.-L. Chen, IEEE TSM 28(1), 2015. [2] MIR-WM811K, http://mirlab.org/dataset/public/ · 원인 후보 출처: {html.escape(CAUSES["sources"]["S1"])} / {html.escape(CAUSES["sources"]["S2"])}</footer>
</body></html>"""
(OUT / "index.html").write_text(page, encoding="utf-8")
print("wrote", OUT / "index.html", "cases", len(chosen))
