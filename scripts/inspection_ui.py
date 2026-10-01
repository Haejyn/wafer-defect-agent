"""Shared inspection shell; preserves recorded results and embedded Plotly data."""
import re
import html


def describe_images(page: str) -> str:
    def describe(match):
        caption = html.escape(html.unescape(re.sub(r"<[^>]+>", "", match[2])), quote=True)
        return f'<img src="{match[1]}" alt="{caption}"><figcaption>{match[2]}</figcaption>'
    return re.sub(r'<img src="([^"]+)"><figcaption>(.*?)</figcaption>', describe, page, flags=re.S)


def restyle(page: str, asset_base: str = "../../docs/ui") -> str:
    page = describe_images(page)
    page = re.sub(r"<style>.*?</style>", f'<link rel="stylesheet" href="{asset_base}/inspection.css">', page, count=1, flags=re.S)
    rail = '''<a class="skip" href="#inspection">판독 사례로 이동</a>
<aside class="rail"><a class="brand" href="#inspection"><span class="brand-mark">W</span> WAFER<span class="brand-sub">INSPECTION WORKSPACE</span></a>
<div class="rail-label">WORKSPACE</div><a class="rail-link selected" href="#inspection">◉ &nbsp; 패턴 판독</a><a class="rail-link" href="#map">⊞ &nbsp; 웨이퍼 분포</a>
<div class="rail-note"><span class="live-dot"></span> 기록된 판독 결과<br><small>WM-811K · 로트 시험 사례</small></div></aside><main>'''
    page = page.replace("<body>", "<body>" + rail, 1)
    page = re.sub(r"<header>.*?</header>", '''<header><div class="breadcrumb">WORKSPACE <span>/</span> PATTERN INSPECTION</div><div class="title-row"><div><h1>웨이퍼 패턴 판독</h1><p class="sub">맵에서 근거를 찾고, 검증된 판독으로 연결합니다.</p></div><span class="dataset">WM-811K <b>811,457</b> maps</span></div></header>''', page, count=1, flags=re.S)
    page = page.replace('<div class="kpis">', '<details class="evaluation"><summary>전체 평가 지표 보기</summary><div class="kpis">', 1)
    page = page.replace('<h2>판독 사례</h2>', '''</details><div class="metrics"><div><span>아는 패턴 분류</span><b>0.853<small>macro-F1</small></b></div><div><span>낯선 패턴 감지</span><b>0.919<small>AUROC · kNN</small></b></div><div><span>판독 검사 통과</span><b>97.5<small>% · 재질문 후</small></b></div><div><span>평균 판독 시간</span><b>1.7<small>초 / 장</small></b></div></div><section id="inspection" class="inspection"><div class="section-heading"><div><span class="eyebrow">INSPECTION QUEUE</span><h2>판독 사례</h2></div><p>사례를 선택해 판정 근거를 확인하세요.</p></div><div class="case-picker" role="group" aria-label="판독 사례 선택"></div>''', 1)
    page = re.sub(r'<h2>81만 장 지도 \(UMAP 표본 ([\d,]+)점\)</h2>', r'</section><section id="map" class="distribution"><div class="section-heading"><div><span class="eyebrow">PATTERN DISTRIBUTION</span><h2>웨이퍼 분포 지도</h2></div><p>UMAP 표본 \1점 · 확대와 범례 필터 지원</p></div>', page, count=1)
    page = page.replace("<footer>", "</section><footer>", 1)
    page = page.replace("</body>", f'</main><script src="{asset_base}/inspection.js"></script></body>', 1)
    return page
