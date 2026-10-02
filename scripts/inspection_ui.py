"""BIW Weld Twin-style shell for recorded wafer results; no model inference."""
import html
import re


def describe_images(page: str) -> str:
    def describe(match):
        caption = html.escape(html.unescape(re.sub(r"<[^>]+>", "", match[2])), quote=True)
        return f'<img src="{match[1]}" alt="{caption}"><figcaption>{match[2]}</figcaption>'
    return re.sub(r'<img src="([^"]+)"><figcaption>(.*?)</figcaption>', describe, page, flags=re.S)


def restyle(page: str, asset_base: str = "../../docs/ui") -> str:
    # Read the original generated document, keeping all case values and image data.
    page = describe_images(page)
    cases = re.findall(r'<section class="case">.*?</section>', page, re.S)
    assert cases, "No recorded inspection cases"
    for i, case in enumerate(cases):
        case = case.replace('<section class="case">', f'<section class="case" id="case-{i}">', 1)
        case = re.sub(r'(<figure class="big">.*?</figure>)\s*(<figure class="big">.*?</figure>)',
                      r'<div class="maps"><div class="viewport-title"><span>웨이퍼 맵 · 판독 근거</span><span>원본 / Grad-CAM</span></div><div class="maps-grid">\1\2</div><div class="legend"><span><i></i>정상 다이</span><span><i class="fail"></i>불량 다이</span><span>히트맵: 모델이 주목한 영역</span></div></div>',
                      case, count=1, flags=re.S)
        cases[i] = case
    kpis = re.search(r'<div class="kpis">(.*?)</div>\s*<h2>', page, re.S)[1]
    map_start = page.index('<div class="map">') + len('<div class="map">')
    footer_start = page.index('<footer>')
    map_content = page[map_start:footer_start].strip().removesuffix('</div>')
    sample = re.search(r'UMAP 표본 ([\d,]+)점', page)[1]
    footer = re.search(r'<footer>(.*?)</footer>', page, re.S)[1]
    # Plotly's template colors are presentation only; traces/data stay unchanged.
    map_content = map_content.replace('"paper_bgcolor":"white"', '"paper_bgcolor":"#1b1c1e"')
    map_content = map_content.replace('"plot_bgcolor":"white"', '"plot_bgcolor":"#1b1c1e"')
    map_content = map_content.replace('"color":"#2a3f5f"', '"color":"#d8dade"')
    map_content = map_content.replace('"line":{"color":"#111","width":1}', '"line":{"color":"#5ee1d4","width":1}')
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#1b1c1e"><title>Wafer Inspect · 웨이퍼 패턴 판독</title><link rel="stylesheet" href="{asset_base}/inspection.css"><script src="{asset_base}/theme.js"></script></head><body>
<a class="skip" href="#inspection">판독 화면으로 이동</a><div class="app">
<header class="menubar"><a class="brand" href="#inspection"><i class="brand-mark"></i>Wafer Inspect</a><nav aria-label="작업 공간"><a href="#inspection" data-section="inspection">판독</a><a href="#map" data-section="map">분포 지도</a><a href="#eval" data-open="eval">평가 지표</a><a href="#about" data-open="about">출처</a></nav><span class="doc-title">WM-811K <b>811,457</b> maps · 기록된 결과</span><button id="theme-toggle" type="button" aria-label="다크 모드로 전환" aria-pressed="false">◐ 다크</button></header>
<div class="toolbar" aria-label="판독 도구"><div class="view-tools" role="group" aria-label="맵 보기"><button type="button" data-map-view="split" aria-pressed="true"><span class="tool-icon">▧</span> 원본 + 히트맵</button><button type="button" data-map-view="original" aria-pressed="false"><span class="tool-icon">▣</span> 원본</button><button type="button" data-map-view="heat" aria-pressed="false"><span class="tool-icon">◈</span> 히트맵</button></div><span class="tb-sep"></span><button type="button" id="previous-case" aria-label="이전 사례">←</button><button type="button" id="next-case" aria-label="다음 사례">→</button><span class="tb-fill"></span><a href="https://github.com/Haejyn/wafer-defect-agent/blob/main/reports/agent_eval.json">평가 기록 ↗</a><span class="recorded"><i></i> 기록 재생</span></div>
<div class="work"><aside class="rail" aria-label="판독 탐색"><div class="panel-h">▾ &nbsp; 판독 사례 <span>{len(cases)}</span></div><div class="case-picker" role="group" aria-label="판독 사례 선택"></div>
<div class="panel-h">▾ &nbsp; 평가 결과</div><div class="metrics"><div><span>분류 · macro-F1</span><b>0.853</b></div><div><span>낯선 패턴 · kNN AUROC</span><b>0.919</b></div><div><span>판독 검사 · 재질문 후</span><b>97.5<small>%</small></b></div><div><span>평균 판독 시간</span><b>1.7<small>초 / 장</small></b></div></div>
<details class="evaluation" id="eval"><summary>전체 평가 지표</summary><div class="kpis">{kpis}</div></details><details id="about" class="sources"><summary>데이터 · 원인 후보 출처</summary><p>{footer}</p></details><p class="rail-note">WM-811K · 로트 시험 사례<br>기록된 결과 탐색 · 모델 재실행 없음</p></aside>
<main id="inspection" class="workspace" data-section="inspection" data-map-view="split"><div class="workspace-bar"><span id="case-label">판독 사례</span><span>원본 맵의 값과 색상 보존</span></div><div class="inspection-surface">{''.join(cases)}</div><section id="map" class="distribution"><div class="panel-h">웨이퍼 분포 지도 <span>UMAP 표본 {sample}점</span></div><p class="map-description">확대하거나 범례를 선택해 패턴별 분포를 확인하세요.</p><div class="map">{map_content}</div></section></main></div>
<footer class="statusbar"><span id="selection-status">사례 선택 없음</span><span>WM-811K · 분류 / 위치 / 유사 사례</span><span id="validation-status">기록된 판독 카드</span></footer></div><script src="{asset_base}/inspection.js"></script></body></html>'''
