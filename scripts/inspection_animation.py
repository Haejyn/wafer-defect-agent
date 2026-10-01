"""Dashboard renderer for recorded inspection states (no model inference)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BG, INK, MUTED, LINE, ACCENT = '#f3f5f9', '#172438', '#68788e', '#e4e9f1', '#4b5ee9'
FONTS = {}


def font(size=14, bold=False):
    key = size, bold
    if key not in FONTS:
        path = ROOT / 'build/fonts' / f'Pretendard-{"Bold" if bold else "Regular"}.ttf'
        if not path.exists():
            path = Path('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf')
        FONTS[key] = ImageFont.truetype(str(path), size)
    return FONTS[key]


def wrap(text, width, size=14, bold=False):
    lines, current = [], ''
    for ch in text:
        if font(size, bold).getlength(current + ch) > width and current:
            lines.append(current)
            current = ch
        else:
            current += ch
    if current:
        lines.append(current)
    return lines


def draw(case, state, causes):
    im = Image.new('RGBA', (1440, 900), BG)
    d = ImageDraw.Draw(im)
    c = case.c

    def text(x, y, value, size=14, color=INK, bold=False):
        d.text((x,y), str(value), font=font(size,bold), fill=color)

    def panel(box, fill='#ffffff', radius=12, outline=LINE):
        d.rounded_rectangle(box, radius, fill=fill, outline=outline, width=1)

    def image(img, x, y, size, opacity=1):
        if opacity <= 0:
            return
        resized = img.convert('RGBA').resize((size,size),Image.Resampling.LANCZOS)
        if opacity < 1:
            resized.putalpha(resized.getchannel('A').point(lambda a: int(a*opacity)))
        im.alpha_composite(resized,(x,y))

    d.rectangle((0,0,194,900),fill='#142238')
    panel((22,32,58,68),fill='#283959',outline='#6474f7',radius=9)
    text(31,38,'W',21,'#acb7ff',True)
    text(70,40,'WAFER',20,'#ffffff',True)
    text(23,84,'INSPECTION WORKSPACE',9,'#9cacc4')
    text(25,153,'WORKSPACE',10,'#8496af',True)
    panel((16,184,178,225),fill='#2d3b63',outline='#2d3b63',radius=7)
    text(28,196,'◉   패턴 판독',13,'#ffffff',True)
    text(28,244,'⊞   웨이퍼 분포',13,'#b6c4d8')
    d.line((24,808,170,808),fill='#30405a')
    text(25,828,'기록된 판독 결과',12,'#d3deee')
    text(25,853,'WM-811K · 모델 재실행 없음',9,'#98aac3')
    text(228,30,'WORKSPACE   /   PATTERN INSPECTION',10,MUTED,True)
    text(228,58,'웨이퍼 패턴 판독',32,INK,True)
    text(229,106,'맵에서 근거를 찾고, 검증된 판독으로 연결합니다.',14,MUTED)
    panel((1178,61,1406,105),radius=7)
    text(1194,76,'WM-811K   811,457 maps',12,MUTED)
    metrics = [('아는 패턴 분류','0.853','macro-F1'),('낯선 패턴 감지','0.919','AUROC · kNN'),('판독 검사 통과','97.5%','재질문 후'),('평균 판독 시간','1.7','초 / 장')]
    panel((228,144,1406,237))
    for i,(label,value,unit) in enumerate(metrics):
        x = 250+i*294
        if i:
            d.line((x-22,166,x-22,215),fill=LINE)
        text(x,163,label,12,MUTED)
        text(x,185,value,29,INK,True)
        text(x+font(29,True).getlength(value)+10,200,unit,10,MUTED)
    text(229,262,'INSPECTION QUEUE',9,MUTED,True)
    text(229,280,'판독 사례',20,INK,True)
    text(330,287,f'{case.idx:02d} / {case.total:02d}   ·   {"재질문으로 수정한 사례" if case.retry else "첫 답이 검사를 통과한 사례"}',12,MUTED)
    # Same areas and information hierarchy as the HTML workspace.
    panel((228,323,933,653))
    panel((950,323,1406,846))
    text(251,343,'01 / WAFER MAP',11,MUTED,True)
    text(600,343,'02 / GRAD-CAM',11,MUTED,True)
    panel((250,370,558,616),fill='#f7f9fc',outline='#f7f9fc',radius=7)
    panel((598,370,911,616),fill='#f7f9fc',outline='#f7f9fc',radius=7)
    image(case.orig,288,379,230,state['map'])
    if case.heat is not None:
        image(case.orig,640,379,230,state['map']*(1-state['heat']))
        image(case.heat,640,379,230,state['heat'])
    else:
        text(660,463,'기록된 히트맵 없음',13,MUTED)
        text(650,487,'원본 맵과 판정 근거 확인',11,MUTED)
    text(253,626,f'원본 · 라벨 {c["true"]}',10,MUTED)
    text(600,626,'모델이 주목한 영역' if case.heat else '이미지 추정·생성 없음',10,MUTED)
    panel((228,667,933,846))
    if state['verdict']>0:
        text(250,686,c['pattern'],24,INK,True)
        badge = '낯선 패턴 · 확인 필요' if c['unknown_flag'] else '아는 패턴'
        panel((397,688,560,718),fill='#fff1db' if c['unknown_flag'] else '#eef0ff',outline=None,radius=5)
        text(407,695,badge,11,'#97601a' if c['unknown_flag'] else ACCENT,True)
        rows = [('분류 확신도',f'{c["confidence"]:.2f}'),('이상 점수 백분위',f'{c["ood_percentile"]:.0f}'),('계산된 위치',c['location']),('불량 다이',f'{c["fail_ratio"]*100:.1f}%')]
        for i,(k,v) in enumerate(rows):
            x=251+i*164
            text(x,746,k,11,MUTED)
            text(x,772,v,21,INK,True)
        text(251,818,'유사 사례는 임베딩 거리로 검색한 기록입니다.',10,MUTED)
    if state['sims']>0:
        # Compact recorded image strip above verdict metrics.
        for i,img in enumerate(case.sims):
            image(img,574+i*65,681,52,state['sims'])
    # Verified reading card: summary, cited candidate, then check order.
    text(974,345,'판독 카드',16,INK,True)
    text(1237,349,'qwen3.5:4b',11,MUTED)
    d.line((974,378,1382,378),fill=LINE)
    if state['card']>0.5:
        card = case.last if state['answer']==2 else case.first
        text(974,392,'재질문 뒤 답' if state['answer']==2 else '첫 답',10,ACCENT,True)
        budget=state['typed']
        n=max(0,min(len(card['summary']),budget))
        budget-=len(card['summary'])
        y=419
        for ln in wrap(card['summary'][:n],394,15):
            text(974,y,ln,15);y+=25
        y=419+25*len(wrap(card['summary'],394,15))+17
        if budget>0:
            text(974,y,'원인 후보 · 출처 근거',10,MUTED,True);y+=23
            for cid in card['cause_ids']:
                ct=causes[cid]
                text(974,y,ct['cause'],13,INK,True);y+=22
                for ln in wrap(f'“{ct["quote"]}” [{ct["source"]}]',386,10):
                    text(984,y,ln,10,MUTED);y+=16
            y+=17
            text(974,y,'점검 순서',10,MUTED,True);y+=23
            for i,step in enumerate(card['check_order']):
                n=max(0,min(len(step),budget));budget-=len(step)
                if n==0: break
                for j,ln in enumerate(wrap(step[:n],375,12)):
                    if y>728: raise ValueError('Recorded card overflows: increase canvas height')
                    text(974 if j==0 else 993,y,f'{i+1}. {ln}' if j==0 else ln,12)
                    if step in case.dropped and state['drop_hl']>0 and state['answer']==1:
                        d.line((993,y+17,1374,y+17),fill='#bd4a41',width=2)
                    y+=21
                y+=5
    if state['check']:
        st=state['check']
        passed=st=='pass'
        failed=st in ('fail','reask')
        fill='#e8f5ef' if passed else ('#fff0ed' if failed else '#f3f5f9')
        col='#16745b' if passed else ('#ad4438' if failed else MUTED)
        panel((974,762,1382,821),fill=fill,outline=None,radius=7)
        text(990,773,'검사 통과' if passed else ('재질문 중' if st=='reask' else '검사에 걸림') if failed else '코드 검사 중',13,col,True)
        note=(f'{"재질문 1회 뒤" if case.retry else "첫 답 그대로"} · {case.r["seconds"]}초' if passed else case.r['attempts'][0]['problems'][0] if failed else '패턴 · 위치 · 원인 후보 대조')
        text(990,799,note,10,col)
    # Step progress is outside the data surface.
    step_names=case.steps
    x=229
    for i,name in enumerate(step_names):
        active=i==state['step']
        col=ACCENT if active else MUTED
        text(x,868,f'{i+1:02d}  {name}',11,col,active)
        x+=142
    return im
