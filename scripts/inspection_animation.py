"""BIW Weld Twin-inspired renderer of recorded wafer inspection states."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
APP, PANEL, FIELD, LINE = '#1b1c1e', '#232427', '#2c2d31', '#313236'
INK, MUTED, ACCENT, BAD, WARN = '#d8dade', '#a1a5ad', '#5ee1d4', '#ff766c', '#ffb762'
FONTS = {}


def font(size=12, bold=False):
    key = size, bold
    if key not in FONTS:
        path = ROOT / 'build/fonts' / f'Pretendard-{"Bold" if bold else "Regular"}.ttf'
        if not path.exists():
            path = Path('C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf')
        FONTS[key] = ImageFont.truetype(str(path),size)
    return FONTS[key]


def wrap(text, width, size=12, bold=False):
    lines,current=[],''
    for ch in text:
        if font(size,bold).getlength(current+ch)>width and current:
            lines.append(current);current=ch
        else: current+=ch
    if current: lines.append(current)
    return lines


def draw(case,state,causes):
    im=Image.new('RGBA',(1440,900),APP)
    d=ImageDraw.Draw(im)
    c=case.c

    def text(x,y,value,size=12,color=INK,bold=False):
        d.text((x,y),str(value),font=font(size,bold),fill=color)

    def box(coords,fill=PANEL,outline=LINE,radius=2):
        d.rounded_rectangle(coords,radius,fill=fill,outline=outline)

    def image(img,x,y,size,opacity=1):
        if opacity<=0:return
        asset=img.convert('RGBA').resize((size,size),Image.Resampling.LANCZOS)
        # The retry cache is an exact crop of the original light demo; remove only
        # its white canvas so the recorded gray/red die colors remain unchanged.
        if asset.getchannel('A').getextrema()==(255,255):
            import numpy as np
            pixels=np.array(asset)
            white=(pixels[:,:,:3]==255).all(axis=2)
            pixels[white,3]=0
            asset=Image.fromarray(pixels)
        if opacity<1:
            asset.putalpha(asset.getchannel('A').point(lambda a:int(a*opacity)))
        im.alpha_composite(asset,(x,y))

    # Compact menu/toolbar, dock divisions, and mint focus match BIW Weld Twin.
    d.rectangle((0,0,1440,29),fill=APP)
    d.line((0,29,1440,29),fill=LINE)
    box((9,9,20,20),ACCENT,ACCENT,3)
    text(28,7,'Wafer Inspect',12,'#f1f2f4',True)
    for x,label in [(149,'판독'),(192,'분포 지도'),(267,'평가 지표'),(340,'출처')]:text(x,8,label,11)
    text(1160,8,'WM-811K · 811,457 maps',10,MUTED)
    d.rectangle((0,30,1440,65),fill=PANEL)
    d.line((0,65,1440,65),fill=LINE)
    box((7,35,141,59),'#2c4543',None,3)
    text(17,42,'▧  원본 + 히트맵',11,ACCENT)
    text(154,42,'▣  원본',11,MUTED)
    text(228,42,'◈  히트맵',11,MUTED)
    d.line((315,39,315,57),fill='#3a3c41')
    text(330,41,'←   →',12,MUTED)
    text(1275,42,'기록된 판독 재생',10,ACCENT)
    d.rectangle((0,66,209,875),fill=PANEL)
    d.line((209,66,209,875),fill=LINE)
    d.rectangle((1110,66,1440,875),fill=PANEL)
    d.line((1110,66,1110,875),fill=LINE)
    d.rectangle((0,66,208,93),fill='#26272a')
    text(10,74,'판독 사례',10,MUTED,True)
    text(182,74,'2',10,MUTED)
    for n,(label,tag) in enumerate([('Loc','첫 답 통과'),('Random','재질문')]):
        y=101+n*31
        active=case.idx==n+1
        if active:d.rectangle((0,y-1,208,y+27),fill='#2c4543')
        text(10,y+6,f'{n+1:02d}',10,ACCENT if active else MUTED)
        d.ellipse((37,y+11,43,y+17),fill='#4da3ff')
        text(51,y+5,label,12,INK,active)
        text(140,y+7,tag,9,ACCENT if active else MUTED)
    d.rectangle((0,185,208,211),fill='#26272a')
    text(10,193,'평가 결과',10,MUTED,True)
    metrics=[('분류 · macro-F1','0.853'),('kNN AUROC','0.919'),('검사 · 재질문 후','97.5%'),('평균 판독 시간','1.7초')]
    for i,(label,value) in enumerate(metrics):
        x=10+(i%2)*104;y=231+(i//2)*76
        text(x,y,label,9,MUTED)
        text(x,y+26,value,20,'#f1f2f4')
    d.line((0,383,209,383),fill=LINE)
    text(10,398,'판독 근거',10,MUTED,True)
    for i,label in enumerate(['원본 웨이퍼 맵','분류 · 위치 계산','유사 사례 5장','원인 후보 출처','코드 검사 결과']):
        y=429+i*26
        text(12,y,'›',12,MUTED);text(28,y,label,11,MUTED)
    d.line((9,811,199,811),fill=LINE)
    text(10,826,'WM-811K · 로트 시험 사례',10,MUTED)
    text(10,847,'기록된 결과 · 모델 재실행 없음',9,MUTED)
    # Central scientific viewport: original values, actual heatmap only.
    d.rectangle((210,66,1109,94),fill=APP)
    text(223,74,f'사례 {case.idx:02d} / 02 · {c["pattern"]}',10,INK)
    text(936,74,'원본 맵 값 · 색상 보존',9,MUTED)
    d.rectangle((210,95,1109,587),fill='#1f2023')
    for x in range(210,1110,30):d.line((x,95,x,587),fill='#292a2d')
    for y in range(95,588,30):d.line((210,y,1109,y),fill='#292a2d')
    text(226,107,'웨이퍼 맵 · 판독 근거',10,MUTED)
    text(986,107,'원본 / Grad-CAM',9,MUTED)
    text(361,164,'WAFER MAP',10,MUTED)
    text(806,164,'GRAD-CAM',10,MUTED)
    image(case.orig,278,197,304,state['map'])
    if case.heat is not None:
        image(case.orig,738,197,304,state['map']*(1-state['heat']))
        image(case.heat,738,197,304,state['heat'])
    else:
        text(828,322,'기록된 히트맵 없음',13,MUTED)
        text(810,349,'원본 맵과 계산된 판정 확인',11,MUTED)
    text(365,520,f'원본 · 라벨 {c["true"]}',11,MUTED)
    text(811,520,'모델이 주목한 영역' if case.heat else '히트맵 재생성 없음',11,MUTED)
    d.rectangle((491,563,498,570),fill='#cfd8d3');text(506,560,'정상 다이',10,MUTED)
    d.rectangle((583,563,590,570),fill='#d6453d');text(598,560,'불량 다이',10,MUTED)
    # Bottom dock: decision, numeric properties, similar cases, progress rail.
    d.rectangle((210,588,1109,874),fill=PANEL)
    d.line((210,588,1109,588),fill=LINE)
    d.rectangle((210,589,1109,616),fill='#26272a')
    text(225,597,'판정 · 유사 사례',11,INK,True)
    if state['verdict']>0:
        text(225,629,c['pattern'],19,'#f1f2f4',True)
        badge='낯선 패턴 · 확인 필요' if c['unknown_flag'] else '아는 패턴'
        text(350,635,badge,10,WARN if c['unknown_flag'] else ACCENT)
        props=[('분류 확신도',f'{c["confidence"]:.2f}'),('이상 점수 백분위',f'{c["ood_percentile"]:.0f}'),('계산된 위치',c['location']),('불량 다이',f'{c["fail_ratio"]*100:.1f}%')]
        for i,(key,value) in enumerate(props):
            x=228+i*217
            text(x,663,key,10,MUTED);text(x,681,value,17,INK)
    d.line((223,712,1096,712),fill=LINE)
    text(228,727,'유사 사례 · TOP 5',10,MUTED)
    if state['sims']>0:
        for i,(img,similar) in enumerate(zip(case.sims,c['similar'])):
            x=380+i*144
            image(img,x,721,63,state['sims'])
            text(x,790,f'{similar["pattern"]} · {similar["similarity"]:.2f}',9,MUTED)
    d.line((210,817,1109,817),fill=LINE)
    text(225,829,'판독 흐름',9,MUTED)
    for i,name in enumerate(case.steps):
        x=305+i*128;active=i==state['step']
        color=ACCENT if active else MUTED
        text(x,829,f'{i+1:02d} {name}',10,color,active)
        if i<=state['step']:d.rectangle((x,855,x+105,858),fill=ACCENT if active else '#376963')
    # Right properties dock contains the original LLM output and checks.
    d.rectangle((1111,66,1440,93),fill='#26272a')
    text(1123,74,'판독 카드',11,MUTED,True);text(1337,75,'qwen3.5:4b',9,MUTED)
    if state['card']>0.5:
        card=case.last if state['answer']==2 else case.first
        text(1126,109,'재질문 뒤 답' if state['answer']==2 else '첫 답',10,ACCENT,True)
        budget=state['typed']
        count=max(0,min(len(card['summary']),budget));budget-=len(card['summary'])
        y=140
        for line in wrap(card['summary'][:count],292,13):text(1126,y,line,13);y+=24
        y=140+24*len(wrap(card['summary'],292,13))+17
        if budget>0:
            d.line((1125,y,1426,y),fill=LINE);y+=13
            text(1126,y,'원인 후보 · 출처 근거',10,MUTED,True);y+=25
            for cid in card['cause_ids']:
                ct=causes[cid]
                for line in wrap(ct['cause'],292,12,True):text(1126,y,line,12,INK,True);y+=22
                for line in wrap(f'“{ct["quote"]}” [{ct["source"]}]',281,10):text(1134,y,line,10,MUTED);y+=17
                y+=12
            d.line((1125,y,1426,y),fill=LINE);y+=14
            text(1126,y,'점검 순서',10,MUTED,True);y+=26
            for i,step in enumerate(card['check_order']):
                count=max(0,min(len(step),budget));budget-=len(step)
                if count==0:break
                for j,line in enumerate(wrap(step[:count],271,12)):
                    if y>722:raise ValueError('Recorded reading card exceeds properties dock')
                    text(1126 if j==0 else 1144,y,f'{i+1}. {line}' if j==0 else line,12)
                    if step in case.dropped and state['drop_hl']>0 and state['answer']==1:
                        d.line((1144,y+17,1424,y+17),fill=BAD)
                    y+=22
                y+=6
    if state['check']:
        check=state['check'];passed=check=='pass';failed=check in ('fail','reask')
        fill='#2c4543' if passed else '#442d2c' if failed else FIELD
        color=ACCENT if passed else BAD if failed else MUTED
        box((1124,759,1426,843),fill,None,3)
        label='검사 통과' if passed else '재질문 중' if check=='reask' else '검사에 걸림' if failed else '코드 검사 중'
        text(1137,773,label,13,color,True)
        note=f'{"재질문 1회 뒤" if case.retry else "첫 답 그대로"} · {case.r["seconds"]}초' if passed else case.r['attempts'][0]['problems'][0] if failed else '패턴 · 위치 · 원인 후보 대조'
        for j,line in enumerate(wrap(note,273,10)):text(1137,798+j*16,line,10,color)
    # App status bar; only actual recorded outcomes are represented.
    d.rectangle((0,875,1440,900),fill='#17181a');d.line((0,875,1440,875),fill=LINE)
    text(9,884,f'선택: {c["pattern"]}',9,MUTED)
    text(224,884,'WM-811K · 분류 / 위치 / 유사 사례',9,MUTED)
    text(1180,884,'코드 검사 통과' if state['check']=='pass' else '기록된 판독 카드',10,ACCENT)
    return im
