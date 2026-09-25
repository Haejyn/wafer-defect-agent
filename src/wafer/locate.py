"""불량 다이의 위치를 코드로 계산한다 — 에이전트가 말한 위치를 이 값과 대조한다.
위치 어휘: 중심 · 도넛 · 가장자리 · 링 · 선 · 국부 · 전면 · 산발 · 없음

실제 맵에는 배경에 흩어진 불량 다이가 많다. 그래서 작은 덩어리(배경 잡음)를 걸러 낸 뒤
남은 '계통 불량' 덩어리의 반지름 위치·각도 범위·길쭉함으로 위치를 정한다."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

LOCATIONS = ["중심", "도넛", "가장자리", "링", "선", "국부", "전면", "산발", "없음"]
EIGHT = np.ones((3, 3))


def systematic_mask(m: np.ndarray, min_cluster_frac: float = 0.004, min_cluster_px: int = 12) -> np.ndarray:
    """배경 잡음(작은 덩어리)을 뺀 계통 불량 다이 마스크 — 히트맵이 제자리를 짚는지 채점할 때 정답으로 쓴다."""
    wafer, fail = m > 0, m == 2
    lab, n = ndimage.label(fail, structure=EIGHT)
    if n == 0:
        return np.zeros_like(fail)
    sizes = ndimage.sum(fail, lab, range(1, n + 1))
    keep_min = max(min_cluster_px, int(min_cluster_frac * wafer.sum()))
    return np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= keep_min])


def describe(m: np.ndarray, min_cluster_frac: float = 0.004, min_cluster_px: int = 12) -> dict:
    """m: 값 0(밖)·1(정상)·2(불량) 맵. 위치 이름과 근거 수치."""
    wafer = m > 0
    fail = m == 2
    n_die, n_fail = int(wafer.sum()), int(fail.sum())
    if n_die == 0 or n_fail == 0:
        return {"location": "없음", "fail_ratio": 0.0}
    fail_ratio = n_fail / n_die

    yy, xx = np.nonzero(wafer)
    cy, cx = yy.mean(), xx.mean()
    rmax = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).max() + 0.5

    lab, n = ndimage.label(fail, structure=EIGHT)
    sizes = ndimage.sum(fail, lab, range(1, n + 1)) if n else np.zeros(0)
    keep_min = max(min_cluster_px, int(min_cluster_frac * n_die))
    kept_ids = [i + 1 for i, s in enumerate(sizes) if s >= keep_min]
    sys_mask = np.isin(lab, kept_ids)
    n_sys = int(sys_mask.sum())
    feats = {"fail_ratio": round(fail_ratio, 4), "n_clusters": int(n), "n_systematic_clusters": len(kept_ids),
             "systematic_share": round(n_sys / n_fail, 3)}

    if fail_ratio > 0.5:
        feats["location"] = "전면"
        return feats
    if n_sys == 0:
        feats["location"] = "산발"
        return feats

    big = kept_ids[int(np.argmax([sizes[i - 1] for i in kept_ids]))]
    by, bx = np.nonzero(lab == big)
    r = np.sqrt((by - cy) ** 2 + (bx - cx) ** 2) / rmax
    ang = np.arctan2(by - cy, bx - cx)
    sectors = np.unique(((ang + np.pi) / (2 * np.pi) * 12).astype(int) % 12).size
    elong = 1.0
    thin = 1.0
    if len(by) >= 5:
        ev = np.linalg.eigvalsh(np.cov(np.vstack([by, bx])))
        elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
        thin = float(len(by) / (np.sqrt(max(ev[1], 1e-9)) * 4 + 1))  # 길이당 넓이 — 긁힘은 가늘다
    feats.update(
        largest_px=int(len(by)), largest_r_median=round(float(np.median(r)), 3), largest_r_min=round(float(r.min()), 3),
        largest_sectors_of_12=int(sectors), largest_elongation=round(elong, 2), largest_thickness=round(thin, 2),
    )
    rm = float(np.median(r))
    if elong >= 3.5 and thin <= 4 and sectors <= 6:
        loc = "선"
    elif rm >= 0.72 and sectors >= 8:
        loc = "링"
    elif rm >= 0.72:
        loc = "가장자리"
    elif rm <= 0.3 and r.min() < 0.15:
        loc = "중심"
    elif 0.25 < rm < 0.72 and sectors >= 8 and r.min() > 0.12:
        loc = "도넛"
    else:
        loc = "국부"
    feats["location"] = loc
    return feats
