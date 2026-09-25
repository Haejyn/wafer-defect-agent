"""웨이퍼 맵 전처리 — 크기 통일(값 보존), 중복 제거, 분할."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SIZE = 64


def resize_nearest(m, size: int = SIZE) -> np.ndarray:
    """최근접 리사이즈. 보간을 하지 않으므로 값(0 웨이퍼 밖·1 정상·2 불량)이 그대로 남는다."""
    a = np.asarray(m, dtype=np.uint8)
    h, w = a.shape
    rows = np.minimum(((np.arange(size) + 0.5) * h / size).astype(int), h - 1)
    cols = np.minimum(((np.arange(size) + 0.5) * w / size).astype(int), w - 1)
    return a[rows][:, cols]


def digest(m) -> str:
    a = np.ascontiguousarray(np.asarray(m, dtype=np.uint8))
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """똑같은 맵은 하나만 남기고, 같은 맵에 라벨이 갈리는 묶음은 모두 뺀다."""
    h = df["waferMap"].map(digest)
    n_labels = df.groupby(h)["failureType"].transform("nunique")
    conflict = n_labels > 1
    kept = df[~conflict].loc[~h[~conflict].duplicated()]
    info = {
        "input": len(df),
        "dropped_conflict_rows": int(conflict.sum()),
        "dropped_conflict_groups": int(h[conflict].nunique()),
        "dropped_duplicates": int(len(df) - int(conflict.sum()) - len(kept)),
        "kept": len(kept),
    }
    return kept, info


def lot_split(lots: pd.Series, y: np.ndarray, frac=(0.7, 0.15, 0.15), seed: int = 0) -> np.ndarray:
    """로트 단위 분할. 로트마다 '가장 드문 불량 종류'를 대표로 삼아 그 묶음 안에서 나눠 클래스 비율을 맞춘다.
    반환: 0 train · 1 val · 2 test (웨이퍼마다)."""
    rng = np.random.default_rng(seed)
    freq = np.bincount(y, minlength=len(CLASSES)).astype(float)
    frame = pd.DataFrame({"lot": lots.to_numpy(), "y": y})
    rarest = frame[frame.y != 0].groupby("lot").y.agg(lambda s: min(s, key=lambda c: freq[c]))
    key = frame.lot.drop_duplicates().to_frame().set_index("lot")
    key["k"] = rarest.reindex(key.index).fillna(0).astype(int)
    assign = {}
    for _, grp in key.groupby("k"):
        names = grp.index.to_numpy().copy()
        rng.shuffle(names)
        n = len(names)
        a, b = int(round(n * frac[0])), int(round(n * (frac[0] + frac[1])))
        for i, name in enumerate(names):
            assign[name] = 0 if i < a else (1 if i < b else 2)
    return frame.lot.map(assign).to_numpy()


def wafer_split(y: np.ndarray, frac=(0.7, 0.15, 0.15), seed: int = 0) -> np.ndarray:
    """웨이퍼 단위 무작위 분할(클래스 층화). 누수 실험의 비교 대상 — 같은 로트가 여러 분할에 퍼진다."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=np.int8)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(round(n * frac[0])), int(round(n * (frac[0] + frac[1])))
        out[idx[:a]], out[idx[a:b]], out[idx[b:]] = 0, 1, 2
    return out
