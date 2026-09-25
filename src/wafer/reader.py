"""판독기 — 분류 모델 하나 + 처음 보는 패턴 점수(임베딩 kNN) + 유사 사례 검색 + 위치 계산으로 판독 사례(case)를 만든다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .agent import UNKNOWN
from .locate import describe
from .model import WaferCNN
from .ood import feature_maps
from .prep import CLASSES
from .train import predict

ROOT = Path(__file__).resolve().parents[2]


class Reader:
    """한 분류 모델 + 이상 점수(임베딩 kNN) + 유사 사례 검색."""

    def __init__(self, ckpt: str, exclude: int | None, X, y, split, device="cuda", rng=None):
        rng = rng if rng is not None else np.random.default_rng(0)  # 평가 스크립트는 사례 뽑기와 같은 생성기를 넘긴다(재현)
        self.X, self.y, self.device = X, y, device
        self.classes = [c for c in range(9) if c != exclude]
        self.model = WaferCNN(len(self.classes)).to(device)
        self.model.load_state_dict(torch.load(ROOT / f"runs/{ckpt}.pt", map_location=device))
        tr = np.flatnonzero((split == 0) & (y != (exclude if exclude is not None else -1)))
        self.db = np.concatenate([rng.choice(tr[y[tr] == k], min(3000, (y[tr] == k).sum()), replace=False) for k in self.classes])
        _, db_emb = feature_maps(self.model, X[self.db], device)
        self.db_emb = F.normalize(db_emb.float().to(device), dim=1)
        va = np.flatnonzero((split == 1) & (y != (exclude if exclude is not None else -1)))
        va = rng.choice(va, min(8000, len(va)), replace=False)
        self.val_scores = np.sort(self.score(X[va])[0])
        self.threshold = float(np.quantile(self.val_scores, 0.95))  # 아는 패턴 검증 웨이퍼 기준 오경보 5 %

    def score(self, Xs):
        _, e = feature_maps(self.model, Xs, self.device)
        sim = F.normalize(e.float().to(self.device), dim=1) @ self.db_emb.T
        top = sim.topk(5, dim=1)
        return (1 - top.values.mean(1)).cpu().numpy(), top.values.cpu().numpy(), top.indices.cpu().numpy()

    def case(self, i: int) -> dict:
        return self.case_from_map(self.X[i], row=int(i), true=CLASSES[self.y[i]])

    def case_from_map(self, m: np.ndarray, row: int = -1, true: str = "?") -> dict:
        """64×64(값 0/1/2) 맵 한 장의 판독 사례."""
        X, y = self.X, self.y
        logits, _ = predict(self.model, m[None], self.device)
        p = torch.softmax(torch.from_numpy(logits), 1)[0].numpy()
        s, sims, nn = self.score(m[None])
        unknown = bool(s[0] > self.threshold)
        pred = CLASSES[self.classes[int(p.argmax())]]
        loc = describe(m)
        return {
            "row": row, "true": true, "classifier_pred": pred,
            "pattern": UNKNOWN if unknown else pred, "confidence": float(p.max()),
            "unknown_flag": unknown, "ood_percentile": float(np.searchsorted(self.val_scores, s[0]) / len(self.val_scores) * 100),
            "location": loc["location"], "location_features": loc, "fail_ratio": float(loc["fail_ratio"]),
            "similar": [{"pattern": CLASSES[y[self.db[j]]], "similarity": float(v), "row": int(self.db[j])} for j, v in zip(nn[0], sims[0])],
        }


