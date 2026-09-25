"""처음 보는 패턴 점수 — 분류기 불확실도(MSP·에너지), 임베딩 kNN, 패치 kNN(PatchCore 식).
점수는 모두 '클수록 처음 보는 것'."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import one_hot


def msp_score(logits: np.ndarray) -> np.ndarray:
    p = torch.softmax(torch.from_numpy(logits), 1)
    return (1 - p.max(1).values).numpy()


def energy_score(logits: np.ndarray) -> np.ndarray:
    return (-torch.logsumexp(torch.from_numpy(logits), 1)).numpy()


@torch.no_grad()
def feature_maps(model, X: np.ndarray, device, batch=512, dtype=torch.float16):
    """(N,C,8,8) 특징맵과 (N,C) 임베딩."""
    model.eval()
    maps, embs = [], []
    for i in range(0, len(X), batch):
        f = model.features(one_hot(torch.from_numpy(X[i:i + batch]).to(device)))
        maps.append(f.to(dtype).cpu())
        embs.append(f.mean((2, 3)).cpu())
    return torch.cat(maps), torch.cat(embs)


@torch.no_grad()
def knn_distance(query: torch.Tensor, bank: torch.Tensor, k: int = 1, device="cuda", chunk=8192) -> torch.Tensor:
    """코사인 거리(1 - cos)로 k 번째 최근접까지의 평균 거리."""
    b = F.normalize(bank.to(device).float(), dim=1).half()
    out = []
    for i in range(0, len(query), chunk):
        q = F.normalize(query[i:i + chunk].to(device).float(), dim=1).half()
        sim = q @ b.T
        top = sim.topk(k, dim=1).values.float()
        out.append((1 - top).mean(1).cpu())
    return torch.cat(out)


def patch_bank(maps: torch.Tensor, max_patches: int, seed: int = 0) -> torch.Tensor:
    """학습 웨이퍼의 패치(8×8 위치마다 C차원)를 무작위로 max_patches 개까지 모은다 (PatchCore 의 코어셋 대신 무작위 부분표본)."""
    n, c, h, w = maps.shape
    flat = maps.permute(0, 2, 3, 1).reshape(-1, c)
    g = torch.Generator().manual_seed(seed)
    if len(flat) > max_patches:
        flat = flat[torch.randperm(len(flat), generator=g)[:max_patches]]
    return flat


@torch.no_grad()
def patch_scores(maps: torch.Tensor, bank: torch.Tensor, device="cuda", batch=256) -> tuple[np.ndarray, np.ndarray]:
    """웨이퍼마다 패치별 최근접 거리 지도(8×8)와 그 최댓값(이미지 점수)."""
    n, c, h, w = maps.shape
    b = F.normalize(bank.to(device).float(), dim=1).half()
    heat = []
    for i in range(0, n, batch):
        q = maps[i:i + batch].permute(0, 2, 3, 1).reshape(-1, c).to(device).float()
        q = F.normalize(q, dim=1).half()
        d = 1 - (q @ b.T).max(1).values.float()
        heat.append(d.reshape(-1, h, w).cpu())
    heat = torch.cat(heat).numpy()
    return heat, heat.reshape(n, -1).max(1)
