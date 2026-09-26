"""분류기 학습·평가. 분할 이름과 '학습에서 뺄 클래스'를 받아 같은 절차로 돈다."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import confusion, macro_f1, per_class_recall
from .model import build, one_hot
from .prep import CLASSES


@dataclass
class Config:
    split: str = "lot"            # lot · wafer · official
    exclude: int | None = None    # 학습·검증에서 뺄 클래스 id (처음 보는 패턴 실험)
    seed: int = 0
    epochs: int = 20
    batch: int = 256
    none_per_epoch: int = 20000   # 에폭마다 none 에서 뽑는 수 (나머지 불량은 전부)
    lr: float = 2e-3
    arch: str = "cnn"             # cnn · resnet18


def augment(x: torch.Tensor) -> torch.Tensor:
    """웨이퍼 패턴은 회전·반전에 대해 같은 종류다 — 배치 전체에 같은 90° 회전·반전."""
    k = int(torch.randint(0, 4, (1,)))
    x = torch.rot90(x, k, dims=(1, 2))
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=(2,))
    return x


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch=1024):
    model.eval()
    logits, embs = [], []
    for i in range(0, len(X), batch):
        xb = one_hot(torch.from_numpy(X[i:i + batch]).to(device))
        e = model.embed(xb)
        logits.append(model.head(e).cpu())
        embs.append(e.cpu())
    return torch.cat(logits).numpy(), torch.cat(embs).numpy()


def run(X: np.ndarray, y: np.ndarray, split: np.ndarray, cfg: Config, device="cuda", log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    keep = np.ones(len(y), bool) if cfg.exclude is None else (y != cfg.exclude)
    tr = np.flatnonzero((split == 0) & keep)
    if (split == 1).any():
        va = np.flatnonzero((split == 1) & keep)
    else:  # 공식 분할에는 val 이 없다 — 학습 로트의 15 % 를 떼어 쓴다 (호출 쪽에서 split 을 이미 나눠 준다)
        raise ValueError("val split required")
    te = np.flatnonzero(split == 2)

    classes = [c for c in range(len(CLASSES)) if c != cfg.exclude]
    remap = np.full(len(CLASSES), -1)
    remap[classes] = np.arange(len(classes))
    n_out = len(classes)

    tr_none, tr_def = tr[y[tr] == 0], tr[y[tr] != 0]
    sampled_counts = np.bincount(remap[y[tr_def]], minlength=n_out).astype(float)
    sampled_counts[0] = min(cfg.none_per_epoch, len(tr_none))
    w = 1.0 / np.sqrt(np.maximum(sampled_counts, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    model = build(cfg.arch, n_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    steps = cfg.epochs * ((min(cfg.none_per_epoch, len(tr_none)) + len(tr_def)) // cfg.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps)
    Xg = torch.from_numpy(X).to(device)  # 64×64 uint8 17만 장 ≈ 0.7 GB, GPU 에 올려 둔다
    yg = torch.from_numpy(remap[y]).to(device)

    best = (-1.0, None, -1)
    for ep in range(cfg.epochs):
        t0 = time.time()
        model.train()
        idx = np.concatenate([rng.choice(tr_none, min(cfg.none_per_epoch, len(tr_none)), replace=False), tr_def])
        rng.shuffle(idx)
        for i in range(0, len(idx), cfg.batch):
            b = torch.from_numpy(idx[i:i + cfg.batch]).to(device)
            xb = one_hot(augment(Xg[b]))
            loss = F.cross_entropy(model(xb), yg[b], weight=w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        lv, _ = predict(model, X[va], device)
        f1 = macro_f1(remap[y[va]], lv.argmax(1), n_out)
        log(f"ep {ep + 1:02d} loss {loss.item():.3f} val macroF1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    lt, et = predict(model, X[te], device)
    result = {"best_epoch": best[2], "val_macro_f1": round(best[0], 4), "classes": [CLASSES[c] for c in classes]}
    if cfg.exclude is None:
        pred = lt.argmax(1)
        yt = remap[y[te]]
        result.update(
            test_macro_f1=round(macro_f1(yt, pred, n_out), 4),
            test_recall={CLASSES[c]: round(r, 4) for c, r in zip(classes, per_class_recall(yt, pred, n_out))},
            confusion=confusion(yt, pred, n_out).tolist(),
            n_test=int(len(te)),
        )
    return model, result, (te, lt, et)
