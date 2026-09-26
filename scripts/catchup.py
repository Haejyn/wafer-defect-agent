"""고도화 B — 나중 로트 따라잡기. 공식 분할의 시험 로트(나중)를 로트 번호순으로 R 묶음으로 흘려보낸다.
묶음마다: 현재 모델로 먼저 잰다(prequential) → 그 묶음에서 라벨 예산만큼 골라 라벨을 달고 → 학습 자료에 더해 짧게 추가 학습 → 다음 묶음.
고르는 방법: none(업데이트 없음) · finetune_only(라벨 없이 추가 학습만 — 절차가 흔드는 몫) · random · uncertainty(1−최대 확률) · novelty(라벨 있는 자료와의 임베딩 kNN 거리) · agent(둘의 순위 평균)
v1(09-26): lr 5e-4 · 전체 층 추가 학습이 모델을 흔들어 모든 방법이 업데이트 없음보다 나빴다 → v2 는 --lr 1e-4 · --train-last(b4·head 만) 로 부드럽게.
사용: python scripts/catchup.py --budget 0.01 --rounds 5 --epochs 2 --seed 0"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.metrics import macro_f1, per_class_recall  # noqa: E402
from wafer.model import WaferCNN, one_hot  # noqa: E402
from wafer.ood import feature_maps, knn_distance  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402
from wafer.train import augment, predict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=float, default=0.01)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--dup", type=int, default=5, help="새로 라벨 단 웨이퍼를 에폭마다 몇 번 넣나")
ap.add_argument("--strategies", default="none,finetune_only,random,uncertainty,novelty,agent")
ap.add_argument("--train-last", action="store_true", help="마지막 블록(b4)과 head 만 학습")
ap.add_argument("--tag", default="")
args = ap.parse_args()
device = "cuda"

d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, lot, off = d["X"], d["y"], d["lot"], d["split_official"]
lotnum = np.array([int("".join(ch for ch in s if ch.isdigit()) or 0) for s in lot])
train0 = np.flatnonzero(off == 0)
stream = np.flatnonzero(off == 2)
stream = stream[np.argsort(lotnum[stream], kind="stable")]
bounds = np.linspace(0, len(stream), args.rounds + 1).astype(int)
# 묶음 경계를 로트 경계에 맞춘다 — 한 로트가 두 묶음에 나뉘지 않게
for k in range(1, args.rounds):
    b = bounds[k]
    while b < len(stream) and lotnum[stream[b]] == lotnum[stream[b - 1]]:
        b += 1
    bounds[k] = b
chunks = [stream[bounds[k]:bounds[k + 1]] for k in range(args.rounds)]
Xg = torch.from_numpy(X).to(device)
yg = torch.from_numpy(y).to(device)


def evaluate(model, idx):
    lg, _ = predict(model, X[idx], device)
    p = lg.argmax(1)
    return {"macro_f1": round(macro_f1(y[idx], p, 9), 4),
            "defect_recall_mean": round(float(np.mean([r for c, r in enumerate(per_class_recall(y[idx], p, 9)) if c > 0 and (y[idx] == c).any()])), 4)}, lg


def finetune(model, labeled, added, rng):
    params = [p for n, p in model.named_parameters() if (not args.train_last) or n.startswith(("b4.", "head."))]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    none_pool, def_pool = labeled[y[labeled] == 0], labeled[y[labeled] != 0]
    counts = np.bincount(y[np.concatenate([def_pool, np.repeat(added, args.dup)])], minlength=9).astype(float)
    counts[0] = min(20000, len(none_pool))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    for _ in range(args.epochs):
        model.train()
        if args.train_last:  # 얼린 층의 BatchNorm 통계도 움직이지 않게
            for n, m in model.named_modules():
                if isinstance(m, torch.nn.BatchNorm2d) and not n.startswith("b4"):
                    m.eval()
        idx = np.concatenate([rng.choice(none_pool, min(20000, len(none_pool)), replace=False), def_pool, np.repeat(added, args.dup)])
        rng.shuffle(idx)
        for i in range(0, len(idx), 256):
            b = torch.from_numpy(idx[i:i + 256]).to(device)
            loss = F.cross_entropy(model(one_hot(augment(Xg[b]))), yg[b], weight=w, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def select(strategy, model, chunk, logits, labeled, k, rng):
    if strategy == "random":
        return rng.choice(chunk, k, replace=False)
    unc = 1 - torch.softmax(torch.from_numpy(logits), 1).max(1).values.numpy()
    if strategy == "uncertainty":
        return chunk[np.argsort(-unc)[:k]]
    ref = np.concatenate([rng.choice(labeled[y[labeled] == c], min(3000, (y[labeled] == c).sum()), replace=False)
                          for c in range(9) if (y[labeled] == c).any()])
    _, e_ref = feature_maps(model, X[ref], device)
    _, e_q = feature_maps(model, X[chunk], device)
    nov = knn_distance(e_q, e_ref, k=5).numpy()
    if strategy == "novelty":
        return chunk[np.argsort(-nov)[:k]]
    rank = np.argsort(np.argsort(-unc)) + np.argsort(np.argsort(-nov))  # agent: 두 순위의 합이 작은 것
    return chunk[np.argsort(rank)[:k]]


results = {"date": str(date.today()), "args": vars(args), "chunks": [
    {"n": int(len(c)), "lots": [int(lotnum[c].min()), int(lotnum[c].max())], "defect": int((y[c] > 0).sum())} for c in chunks], "runs": {}}
for strategy in args.strategies.split(","):
    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = WaferCNN(9).to(device)
    model.load_state_dict(torch.load(ROOT / f"runs/official_s{args.seed}.pt", map_location=device))
    labeled = train0.copy()
    added = np.array([], dtype=int)
    per_round = []
    for r, chunk in enumerate(chunks):
        m, lg = evaluate(model, chunk)
        per_round.append(m)
        if strategy != "none" and r < len(chunks) - 1:
            k = max(1, int(round(args.budget * len(chunk))))
            pick = np.array([], dtype=int) if strategy == "finetune_only" else select(strategy, model, chunk, lg, labeled, k, rng)
            added = np.concatenate([added, pick])
            labeled = np.concatenate([labeled, pick])
            finetune(model, labeled, added, rng)
    res = {"per_round": per_round, "labels_added": int(len(added)),
           "added_defect_share": round(float((y[added] > 0).mean()), 3) if len(added) else None,
           "mean_macro_f1_rounds_2_on": round(float(np.mean([p["macro_f1"] for p in per_round[1:]])), 4),
           "minutes": round((time.time() - t0) / 60, 1)}
    results["runs"][strategy] = res
    print(strategy, json.dumps(res, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()

out = ROOT / f"reports/catchup_b{args.budget}_s{args.seed}{args.tag}.json"
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
