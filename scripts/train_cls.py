"""분류기 한 번 학습. 예: python scripts/train_cls.py --split lot
   --exclude Scratch 이면 그 클래스를 학습·검증에서 빼고, 시험 로짓·임베딩을 저장한다(3단계용)."""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.prep import CLASS_TO_ID  # noqa: E402
from wafer.train import Config, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="lot", choices=["lot", "wafer", "official"])
ap.add_argument("--exclude", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"])
ap.add_argument("--lr", type=float, default=2e-3)
args = ap.parse_args()

d = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
X, y, lot = d["X"], d["y"], d["lot"]
split = d[f"split_{args.split}"].copy()
if args.split == "official":
    # 공식 분할엔 val 이 없다 — 학습 로트의 15 % 를 로트 단위로 떼어 val 로 쓴다
    rng = np.random.default_rng(args.seed)
    train_lots = np.unique(lot[split == 0])
    val_lots = set(rng.choice(train_lots, int(len(train_lots) * 0.15), replace=False))
    split[(split == 0) & np.isin(lot, list(val_lots))] = 1

ex = None if args.exclude is None else CLASS_TO_ID[args.exclude]
tag = ("" if args.arch == "cnn" else f"{args.arch}_") + f"{args.split}_s{args.seed}" + (f"_ex-{args.exclude}" if ex is not None else "")
cfg = Config(split=args.split, exclude=ex, seed=args.seed, epochs=args.epochs, arch=args.arch, lr=args.lr)
t0 = time.time()
lines = []
def log(msg):
    print(msg, flush=True)
    lines.append(msg)
model, result, (te, logits, emb) = run(X, y, split, cfg, log=log)
result.update(tag=tag, date=str(date.today()), minutes=round((time.time() - t0) / 60, 1), config=vars(cfg), train_log=lines)

(ROOT / "runs").mkdir(exist_ok=True)
torch.save(model.state_dict(), ROOT / f"runs/{tag}.pt")
np.savez_compressed(ROOT / f"runs/{tag}_test.npz", idx=te, logits=logits, emb=emb)
out = ROOT / "reports/cls"
out.mkdir(parents=True, exist_ok=True)
(out / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("confusion", "train_log", "config")}, ensure_ascii=False, indent=2))
