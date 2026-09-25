"""라벨 있는 맵을 64×64 로 캐시하고 세 가지 분할(로트 · 웨이퍼 무작위 · 공식)을 저장한다."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.prep import CLASS_TO_ID, dedupe, lot_split, resize_nearest, wafer_split  # noqa: E402

df = pd.read_pickle(ROOT / "data/raw/MIR-WM811K/Python/WM811K.pkl")
lab = df[df["trainTestLabel"].isin(["Training", "Test"])].copy()
lab["failureType"] = lab["failureType"].astype(str)
lab["trainTestLabel"] = lab["trainTestLabel"].astype(str)
kept, info = dedupe(lab)

X = np.stack([resize_nearest(m) for m in kept["waferMap"]])
y = kept["failureType"].map(CLASS_TO_ID).to_numpy().astype(np.int64)
lots = kept["lotName"].astype(str)
split_lot = lot_split(lots, y, seed=0)
split_wafer = wafer_split(y, seed=0)
split_official = np.where(kept["trainTestLabel"].to_numpy() == "Training", 0, 2).astype(np.int8)

out = ROOT / "data/proc"
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out / "labeled64.npz",
    X=X, y=y, lot=lots.to_numpy(), src_index=kept.index.to_numpy(),
    die_size=kept["dieSize"].to_numpy(), split_lot=split_lot, split_wafer=split_wafer, split_official=split_official,
)

def counts(split):
    return {name: np.bincount(y[split == s], minlength=9).tolist() for s, name in enumerate(["train", "val", "test"])}

info["splits"] = {"lot": counts(split_lot), "wafer": counts(split_wafer), "official": counts(split_official)}
lt = pd.Series(lots.to_numpy())
info["lot_overlap"] = {
    name: int(len(set(lt[sp == 0]) & set(lt[sp == 2]))) for name, sp in [("lot", split_lot), ("wafer", split_wafer), ("official", split_official)]
}
(ROOT / "reports/prep.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(info, ensure_ascii=False, indent=2))
