"""81만 장 전체(라벨 없는 것 포함)를 64×64 uint8 로 캐시 — 지도·검색용. data/proc/all64.npy (memmap) + all_meta.npz"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wafer.prep import CLASS_TO_ID, resize_nearest  # noqa: E402

df = pd.read_pickle(ROOT / "data/raw/MIR-WM811K/Python/WM811K.pkl")
out = ROOT / "data/proc"
out.mkdir(parents=True, exist_ok=True)
mm = np.lib.format.open_memmap(out / "all64.npy", mode="w+", dtype=np.uint8, shape=(len(df), 64, 64))
for i, m in enumerate(df["waferMap"]):
    mm[i] = resize_nearest(m)
mm.flush()
ft = df["failureType"].astype(str)
y = ft.map(CLASS_TO_ID).fillna(-1).astype(np.int64).to_numpy()  # 라벨 없음 = -1
np.savez_compressed(out / "all_meta.npz", y=y, lot=df["lotName"].astype(str).to_numpy(),
                    wafer_index=df["waferIndex"].to_numpy(), die_size=df["dieSize"].to_numpy(),
                    official=df["trainTestLabel"].astype(str).to_numpy())
print("saved", mm.shape, "labeled", int((y >= 0).sum()))
