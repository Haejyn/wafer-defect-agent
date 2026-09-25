"""WM-811K 첫 진단 — 라벨 분포, 공식 분할의 로트 겹침, 맵 크기, 분할 사이 똑같은 맵."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_pickle(ROOT / "data/raw/MIR-WM811K/Python/WM811K.pkl")

out = {}
out["rows"] = len(df)
out["failureType"] = df["failureType"].astype(str).value_counts().to_dict()
out["trainTestLabel"] = df["trainTestLabel"].astype(str).value_counts().to_dict()

labeled = df[df["trainTestLabel"].isin(["Training", "Test"])].copy()
out["labeled_by_split_type"] = (
    labeled.groupby(["trainTestLabel", "failureType"]).size().unstack(fill_value=0).to_dict(orient="index")
)

train_lots = set(labeled.loc[labeled.trainTestLabel == "Training", "lotName"])
test_lots = set(labeled.loc[labeled.trainTestLabel == "Test", "lotName"])
out["lots"] = {"train": len(train_lots), "test": len(test_lots), "shared": len(train_lots & test_lots)}

shapes = Counter(np.asarray(m).shape for m in labeled["waferMap"])
out["labeled_shapes_distinct"] = len(shapes)
out["labeled_shapes_top10"] = {str(k): v for k, v in shapes.most_common(10)}
vals = Counter()
for m in labeled["waferMap"].iloc[:2000]:
    vals.update(np.unique(np.asarray(m)).tolist())
out["pixel_values_sample"] = dict(vals)


def digest(m):
    a = np.ascontiguousarray(np.asarray(m, dtype=np.uint8))
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


labeled["h"] = labeled["waferMap"].map(digest)
tr = labeled[labeled.trainTestLabel == "Training"]
te = labeled[labeled.trainTestLabel == "Test"]
shared_h = set(tr.h) & set(te.h)
out["exact_dup_maps"] = {
    "within_labeled": int(labeled.h.duplicated().sum()),
    "test_maps_also_in_train": int(te.h.isin(shared_h).sum()),
    "test_size": len(te),
}
dup_te = te[te.h.isin(shared_h)]
out["test_dups_by_type"] = dup_te.failureType.astype(str).value_counts().to_dict()
conflict = labeled.groupby("h").failureType.nunique()
out["same_map_different_label_groups"] = int((conflict > 1).sum())

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/eda.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
