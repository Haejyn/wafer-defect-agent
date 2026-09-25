"""로트 안 패턴 동질성 — 웨이퍼 단위 무작위 분할이 얼마나 낙관적일 수 있는지 가늠한다."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_pickle(ROOT / "data/raw/MIR-WM811K/Python/WM811K.pkl")
lab = df[df["trainTestLabel"].isin(["Training", "Test"])].copy()
lab["failureType"] = lab["failureType"].astype(str)

out = {}
per_lot = lab.groupby("lotName")
out["labeled_lots"] = int(per_lot.ngroups)
out["wafers_per_lot"] = per_lot.size().describe().round(2).to_dict()

defect = lab[lab.failureType != "none"]
g = defect.groupby("lotName").failureType
sizes = g.size()
multi = sizes[sizes >= 2].index
d = defect[defect.lotName.isin(multi)]
# 같은 로트의 다른 불량 웨이퍼가 같은 패턴일 확률 (무작위 분할에서 '이웃'이 정답을 알려 줄 확률의 대략값)
same = 0
total = 0
for _, s in d.groupby("lotName").failureType:
    vc = s.value_counts()
    n = len(s)
    same += int((vc * (vc - 1)).sum())
    total += n * (n - 1)
out["defect_lots_with_2plus"] = int(len(multi))
out["p_same_pattern_within_lot_pair"] = round(same / total, 4) if total else None
out["defect_lot_dominant_share_mean"] = round(float(g.agg(lambda s: s.value_counts(normalize=True).iloc[0])[multi].mean()), 4)
out["label_count_per_type_in_lot_median"] = defect.groupby(["failureType", "lotName"]).size().groupby("failureType").median().to_dict()

# 로트 번호 순서와 공식 분할의 관계 (시간순 분할인지)
lab["lotnum"] = lab.lotName.str.extract(r"(\d+)").astype(float)
out["lotnum_range_by_split"] = lab.groupby("trainTestLabel").lotnum.agg(["min", "max", "median"]).to_dict(orient="index")

(ROOT / "reports/eda_lot.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
