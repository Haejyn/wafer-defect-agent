import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer.prep import dedupe, lot_split, resize_nearest, wafer_split  # noqa: E402


@settings(max_examples=200, deadline=None)
@given(arrays(np.uint8, st.tuples(st.integers(5, 90), st.integers(5, 90)), elements=st.integers(0, 2)))
def test_resize_keeps_only_original_values(m):
    r = resize_nearest(m, 64)
    assert r.shape == (64, 64)
    assert set(np.unique(r)) <= set(np.unique(m))


def test_resize_identity_at_same_size():
    m = np.random.default_rng(0).integers(0, 3, (64, 64)).astype(np.uint8)
    assert np.array_equal(resize_nearest(m, 64), m)


def test_resize_upsamples_blocks():
    m = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    r = resize_nearest(m, 4)
    assert np.array_equal(r, np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]]))


def test_dedupe_keeps_one_copy_and_drops_label_conflicts():
    a, b, c = np.zeros((3, 3)), np.ones((3, 3)), np.full((3, 3), 2)
    df = pd.DataFrame({
        "waferMap": [a, a, b, b, c],
        "failureType": ["none", "none", "Loc", "Scratch", "Center"],
    })
    kept, info = dedupe(df)
    assert len(kept) == 2  # a 하나, c 하나 — b 는 라벨이 갈려 둘 다 빠진다
    assert info["dropped_conflict_groups"] == 1 and info["dropped_conflict_rows"] == 2
    assert info["dropped_duplicates"] == 1


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 10_000))
def test_lot_split_never_shares_a_lot(seed):
    rng = np.random.default_rng(seed)
    lots = pd.Series(rng.integers(0, 60, 800).astype(str))
    y = rng.integers(0, 9, 800)
    s = lot_split(lots, y, seed=seed)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert not set(lots[s == a]) & set(lots[s == b])
    assert set(np.unique(s)) <= {0, 1, 2}


def test_wafer_split_is_stratified_and_covers_all():
    y = np.repeat(np.arange(9), 100)
    s = wafer_split(y, seed=1)
    assert len(s) == len(y)
    for c in range(9):
        assert np.bincount(s[y == c], minlength=3).tolist() == [70, 15, 15]


def test_saved_lot_split_has_no_shared_lots():
    """실제 캐시에서도 로트 분할의 train·test 가 로트를 공유하지 않는다."""
    p = Path(__file__).resolve().parents[1] / "data/proc/labeled64.npz"
    if not p.exists():
        import pytest
        pytest.skip("data/proc 없음")
    d = np.load(p, allow_pickle=True)
    lot, s = d["lot"], d["split_lot"]
    assert not set(lot[s == 0]) & set(lot[s == 2])
    assert not set(lot[s == 0]) & set(lot[s == 1])
