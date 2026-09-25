import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer.locate import describe  # noqa: E402

N = 64
yy, xx = np.mgrid[:N, :N]
R = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2) / 32


def wafer():
    return np.where(R <= 1.0, 1, 0).astype(np.uint8)


def test_center_blob():
    m = wafer()
    m[R < 0.2] = 2
    assert describe(m)["location"] == "중심"


def test_full_edge_ring():
    m = wafer()
    m[(R > 0.85) & (R <= 1.0)] = 2
    assert describe(m)["location"] == "링"


def test_edge_patch_is_edge_not_ring():
    m = wafer()
    m[(R > 0.85) & (R <= 1.0) & (xx > 50)] = 2
    assert describe(m)["location"] == "가장자리"


def test_scratch_line():
    m = wafer()
    for t in range(10, 50):
        m[t, t // 2 + 12] = 2
    assert describe(m)["location"] == "선"


def test_near_full():
    m = wafer()
    m[(R <= 1.0) & ((yy + xx) % 5 != 0)] = 2
    assert describe(m)["location"] == "전면"


def test_random_scatter():
    rng = np.random.default_rng(0)
    m = wafer()
    inside = np.argwhere(m == 1)
    pick = inside[rng.choice(len(inside), 60, replace=False)]
    m[pick[:, 0], pick[:, 1]] = 2
    assert describe(m)["location"] == "산발"


def test_no_fail():
    assert describe(wafer())["location"] == "없음"


def test_rotation_keeps_ring_and_center():
    m = wafer()
    m[R < 0.2] = 2
    assert describe(np.rot90(m))["location"] == "중심"
    m2 = wafer()
    m2[(R > 0.85) & (R <= 1.0)] = 2
    assert describe(np.fliplr(m2))["location"] == "링"
