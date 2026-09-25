import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import f1_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer.metrics import auroc, macro_f1, per_class_recall, recall_at_fpr  # noqa: E402


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000), st.integers(2, 9), st.integers(5, 300))
def test_macro_f1_matches_sklearn(seed, n, size):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n, size)
    p = rng.integers(0, n, size)
    labels = np.unique(y)  # 우리 정의: 시험에 있는 클래스만 평균
    ref = f1_score(y, p, labels=labels, average="macro", zero_division=0)
    assert abs(macro_f1(y, p, n) - ref) < 1e-9


@settings(max_examples=100, deadline=None)
@given(st.integers(0, 10_000))
def test_recall_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, 200)
    p = rng.integers(0, 5, 200)
    ref = recall_score(y, p, labels=range(5), average=None, zero_division=0)
    assert np.allclose(per_class_recall(y, p, 5), ref)


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 10_000))
def test_auroc_matches_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, 6, rng.integers(1, 50)).astype(float)  # 동점이 많다
    neg = rng.integers(0, 6, rng.integers(1, 50)).astype(float)
    ref = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
    assert abs(auroc(pos, neg) - ref) < 1e-9


def test_recall_at_fpr_threshold_keeps_false_alarms_at_or_below_target():
    rng = np.random.default_rng(0)
    neg = rng.normal(0, 1, 10_000)
    pos = rng.normal(2, 1, 1_000)
    r, thr = recall_at_fpr(pos, neg, 0.05)
    assert (neg > thr).mean() <= 0.0501
    assert 0.5 < r < 0.9
