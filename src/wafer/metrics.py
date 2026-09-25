"""지표 — 외부 라이브러리 없이 정의대로. 시험에서 scikit-learn 값과 대조한다."""
import numpy as np


def confusion(y, p, n):
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


def per_class_recall(y, p, n):
    m = confusion(y, p, n)
    return (np.diag(m) / np.maximum(m.sum(1), 1)).tolist()


def macro_f1(y, p, n):
    m = confusion(y, p, n)
    tp = np.diag(m).astype(float)
    prec = tp / np.maximum(m.sum(0), 1)
    rec = tp / np.maximum(m.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = m.sum(1) > 0
    return float(f1[present].mean())


def auroc(scores_pos, scores_neg):
    """양성 점수가 음성보다 클 확률 (동점은 절반). 순위합으로 계산."""
    s = np.concatenate([scores_pos, scores_neg])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_fpr(scores_pos, scores_neg, fpr=0.05):
    """음성의 (1-fpr) 분위수를 문턱으로 삼았을 때 양성 재현율."""
    thr = np.quantile(scores_neg, 1 - fpr)
    return float((scores_pos > thr).mean()), float(thr)
