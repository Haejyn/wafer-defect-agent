"""판독 검사기가 규칙 위반을 실제로 잡는지 — LLM 없이 카드를 손으로 만들어 넣는다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer.agent import UNKNOWN, build_prompt, check  # noqa: E402

CASE = {"pattern": "Edge-Ring", "confidence": 0.97, "unknown_flag": False, "ood_percentile": 40.0,
        "location": "링", "fail_ratio": 0.08, "similar": [{"pattern": "Edge-Ring", "similarity": 0.95}]}
GOOD = {"pattern": "Edge-Ring", "location": "링", "cause_ids": ["etch"],
        "check_order": ["식각 장비 가장자리 균일도 확인", "같은 로트 다른 웨이퍼 링 발생 여부 확인"], "summary": "식각 단계 이상이 원인 후보다."}


def test_good_card_passes():
    assert check(CASE, GOOD) == []


def test_cause_outside_table_is_caught():
    bad = dict(GOOD, cause_ids=["etch", "handling"])
    assert any("표 밖" in p for p in check(CASE, bad))


def test_location_mismatch_is_caught():
    bad = dict(GOOD, location="중심")
    assert any("location" in p for p in check(CASE, bad))


def test_pattern_mismatch_is_caught():
    bad = dict(GOOD, pattern="Center")
    assert any("pattern" in p for p in check(CASE, bad))


def test_foreign_cause_in_text_is_caught():
    bad = dict(GOOD, summary="박막 증착 문제일 수 있다.")
    assert any("허용되지 않은 원인" in p for p in check(CASE, bad))


def test_unknown_pattern_must_not_pick_causes():
    case = dict(CASE, pattern=UNKNOWN, unknown_flag=True)
    card = dict(GOOD, pattern=UNKNOWN, cause_ids=["etch"])
    assert check(case, card)


def test_prompt_lists_only_allowed_causes():
    p = build_prompt(CASE)
    assert "etch:" in p and "handling:" not in p and "deposition:" not in p
