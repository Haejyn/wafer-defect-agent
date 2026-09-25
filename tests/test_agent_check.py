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


def test_word_inside_allowed_cause_is_not_foreign():
    """Donut 의 허용 원인 '식각 균일도 저하' 속 '식각'은 Edge-Ring 원인('식각 단계 이상')으로 치지 않는다 (09-25 스모크에서 잘못 걸림)."""
    case = dict(CASE, pattern="Donut", location="도넛")
    card = {"pattern": "Donut", "location": "도넛", "cause_ids": ["etch_uniformity"],
            "check_order": ["식각 균일도 확인", "도넛 반경 확인"], "summary": "식각 균일도 저하가 원인 후보다."}
    assert check(case, card) == []


def test_claiming_unknown_without_flag_is_caught():
    """경고가 꺼져 있는데 '처음 보는 패턴'이라고 쓰면 판정과 모순 (09-25 스모크에서 놓침)."""
    bad = dict(GOOD, summary="식각 단계 이상이 후보다. 처음 보는 패턴이므로 원인 표에 없어 확인해야 한다.")
    assert any("경고가 없는데" in p for p in check(CASE, bad))


def test_negated_unknown_is_not_a_claim():
    ok = dict(GOOD, summary="식각 단계 이상이 후보다. 처음 보는 패턴이 아니므로 표의 원인을 먼저 본다.")
    assert check(CASE, ok) == []


def test_unknown_rule_only_in_prompt_when_flagged():
    assert "처음 보는 패턴 경고가 켜졌다" not in build_prompt(CASE)
    flagged = dict(CASE, pattern=UNKNOWN, unknown_flag=True)
    assert "처음 보는 패턴 경고가 켜졌다" in build_prompt(flagged)


def test_prompt_lists_only_allowed_causes():
    p = build_prompt(CASE)
    assert "etch:" in p and "handling:" not in p and "deposition:" not in p
