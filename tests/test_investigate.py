"""조사 에이전트 — 대본 LLM 과 가짜 도구로 루프와 submit 검사를 확인한다. GPU · Ollama 가 필요 없다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer.agent import UNKNOWN, UNKNOWN_STEPS  # noqa: E402
from wafer.investigate import _same, flatten, investigate, validate_card  # noqa: E402

CASE = {"pattern": "Edge-Ring", "confidence": 0.97, "unknown_flag": False, "ood_percentile": 40.0, "location": "링", "fail_ratio": 0.08}
LOOK_OK = {"top3": [{"pattern": "Edge-Ring", "prob": 0.97}, {"pattern": "Loc", "prob": 0.02}, {"pattern": "none", "prob": 0.01}],
           "novelty_percentile": 40.0, "novelty_warning": False, "area": 0.78, "quality_ok": True}
LOOK_NOVEL = dict(LOOK_OK, top3=[{"pattern": "Scratch", "prob": 0.46}, {"pattern": "Random", "prob": 0.3}, {"pattern": "none", "prob": 0.1}],
                  novelty_percentile=99.8, novelty_warning=True)
LOOK_BROKEN = dict(LOOK_OK, area=0.27, quality_ok=False, novelty_warning=True, novelty_percentile=100.0)
LOT = {"lot": "lot15226", "die_size": 4899, "n_neighbors": 15, "n_compared": 15, "n_close": 12, "close_pred": {"Scratch": 9, "Random": 3}}
STEPS = UNKNOWN_STEPS[:2]


def fake_tools(look):
    return {"look": lambda row: look, "locate": lambda row: {"location": "링", "fail_ratio": 0.08},
            "similar": lambda row, pool="labeled", k=5: {"pool": pool, "hits": [{"row": 1, "label": "Scratch", "dist": 0.08}]},
            "lot_neighbors": lambda row: LOT,
            "causes": lambda pattern: {"pattern": pattern, "causes": [{"cause_id": "etch", "cause": "식각 단계 이상"}]}}


def tc(name, **args):
    return {"function": {"name": name, "arguments": args}}


def script(*steps):
    """steps: LLM 응답 한 번마다 tool_calls 목록. 대본이 끝난 뒤에 불리면 실패한다."""
    it = iter(steps)
    calls = []

    def llm(messages, tools):
        calls.append(messages)
        try:
            return {"role": "assistant", "content": "", "tool_calls": next(it)}
        except StopIteration:
            raise AssertionError("대본보다 많이 LLM 을 불렀다")
    llm.calls = calls
    return llm


def novel_card(**over):
    card = {"route": "새 패턴 후보", "pattern": UNKNOWN, "location": "링", "cause_ids": [], "check_order": STEPS,
            "summary": "처음 보는 패턴 후보다. 사람이 확인해야 한다.",
            "evidence": [{"tool": "lot_neighbors", "key": "n_close", "value": 12}]}
    return {**card, **over}


def run(llm, look=LOOK_NOVEL, **kw):
    return investigate(7, fake_tools(look), llm, CASE, **kw)


def test_budget_ends_with_human_check():
    llm = script(*[[tc("look", row=7)]] * 8)   # 9번째를 부르면 대본 부족으로 실패한다
    res = run(llm)
    assert res["route"] == "사람 확인" and res["card"]["reason"] == "조사 한도 초과"
    assert res["end"] == "budget" and res["n_tool_calls"] == 8 and len(llm.calls) == 8 and not res["accepted"]
    assert len(res["trace"]) == 8 and all("ms" in s and "result_summary" in s for s in res["trace"])


def test_budget_counts_submit_and_multiple_calls_in_one_reply():
    bad = {"route": "맵 이상"}
    res = run(script([tc("look", row=7), tc("locate", row=7), tc("similar", row=7, pool="labeled"), tc("lot_neighbors", row=7)],
                     [tc("submit", card=bad), tc("causes", pattern="Scratch"), tc("look", row=7), tc("look", row=7), tc("look", row=7)]))
    assert res["end"] == "budget" and res["n_tool_calls"] == 8


def test_evidence_mismatch_rejected_then_fixed():
    fake = novel_card(evidence=[{"tool": "lot_neighbors", "key": "n_close", "value": 99}])
    res = run(script([tc("look", row=7)], [tc("lot_neighbors", row=7)], [tc("submit", card=fake)], [tc("submit", card=novel_card())]))
    assert res["fabrication_rejections"] == 1 and res["n_submits"] == 2 and res["accepted"]
    assert "도구 결과와 다르다" in res["submits"][0]["problems"][0] and "12" in res["submits"][0]["problems"][0]
    assert res["route"] == "새 패턴 후보"


def test_evidence_from_tool_never_called_is_rejected():
    probs = validate_card(novel_card(), CASE, [("look", {}, LOOK_NOVEL)])
    assert any("lot_neighbors 결과가 없다" in p for p in probs)


def test_broken_map_routes_to_map_anomaly():
    card = {"route": "맵 이상", "pattern": UNKNOWN, "location": "링", "cause_ids": [], "check_order": [UNKNOWN_STEPS[2], UNKNOWN_STEPS[3]],
            "summary": "웨이퍼 면적이 작아 맵 이상 후보다.",
            "evidence": [{"tool": "look", "key": "quality_ok", "value": False}, {"tool": "look", "key": "area", "value": 0.27}]}
    res = run(script([tc("look", row=7)], [tc("submit", card=card)]), look=LOOK_BROKEN)
    assert res["route"] == "맵 이상" and res["accepted"] and res["n_tool_calls"] == 2 and res["fabrication_rejections"] == 0


def test_map_anomaly_rejected_when_quality_is_ok():
    card = {"route": "맵 이상", "pattern": UNKNOWN, "location": "링", "cause_ids": [], "check_order": STEPS, "summary": "맵 이상 후보다.", "evidence": []}
    assert any("quality_ok" in p for p in validate_card(card, CASE, [("look", {}, LOOK_NOVEL)]))
    assert any("quality_ok" in p for p in validate_card(card, CASE, []))   # look 을 안 부르고는 낼 수 없다


def test_new_pattern_without_lot_neighbors_rejected():
    card = novel_card(evidence=[{"tool": "look", "key": "novelty_warning", "value": True}])
    res = run(script([tc("look", row=7)], [tc("submit", card=card)], [tc("lot_neighbors", row=7)], [tc("submit", card=novel_card())]))
    assert any("lot_neighbors" in p for p in res["submits"][0]["problems"])
    assert res["n_submits"] == 2 and res["accepted"] and res["fabrication_rejections"] == 0


def test_new_pattern_needs_warning_on():
    probs = validate_card(novel_card(), CASE, [("look", {}, LOOK_OK), ("lot_neighbors", {}, LOT)])
    assert any("novelty_warning" in p for p in probs)


def test_resubmit_flow_allows_two_resubmits():
    bad = novel_card(cause_ids=["etch"])
    res = run(script([tc("look", row=7)], [tc("lot_neighbors", row=7)], [tc("submit", card=bad)], [tc("submit", card=bad)],
                     [tc("submit", card=novel_card())]))
    assert res["accepted"] and res["n_submits"] == 3 and [s["ok"] for s in res["submits"]] == [False, False, True]
    assert any("cause_ids" in p for p in res["submits"][0]["problems"])
    assert res["trace"][2]["result"]["resubmits_left"] == 2


def test_third_rejection_ends_with_human_check():
    bad = novel_card(cause_ids=["etch"])
    res = run(script([tc("look", row=7)], [tc("lot_neighbors", row=7)], [tc("submit", card=bad)], [tc("submit", card=bad)], [tc("submit", card=bad)]))
    assert res["end"] == "submit_failed" and res["route"] == "사람 확인" and not res["accepted"] and res["n_submits"] == 3


def test_known_pattern_verdict():
    card = {"route": "판정", "pattern": "Edge-Ring", "location": "링", "cause_ids": ["etch"],
            "check_order": ["식각 장비 가장자리 균일도 확인", "같은 로트 다른 웨이퍼 링 발생 여부 확인"],
            "summary": "식각 단계 이상이 원인 후보다.",
            "evidence": [{"tool": "look", "key": "top3.0.pattern", "value": "Edge-Ring"}, {"tool": "look", "key": "top3.0.prob", "value": "0.97"}]}
    res = run(script([tc("look", row=7)], [tc("submit", card=card)]), look=LOOK_OK)
    assert res["accepted"] and res["route"] == "판정"
    assert validate_card(dict(card, pattern="Loc"), CASE, [("look", {}, LOOK_OK)])   # look 1위와 다른 패턴
    assert any("novelty_warning" in p for p in validate_card(card, CASE, [("look", {}, dict(LOOK_OK, novelty_warning=True))]))
    assert any("look" in p for p in validate_card(card, CASE, []))


def test_no_cause_ids_outside_verdict():
    probs = validate_card(novel_card(cause_ids=["etch"]), CASE, [("look", {}, LOOK_NOVEL), ("lot_neighbors", {}, LOT)])
    assert any("cause_ids" in p for p in probs)


def test_wrong_row_is_corrected_and_bad_tool_is_an_error():
    seen = []
    tools = fake_tools(LOOK_OK)
    tools["look"] = lambda row: seen.append(row) or LOOK_OK
    res = investigate(7, tools, script([tc("look", row=999), tc("nonexistent")], [tc("look", row=7, color="red")]), CASE, max_calls=3)
    assert seen[0] == 7 and res["trace"][0]["row_corrected"] == 999
    assert "없는 도구" in res["trace"][1]["result"]["error"] and "error" in res["trace"][2]["result"]


def test_reply_without_tool_call_is_nudged_then_given_up():
    def llm(messages, tools):
        return {"role": "assistant", "content": "잘 모르겠다"}
    res = run(llm)
    assert res["end"] == "no_tool_call" and res["route"] == "사람 확인" and res["n_tool_calls"] == 0


def test_flatten_and_same():
    assert flatten({"a": [{"b": 1}], "c": {"d": True}}) == {"a.0.b": 1, "c.d": True}
    assert _same("0.97", 0.97) and _same(1, 1.0) and not _same(0.9, 0.97) and _same("true", True) and not _same(True, 1) and _same("Scratch", "Scratch")


def test_card_given_as_broken_json_string_is_rejected_with_clear_reason():
    import json
    good = json.dumps(novel_card(), ensure_ascii=False)
    res = run(script([tc("look", row=7)], [tc("lot_neighbors", row=7)], [tc("submit", card=good[:-3])], [tc("submit", card=good)]))
    assert "JSON" in res["submits"][0]["problems"][0] and res["accepted"] and res["n_submits"] == 2

