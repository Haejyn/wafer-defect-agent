"""판독 에이전트 — 분류 결과 · 처음 보는 패턴 경고 · 코드가 계산한 위치 · 유사 사례를 받아
로컬 LLM(Ollama)이 원인 후보와 점검 순서를 쓰고, 코드가 그 답을 검사한다. 검사에 걸리면 이유를 알려 주고 다시 묻는다."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .locate import LOCATIONS

CAUSES = json.loads((Path(__file__).with_name("causes.json")).read_text(encoding="utf-8"))
UNKNOWN = "처음 보는 패턴"
OLLAMA = "http://127.0.0.1:11434/api/chat"

SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "location": {"type": "string", "enum": LOCATIONS},
        "cause_ids": {"type": "array", "items": {"type": "string"}},
        "check_order": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["pattern", "location", "cause_ids", "check_order", "summary"],
}


def allowed_causes(pattern: str) -> list[dict]:
    return [] if pattern == UNKNOWN else CAUSES["patterns"].get(pattern, [])


def build_prompt(case: dict) -> str:
    pattern = case["pattern"]
    cands = allowed_causes(pattern)
    cand_text = "\n".join(f'- {c["cause_id"]}: {c["cause"]} (근거: "{c["quote"]}" [{c["source"]}])' for c in cands) or "- (없음 — 원인 표에 없는 패턴이다)"
    sim = ", ".join(f'{s["pattern"]}({s["similarity"]:.2f})' for s in case["similar"])
    return f"""당신은 반도체 수율 엔지니어를 돕는 웨이퍼 맵 판독 보조다. 아래 사실만 써서 판독 카드를 JSON 으로 쓴다.

[사실 — 코드가 계산한 값, 바꾸지 말 것]
- 판정 패턴: {pattern}  (분류기 확신도 {case["confidence"]:.2f})
- 처음 보는 패턴 경고: {"예" if case["unknown_flag"] else "아니오"} (이상 점수 백분위 {case["ood_percentile"]:.1f})
- 불량 위치(계산값): {case["location"]}
- 불량 다이 비율: {case["fail_ratio"]:.3f}
- 유사한 과거 웨이퍼 패턴(유사도): {sim}

[원인 후보 — 이 목록의 cause_id 만 쓸 수 있다]
{cand_text}

[규칙]
1. pattern 은 판정 패턴을 그대로 쓴다.
2. location 은 계산값 "{case["location"]}" 을 그대로 쓴다.
3. cause_ids 는 위 목록의 id 만. 목록이 없으면 빈 배열.
4. check_order 는 엔지니어가 확인할 순서 2~4개, 한 줄씩 한국어. 원인 후보와 위치에서만 이끌어 낸다. 목록에 없는 공정 이름을 새로 만들지 않는다.
5. summary 는 한국어 2문장 이내. 확정하지 말고 '후보'라고 쓴다. 처음 보는 패턴이면 원인 표에 없으니 사람이 확인해야 한다고 쓴다."""


def check(case: dict, card: dict) -> list[str]:
    """판독 카드의 규칙 위반 목록. 빈 목록이면 통과."""
    problems = []
    if card.get("pattern") != case["pattern"]:
        problems.append(f'pattern 이 판정("{case["pattern"]}")과 다르다: "{card.get("pattern")}"')
    if card.get("location") != case["location"]:
        problems.append(f'location 이 계산값("{case["location"]}")과 다르다: "{card.get("location")}"')
    ok_ids = {c["cause_id"] for c in allowed_causes(case["pattern"])}
    bad = [c for c in card.get("cause_ids", []) if c not in ok_ids]
    if bad:
        problems.append(f"원인 표 밖의 cause_id: {bad} (허용: {sorted(ok_ids) or '없음'})")
    # 요약·점검 순서에 다른 패턴의 원인 이름이 끼어들었는지
    text = card.get("summary", "") + " " + " ".join(card.get("check_order", []))
    foreign = set()
    for pat, lst in CAUSES["patterns"].items():
        for c in lst:
            if c["cause_id"] not in ok_ids:
                core = c["cause"].split("(")[0].replace(" 단계 이상", "").replace(" 이상", "").strip()
                if core and core in text:
                    foreign.add(core)
    if foreign:
        problems.append(f"허용되지 않은 원인이 문장에 나온다: {sorted(foreign)}")
    if case["unknown_flag"] and card.get("cause_ids"):
        problems.append("처음 보는 패턴인데 원인을 골랐다")
    if not (2 <= len(card.get("check_order", [])) <= 4):
        problems.append("check_order 는 2~4개")
    return problems


def ask(prompt: str, model: str, messages: list | None = None, timeout=180) -> tuple[dict, str]:
    msgs = (messages or []) + [{"role": "user", "content": prompt}]
    body = json.dumps({"model": model, "messages": msgs, "stream": False, "format": SCHEMA, "think": False,
                       "options": {"temperature": 0.2, "num_ctx": 4096}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = json.loads(r.read())["message"]["content"]
    return json.loads(raw), raw


def read_card(case: dict, model: str = "qwen3.5:4b", retries: int = 1) -> dict:
    """한 웨이퍼의 판독. 첫 답과 재질문 뒤 답의 검사 결과를 모두 남긴다."""
    prompt = build_prompt(case)
    t0 = time.time()
    card, raw = ask(prompt, model)
    attempts = [{"card": card, "problems": check(case, card)}]
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}]
    while attempts[-1]["problems"] and len(attempts) <= retries:
        fix = "검사에서 다음이 걸렸다. 규칙을 지켜 JSON 을 다시 써라:\n- " + "\n- ".join(attempts[-1]["problems"])
        card, raw = ask(fix, model, messages)
        messages += [{"role": "user", "content": fix}, {"role": "assistant", "content": raw}]
        attempts.append({"card": card, "problems": check(case, card)})
    return {"case": case, "attempts": attempts, "final": attempts[-1]["card"],
            "pass_first": not attempts[0]["problems"], "pass_final": not attempts[-1]["problems"],
            "seconds": round(time.time() - t0, 1), "model": model}
