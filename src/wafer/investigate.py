"""조사 에이전트 — LLM 이 도구를 골라 부르며 어려운 웨이퍼(새 패턴 후보 · 깨진 맵 · 낮은 확신)를 조사한다.
도구는 전부 결정적 코드(Reader 등을 감싼 것)이고, LLM 은 어떤 도구를 부를지와 카드 내용을 정한다.
카드는 submit 에서 코드가 검사한다: 근거 값이 이 세션의 도구 결과와 같은지, route 별 조건을 만족하는지, 기존 check() 를 통과하는지.
판독 에이전트(agent.py)는 그대로 두고, 이 모듈은 따로 쓴다."""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from .agent import CAUSES, OLLAMA, UNKNOWN, UNKNOWN_STEPS, check
from .locate import describe, LOCATIONS
from .prep import CLASSES

ROOT = Path(__file__).resolve().parents[2]

ROUTES = ["판정", "새 패턴 후보", "맵 이상", "사람 확인"]
MAX_CALLS = 8          # submit 을 포함한 도구 호출 한도
MAX_RESUBMIT = 2       # submit 이 거절된 뒤 다시 낼 수 있는 횟수
MIN_AREA = 0.5         # 맵 품질 검사 규칙(reports/2026-09-26_stepF_quality_gate.md): 웨이퍼 면적 < 0.5 → 맵 이상
LOT_SPAN = 3           # lot_neighbors: 같은 로트 번호 ± 이만큼
CLOSE_PERCENTILE = 5   # lot_neighbors 의 '가깝다' 기준: 라벨이 다른 학습 웨이퍼 쌍의 코사인 거리 하위 5 %
LOT_CAP = 300          # lot_neighbors 가 한 번에 비교하는 이웃 웨이퍼 수 상한(같은 로트는 전부 포함)
UNL_POOL = 60000       # similar(pool="unlabeled") 가 찾는 라벨 없는 웨이퍼 표본 크기
LOW_CONF = 0.7         # 프롬프트 안내용: 분류 확률이 이 값 미만이면 사람 확인(코드가 강제하지 않는다)

PROMPT_VERSION = "v3"   # v1: 첫 시험 20장(reports/investigate_prompt_v1_partial.json) · v2: 예시와 재촉 문구 추가 · v3: 맵 이상 우선(최종, reports/investigate_eval.json)
SYSTEM_PROMPT = f"""웨이퍼 맵 조사 에이전트다. 도구로 사실을 모은 뒤 submit 으로 카드를 낸다. 도구 호출은 submit 포함 {MAX_CALLS}번까지다.
먼저 look 을 부른다. 경고가 켜졌으면 lot_neighbors 와 similar 로 확인하고, 판정이면 causes 로 원인 후보를 본다.

route 는 넷 중 하나다.
- 판정: 아는 패턴. look 의 1위 패턴과 같고 novelty_warning 이 false 여야 한다. 1위 확률이 {LOW_CONF} 미만이면 판정하지 말고 사람 확인으로 한다.
- 맵 이상: look 의 quality_ok 가 false. 깨진 맵은 새 패턴이 아니므로 새 패턴 후보보다 먼저 이 route 를 고른다.
- 새 패턴 후보: quality_ok 가 true 이고 novelty_warning 이 true. evidence 에 lot_neighbors 값을 하나 이상 넣어야 한다.
- 사람 확인: 그 밖에 사람이 봐야 하는 경우. reason 에 이유를 쓴다.

카드 규칙
- pattern: 판정이면 look 1위 이름. 그 밖에는 "{UNKNOWN}".
- location: locate 를 불러 나온 location 을 그대로 쓴다.
- cause_ids: 판정일 때만 causes 에서 나온 id. 그 밖에는 빈 배열.
- check_order: 2~4개. 판정이 아니면 아래 문장에서만 골라 그대로 쓴다.
{chr(10).join("  - " + s for s in UNKNOWN_STEPS)}
- summary: 한국어 2문장 이내. 확정하지 말고 '후보'라고 쓴다.
- evidence: [{{tool, key, value}}]. 세 칸을 모두 채운다. 도구가 돌려준 값을 그대로 옮기며, 지어내면 거절된다. key 는 값 하나를 가리키는 이름이고 중첩은 점으로 잇는다.
  예: {{"tool":"look","key":"novelty_warning","value":false}}, {{"tool":"look","key":"top3.0.prob","value":0.92}}, {{"tool":"lot_neighbors","key":"n_close","value":12}}, {{"tool":"similar","key":"hits.0.label","value":"Scratch"}}
설명을 글로 쓰지 말고 도구를 호출한다. 카드는 말이 아니라 submit 호출로 낸다. submit 이 거절되면 이유대로 고쳐 다시 낸다(최대 {MAX_RESUBMIT}번)."""

_STR = {"type": "string"}
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "look", "description": "분류 상위 3개 패턴 확률, 처음 보는 패턴 백분위와 경고, 맵 품질(면적·정상 여부)",
        "parameters": {"type": "object", "properties": {"row": {"type": "integer"}}, "required": ["row"]}}},
    {"type": "function", "function": {
        "name": "locate", "description": "불량 위치와 근거 수치(불량 비율, 덩어리 수 등)",
        "parameters": {"type": "object", "properties": {"row": {"type": "integer"}}, "required": ["row"]}}},
    {"type": "function", "function": {
        "name": "similar", "description": "임베딩이 가까운 웨이퍼 k개. pool 은 labeled(라벨 있는 학습 웨이퍼) 또는 unlabeled(라벨 없는 웨이퍼)",
        "parameters": {"type": "object", "properties": {"row": {"type": "integer"}, "pool": {"type": "string", "enum": ["labeled", "unlabeled"]},
                                                        "k": {"type": "integer"}}, "required": ["row", "pool"]}}},
    {"type": "function", "function": {
        "name": "lot_neighbors", "description": f"같은 로트와 번호가 ±{LOT_SPAN} 이내인 이웃 로트(같은 다이 크기)에서 이 웨이퍼와 가까운 웨이퍼 수와 그 예측 패턴",
        "parameters": {"type": "object", "properties": {"row": {"type": "integer"}}, "required": ["row"]}}},
    {"type": "function", "function": {
        "name": "causes", "description": "원인 표(causes.json)에서 패턴의 원인 후보",
        "parameters": {"type": "object", "properties": {"pattern": _STR}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "submit", "description": "판독 카드를 제출한다. 코드가 검사해 ok 또는 문제 목록을 돌려준다.",
        "parameters": {"type": "object", "properties": {"card": {"type": "object", "properties": {
            "route": {"type": "string", "enum": ROUTES},
            "pattern": _STR, "location": {"type": "string", "enum": LOCATIONS},
            "cause_ids": {"type": "array", "items": _STR}, "check_order": {"type": "array", "items": _STR},
            "summary": _STR, "reason": _STR,
            "evidence": {"type": "array", "items": {"type": "object", "properties": {
                "tool": _STR, "key": _STR, "value": {"type": ["string", "number", "boolean"]}}, "required": ["tool", "key", "value"]}},
        }, "required": ["route", "pattern", "location", "cause_ids", "check_order", "summary", "evidence"]}}, "required": ["card"]}}},
]
ROW_TOOLS = {"look", "locate", "similar", "lot_neighbors"}


# ---------------------------------------------------------------- 근거 검사

def flatten(obj, prefix: str = "") -> dict:
    """결과 JSON 을 점 경로 → 잎 값으로 편다. 리스트는 인덱스를 경로에 쓴다."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}.{i}" if prefix else str(i)))
    else:
        out[prefix] = obj
    return out


def _same(claimed, actual) -> bool:
    if isinstance(actual, bool):
        return str(claimed).strip().lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        if isinstance(claimed, bool):
            return False
        try:
            return abs(float(claimed) - float(actual)) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(claimed).strip() == str(actual).strip()


def verify_evidence(evidence, log: list) -> list[str]:
    """근거 값이 이 세션에서 그 도구가 실제로 돌려준 값과 같은지."""
    if not isinstance(evidence, list):
        return ["근거 형식: evidence 는 {tool, key, value} 목록이어야 한다"]
    by_tool: dict[str, list[dict]] = {}
    for tool, _args, res in log:
        if isinstance(res, dict) and "error" not in res:
            by_tool.setdefault(tool, []).append(flatten(res))
    probs = []
    for e in evidence:
        if not isinstance(e, dict) or not {"tool", "key", "value"} <= e.keys():
            probs.append(f"근거 형식: evidence 항목이 {{tool, key, value}} 가 아니다: {e}")
            continue
        tool, key, val = e["tool"], str(e["key"]), e["value"]
        if tool not in by_tool:
            probs.append(f"근거가 {tool} 인데 이 세션에서 {tool} 결과가 없다")
            continue
        cands = [f[key] for f in by_tool[tool] if key in f]
        if not cands:
            probs.append(f'근거 key 없음: {tool} 결과에 "{key}" 가 없다 (있는 key: {sorted(by_tool[tool][-1])[:8]})')
        elif not any(_same(val, c) for c in cands):
            probs.append(f"근거 값이 도구 결과와 다르다: {tool}.{key} = {val!r}, 도구가 돌려준 값 {cands[:3]}")
    return probs


def _is_fabrication(problem: str) -> bool:
    """근거가 도구 결과와 맞지 않아 걸린 경우(지어낸 값 · 부른 적 없는 도구 · 없는 key)."""
    return problem.startswith(("근거 값이", "근거 key", "근거가"))


def validate_card(card, case: dict, log: list) -> list[str]:
    """submit 의 코드 검사. 빈 목록이면 통과."""
    if not isinstance(card, dict):
        return ["card 는 객체여야 한다"]
    probs = []
    route = card.get("route")
    if route not in ROUTES:
        return [f"route 는 {ROUTES} 중 하나여야 한다: {route!r}"]
    ev = card.get("evidence")
    probs += verify_evidence(ev, log)
    ev_items = [e for e in ev if isinstance(e, dict)] if isinstance(ev, list) else []

    looks = [r for t, _a, r in log if t == "look" and isinstance(r, dict) and "error" not in r]
    look = looks[-1] if looks else None
    top1 = look["top3"][0]["pattern"] if look else None
    cause_ids, steps = card.get("cause_ids", []), card.get("check_order", [])
    if not isinstance(cause_ids, list) or not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        return probs + ["cause_ids 와 check_order 는 문자열 목록이어야 한다"]

    if route == "판정":
        if look is None:
            probs.append("판정은 look 을 먼저 불러야 한다")
        elif look["novelty_warning"]:
            probs.append("판정인데 look 의 novelty_warning 이 true 다")
    elif route == "맵 이상":
        if look is None or look["quality_ok"]:
            probs.append("맵 이상은 look 이 quality_ok=false 를 보여 준 경우만 낼 수 있다")
    elif route == "새 패턴 후보":
        if look is None or not look["novelty_warning"]:
            probs.append("새 패턴 후보는 look 의 novelty_warning 이 true 일 때만 낼 수 있다")
        if not any(e.get("tool") == "lot_neighbors" for e in ev_items):
            probs.append("새 패턴 후보는 evidence 에 lot_neighbors 값이 하나 이상 있어야 한다")
    if route != "판정" and cause_ids:
        probs.append("판정이 아니면 cause_ids 는 비어 있어야 한다")

    # 기존 check() — 판정은 아는 패턴 카드, 그 밖은 '처음 보는 패턴' 카드 규칙(원인 없음 · 고정 점검 문장)을 쓴다
    if route == "판정":
        cc = dict(case, pattern=top1 or case["pattern"], unknown_flag=False)
    else:
        expected = top1 if (route == "사람 확인" and top1 and card.get("pattern") == top1) else UNKNOWN
        cc = dict(case, pattern=expected, unknown_flag=True)
    core = {"pattern": card.get("pattern"), "location": card.get("location"), "cause_ids": cause_ids,
            "check_order": steps, "summary": card.get("summary") if isinstance(card.get("summary"), str) else ""}
    probs += check(cc, core)
    return probs


def needs_agent(look: dict, low_conf: float = LOW_CONF) -> list[str]:
    """혼합 모드의 배정 규칙. 이유가 하나라도 있으면 조사 에이전트로, 없으면 판독 파이프라인이 판정으로 처리한다.
    look: look() 의 반환. low_conf 는 프롬프트가 안내하던 값(0.7)과 같다 — 파이프라인에는 낮은 확신 규칙이 없다."""
    why = []
    if look["novelty_warning"]:
        why.append("처음 보는 패턴 경고")
    if not look["quality_ok"]:
        why.append("맵 품질 이상")
    if look["top3"][0]["prob"] < low_conf:
        why.append(f"1위 확률 {look['top3'][0]['prob']} < {low_conf}")
    return why


# ---------------------------------------------------------------- 조사 루프

def ollama_llm(model: str = "qwen3.5:4b", url: str = OLLAMA, timeout: int = 180, seed: int | None = 0):
    """llm(messages, tools) -> assistant message. 테스트에서는 이 자리에 대본 함수를 넣는다."""
    def call(messages: list, tools: list) -> dict:
        body = json.dumps({"model": model, "messages": messages, "tools": tools, "stream": False, "think": False,
                           "options": {"temperature": 0.2, "num_ctx": 8192, **({"seed": seed} if seed is not None else {})}}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["message"]
    return call


def _args(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _summary(res, n: int = 240) -> str:
    s = json.dumps(res, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


def _human_card(reason: str, case: dict, **extra) -> dict:
    return {"route": "사람 확인", "reason": reason, "pattern": UNKNOWN, "location": case.get("location", ""), "cause_ids": [],
            "check_order": [UNKNOWN_STEPS[1], UNKNOWN_STEPS[3]], "summary": f"{reason}. 사람이 확인해야 한다.", "evidence": [], **extra}


def investigate(row: int, tools: dict, llm, case: dict, max_calls: int = MAX_CALLS, max_resubmit: int = MAX_RESUBMIT,
                system_prompt: str = SYSTEM_PROMPT, max_nudge: int = 2) -> dict:
    """한 웨이퍼 조사. tools: 이름 → 함수, llm: llm(messages, tool_specs) → message, case: check() 에 쓸 판독 사례."""
    t_start = time.time()
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"조사할 웨이퍼: row {row}"}]
    log, trace, submits = [], [], []
    calls = nudges = 0
    llm_ms = 0.0
    end, card, accepted = None, None, False

    while end is None:
        t0 = time.time()
        msg = llm(msgs, TOOL_SPECS)
        llm_ms += (time.time() - t0) * 1000
        tcs = msg.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": msg.get("content", "") or "", **({"tool_calls": tcs} if tcs else {})})
        if not tcs:
            nudges += 1
            if nudges > max_nudge:
                end, card = "no_tool_call", _human_card("도구 호출 없이 끝남", case, last_reply=(msg.get("content") or "")[:300])
                break
            msgs.append({"role": "user", "content": "글로 답하지 말고 도구를 호출해라. 카드는 submit 도구를 호출해 낸다."})
            continue
        for tc in tcs:
            if calls >= max_calls:
                end, card = "budget", _human_card("조사 한도 초과", case)
                break
            fn = tc.get("function", {})
            name, args = fn.get("name", ""), _args(fn.get("arguments"))
            calls += 1
            t1 = time.time()
            step = {"step": calls, "tool": name, "args": dict(args)}
            if name == "submit":
                c = args.get("card", args if "route" in args else None)
                if isinstance(c, str):
                    parsed = _args(c)
                    c = parsed or None
                problems = ["card 가 올바른 JSON 객체가 아니다(문자열이 깨졌다). 객체로 다시 낸다"] if c is None else validate_card(c, case, log)
                res = {"ok": not problems} if not problems else {"ok": False, "problems": problems, "resubmits_left": max(0, max_resubmit - len(submits))}
                submits.append({"n": len(submits) + 1, "ok": not problems, "problems": problems,
                                "mismatch": [p for p in problems if _is_fabrication(p)]})
                if not problems:
                    end, card, accepted = "submitted", c, True
                elif len(submits) > max_resubmit:
                    end, card = "submit_failed", _human_card("제출 검사를 통과하지 못함", case, last_card=c, last_problems=problems)
            elif name in tools:
                if name in ROW_TOOLS and args.get("row") != row:
                    step["row_corrected"] = args.get("row")
                    args["row"] = row
                try:
                    res = tools[name](**args)
                except Exception as exc:  # 인자 오류 · 도구 오류도 한 번의 호출로 센다
                    res = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                res = {"error": f"없는 도구: {name}"}
            if name != "submit":
                log.append((name, args, res))
            step.update(result=res, result_summary=_summary(res), ms=round((time.time() - t1) * 1000, 1))
            trace.append(step)
            msgs.append({"role": "tool", "tool_name": name, "content": json.dumps(res, ensure_ascii=False, default=str)})
            if end:
                break
            if calls >= max_calls:
                end, card = "budget", _human_card("조사 한도 초과", case)
                break

    return {"row": row, "route": card["route"], "card": card, "accepted": accepted, "end": end,
            "n_tool_calls": calls, "n_submits": len(submits), "submits": submits,
            "fabrication_rejections": sum(1 for s in submits if s["mismatch"]),
            "trace": trace, "llm_seconds": round(llm_ms / 1000, 2), "seconds": round(time.time() - t_start, 2)}


# ---------------------------------------------------------------- 실제 도구

def load_data() -> dict:
    """조사에 필요한 배열: 라벨 캐시(labeled64) + 전체 81만 장(all64 · all_meta)."""
    lab = np.load(ROOT / "data/proc/labeled64.npz", allow_pickle=True)
    meta = np.load(ROOT / "data/proc/all_meta.npz", allow_pickle=True)
    return {"X": lab["X"], "y": lab["y"], "split": lab["split_lot"], "src_index": lab["src_index"], "lab_lot": lab["lot"],
            "all_X": np.load(ROOT / "data/proc/all64.npy", mmap_mode="r"), "all_y": meta["y"], "all_lot": meta["lot"],
            "die_size": meta["die_size"]}


def lot_number(s) -> int:
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else -1


class Toolbox:
    """Reader(분류 모델 하나) + 전체 데이터를 감싼 결정적 도구 모음. row 는 WM-811K 원본 행 번호."""

    def __init__(self, reader, data: dict, seed: int = 0):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.r, self.d, self.seed = reader, data, seed
        self.lot_num = np.array([lot_number(s) for s in data["all_lot"]])
        self._cache: dict[int, dict] = {}
        self._unl = None
        # lot_neighbors 의 '가깝다' 기준 — 라벨이 다른 학습 웨이퍼 쌍 20만 개의 코사인 거리 하위 5 % (모델마다 임베딩 공간이 달라 모델별로 잰다)
        rng = np.random.default_rng(seed)
        n = len(reader.db)
        a, b = rng.integers(0, n, 200000), rng.integers(0, n, 200000)
        lab = data["y"][reader.db]
        dist = (1 - (reader.db_emb[torch.from_numpy(a).to(reader.device)] * reader.db_emb[torch.from_numpy(b).to(reader.device)]).sum(1)).cpu().numpy()
        diff = (lab[a] != lab[b])
        same_def = (lab[a] == lab[b]) & (lab[a] > 0) & (a != b)
        self.close_t = float(np.percentile(dist[diff], CLOSE_PERCENTILE))
        self.calibration = {"percentile": CLOSE_PERCENTILE, "pairs": 200000, "threshold": round(self.close_t, 4),
                            "same_defect_pairs_below": round(float((dist[same_def] < self.close_t).mean()), 3)}
        self.tools = {"look": self.look, "locate": self.locate, "similar": self.similar,
                      "lot_neighbors": self.lot_neighbors, "causes": self.causes}

    # -- 내부
    def case_for_check(self, row: int) -> dict:
        """check() 에 넘기는 판독 사례(내부 배열을 뺀 것)."""
        return {k: v for k, v in self.case(row).items() if not k.startswith("_")}

    def _map(self, row: int) -> np.ndarray:
        return np.asarray(self.d["all_X"][int(row)])

    def case(self, row: int) -> dict:
        if row not in self._cache:
            m = self._map(row)
            yv = int(self.d["all_y"][row])
            c = self.r.case_from_map(m, row=int(row), true=CLASSES[yv] if yv >= 0 else "라벨 없음")
            probs, emb = self._forward(m[None])
            c["_probs"], c["_emb"], c["_area"] = probs[0], emb[0], float((m > 0).mean())
            self._cache[row] = c
        return self._cache[row]

    def _forward(self, Xs: np.ndarray):
        from .train import predict
        logits, emb = predict(self.r.model, np.ascontiguousarray(Xs), self.r.device)
        probs = self.torch.softmax(self.torch.from_numpy(logits), 1).numpy()
        return probs, emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    def _embed_rows(self, rows: np.ndarray, batch: int = 2048):
        out = []
        for i in range(0, len(rows), batch):
            _, e = self._forward(np.asarray(self.d["all_X"][rows[i:i + batch]]))
            out.append(self.torch.from_numpy(e))
        return self.torch.cat(out).to(self.r.device)

    def _unlabeled_pool(self):
        if self._unl is None:
            unl = np.flatnonzero(self.d["all_y"] < 0)
            rows = np.sort(np.random.default_rng(self.seed).choice(unl, min(UNL_POOL, len(unl)), replace=False))
            self._unl = (rows, self._embed_rows(rows))
        return self._unl

    # -- 도구
    def look(self, row: int) -> dict:
        c = self.case(row)
        p = c["_probs"]
        top = np.argsort(-p)[:3]
        area = c["_area"]
        return {"top3": [{"pattern": CLASSES[self.r.classes[int(i)]], "prob": round(float(p[i]), 3)} for i in top],
                "novelty_percentile": round(c["ood_percentile"], 1), "novelty_warning": bool(c["unknown_flag"]),
                "area": round(area, 3), "quality_ok": bool(area >= MIN_AREA)}

    def locate(self, row: int) -> dict:
        return describe(self._map(row))

    def similar(self, row: int, pool: str = "labeled", k: int = 5) -> dict:
        k = max(1, min(int(k), 10))
        q = self.torch.from_numpy(self.case(row)["_emb"]).to(self.r.device)
        if pool == "labeled":
            sims = self.r.db_emb @ q
            top = sims.topk(k + 1).indices.cpu().numpy()
            hits = []
            for j in top:
                li = int(self.r.db[j])
                src = int(self.d["src_index"][li])
                if src != row and len(hits) < k:
                    hits.append({"row": src, "label": CLASSES[self.d["y"][li]], "lot": str(self.d["lab_lot"][li]), "dist": round(float(1 - sims[j]), 3)})
        elif pool == "unlabeled":
            rows, emb = self._unlabeled_pool()
            sims = emb @ q
            top = sims.topk(k + 1).indices.cpu().numpy()
            hits = [{"row": int(rows[j]), "label": "라벨 없음", "lot": str(self.d["all_lot"][rows[j]]), "dist": round(float(1 - sims[j]), 3)}
                    for j in top if rows[j] != row][:k]
        else:
            return {"error": 'pool 은 "labeled" 또는 "unlabeled"'}
        return {"pool": pool, "k": k, "hits": hits}

    def lot_neighbors(self, row: int) -> dict:
        d = self.d
        n, ds = self.lot_num[row], d["die_size"][row]
        base = {"lot": str(d["all_lot"][row]), "die_size": None if np.isnan(ds) else int(ds), "lot_span": LOT_SPAN}
        if n < 0 or np.isnan(ds):
            return {**base, "n_neighbors": 0, "n_compared": 0, "n_close": 0}
        mask = (np.abs(self.lot_num - n) <= LOT_SPAN) & (self.lot_num >= 0) & (d["die_size"] == ds)
        mask[row] = False
        idx = np.flatnonzero(mask)
        same = idx[self.lot_num[idx] == n]
        adj = idx[self.lot_num[idx] != n]
        rng = np.random.default_rng(self.seed + int(row))
        if len(same) + len(adj) > LOT_CAP:
            adj = np.sort(rng.choice(adj, max(0, LOT_CAP - len(same)), replace=False))
        cmp_rows = np.concatenate([same, adj]).astype(int)
        res = {**base, "n_neighbors": int(len(idx)), "n_compared": int(len(cmp_rows)), "close_threshold": round(self.close_t, 3)}
        if len(cmp_rows) == 0:
            return {**res, "n_close": 0, "n_close_same_lot": 0, "n_close_adjacent_lots": 0, "close_pred": {}, "close_novelty_flagged": 0, "closest": []}
        probs, emb = self._forward(np.asarray(d["all_X"][cmp_rows]))
        dist = 1 - emb @ self.case(row)["_emb"]
        close = np.flatnonzero(dist < self.close_t)
        preds = [CLASSES[self.r.classes[int(i)]] for i in probs[close].argmax(1)] if len(close) else []
        flagged = int((self.r.score(np.asarray(d["all_X"][cmp_rows[close]]))[0] > self.r.threshold).sum()) if len(close) else 0
        order = close[np.argsort(dist[close])][:3]
        n_same = len(same)
        return {**res, "n_close": int(len(close)), "n_close_same_lot": int((close < n_same).sum()), "n_close_adjacent_lots": int((close >= n_same).sum()),
                "close_pred": {p: preds.count(p) for p in sorted(set(preds))}, "close_novelty_flagged": flagged,
                "closest": [{"row": int(cmp_rows[i]), "lot": str(d["all_lot"][cmp_rows[i]]), "dist": round(float(dist[i]), 3),
                             "pred": CLASSES[self.r.classes[int(probs[i].argmax())]]} for i in order]}

    def causes(self, pattern: str) -> dict:
        lst = [] if pattern == UNKNOWN else CAUSES["patterns"].get(pattern, [])
        res = {"pattern": pattern, "causes": [{"cause_id": c["cause_id"], "cause": c["cause"], "source": c["source"]} for c in lst]}
        if not lst:
            res["note"] = "원인 표에 없음"
        return res
