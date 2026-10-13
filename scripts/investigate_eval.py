"""조사 에이전트 평가 — 판독 에이전트(v3 파이프라인)와 같은 웨이퍼로 비교한다.
 (a) reports/agent_eval.json 의 120장: 아는 패턴 96 + 학습에서 한 종류를 뺀 모델에 그 종류를 준 24 (같은 leave-one-out 체크포인트)
 (b) 라벨 없는 웨이퍼 중 깨진 맵(면적 < 0.5) 20장
 (c) 맵 품질 검사를 통과한 라벨 없는 웨이퍼 중 가장 낯선 20장 (정답 없음)
사용: python scripts/investigate_eval.py            (스모크: --limit 2)"""
import argparse
import json
import platform
import sys
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from wafer.agent import OLLAMA, read_card  # noqa: E402
from wafer.investigate import LOW_CONF, PROMPT_VERSION, Toolbox, needs_agent, investigate, load_data, ollama_llm  # noqa: E402
from wafer.prep import CLASSES  # noqa: E402
from wafer.reader import Reader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3.5:4b")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-broken", type=int, default=20)
ap.add_argument("--n-novel", type=int, default=20)
ap.add_argument("--broken-seed", type=int, default=None, help="깨진 맵 표본 시드(기본 seed+1)")
ap.add_argument("--limit", type=int, default=0, help="스모크 시험: 집합마다 이만큼만")
ap.add_argument("--out", default="investigate_eval")
ap.add_argument("--hybrid-only", action="store_true", help="LLM 을 다시 돌리지 않고 investigate_eval.json 과 agent_eval.json 으로 혼합 모드를 계산한다")
ap.add_argument("--from-partial", help="reports/<이름>.partial.json 의 원자료로 집계만 한다")
args = ap.parse_args()


def hybrid_eval():
    """혼합 모드: needs_agent 가 이유를 대면 저장된 에이전트 기록을, 아니면 파이프라인 기록을 판정으로 쓴다."""
    ev = json.loads((ROOT / "reports/investigate_eval.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "reports/agent_eval.json").read_text(encoding="utf-8"))["results"]
    old_sec = {r["case"]["row"]: r["seconds"] for r in old}
    new_pipe = {(k, p["row"]): p for k, v in ev["pipeline_v3_new"].items() for p in v}
    out = {"low_conf": LOW_CONF, "rule": "경고 켜짐 또는 quality_ok=false 또는 1위 확률 < low_conf 이면 에이전트, 아니면 파이프라인 판정(look 1위 패턴)"}
    for key, recs in ev["agent"].items():
        rows = []
        for r in recs:
            why = needs_agent(r["look"])
            if why:
                rows.append({"row": r["row"], "to_agent": True, "why": why, "route": r["route"], "pattern": r["card"].get("pattern"), "seconds": r["seconds"], "true": r["true"], "end": r["end"]})
            else:
                sec = old_sec[r["labeled_index"]] if "labeled_index" in r else new_pipe[(key, r["row"])]["seconds"]
                rows.append({"row": r["row"], "to_agent": False, "why": [], "route": "판정", "pattern": r["look"]["top3"][0]["pattern"], "seconds": sec, "true": r["true"], "end": "pipeline"})
        n = len(rows)
        s = {"n": n, "to_agent": sum(x["to_agent"] for x in rows), "share_to_agent": round(sum(x["to_agent"] for x in rows) / n, 4),
             "mean_seconds": round(float(np.mean([x["seconds"] for x in rows])), 3), "routes": dict(Counter(x["route"] for x in rows))}
        if key == "a_known":
            s["correct_verdict"] = sum(x["route"] == "판정" and x["pattern"] == x["true"] for x in rows)
            s["wrong_verdict"] = sum(x["route"] == "판정" and x["pattern"] != x["true"] for x in rows)
        elif key == "a_novel":
            s["not_verdict"] = sum(x["route"] != "판정" for x in rows)
        elif key == "b_broken":
            s["map_anomaly"] = sum(x["route"] == "맵 이상" for x in rows)
        out[key] = {"summary": s, "rows": rows}
    allrows = [x for k in ev["agent"] for x in out[k]["rows"]]
    out["overall"] = {"n": len(allrows), "to_agent": sum(x["to_agent"] for x in allrows), "share_to_agent": round(sum(x["to_agent"] for x in allrows) / len(allrows), 4),
                      "mean_seconds": round(float(np.mean([x["seconds"] for x in allrows])), 3)}
    (ROOT / "reports/investigate_hybrid_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["summary"] if isinstance(v, dict) and "summary" in v else v) for k, v in out.items() if k not in ("rule",)}, ensure_ascii=False, indent=2))


if args.hybrid_only:
    hybrid_eval()
    sys.exit()
device = "cuda"
llm = ollama_llm(args.model, seed=args.seed)

data = load_data()
X, y, split = data["X"], data["y"], data["split"]
area = np.load(ROOT / "runs/map_quality_stats.npz")["area"]
old = json.loads((ROOT / "reports/agent_eval.json").read_text(encoding="utf-8"))["results"]
old_known = [r for r in old if r["set"] == "known"]
old_unseen = {n: [r for r in old if r["set"] == f"unseen:{n}"] for n in CLASSES[1:]}
lim = (lambda lst: lst[:args.limit]) if args.limit else (lambda lst: lst)

results = {"a_known": [], "a_novel": [], "b_broken": [], "c_novel_unlabeled": []}
pipeline = {k: [] for k in results}
errors = []


def run_agent(box, row, set_name, extra=None):
    try:
        res = investigate(int(row), box.tools, llm, box.case_for_check(int(row)))
    except Exception as e:  # Ollama · GPU 오류는 숨기지 않고 기록한다
        errors.append({"set": set_name, "row": int(row), "error": f"{type(e).__name__}: {e}"})
        print(f"[오류] {set_name} row {row}: {e}", flush=True)
        if len(errors) >= 5:   # 연달아 실패하면 결과를 꾸미지 않고 멈춘다
            raise
        return
    case = box.case(int(row))
    res.update(set=set_name, true=case["true"], look=box.look(int(row)), **(extra or {}))
    results[set_name].append(res)
    dump_partial()
    c = res["card"]
    print(f'[{set_name}] row {row} 정답 {case["true"]:8s} -> {res["route"]} / {c.get("pattern")} · 호출 {res["n_tool_calls"]} · submit {res["n_submits"]} · {res["seconds"]}s', flush=True)


def dump_partial():
    """중간 결과를 남긴다(오래 걸리는 실행이 중간에 끊겨도 원자료가 남도록)."""
    (ROOT / "reports" / f"{args.out}.partial.json").write_text(
        json.dumps({"agent": results, "pipeline_v3_new": pipeline, "errors": errors}, ensure_ascii=False, default=str), encoding="utf-8")


def run_pipeline(reader, row, set_name):
    m = np.asarray(data["all_X"][int(row)])
    try:
        r = read_card(reader.case_from_map(m, row=int(row)), model=args.model)
    except Exception as e:
        errors.append({"set": "pipeline:" + set_name, "row": int(row), "error": f"{type(e).__name__}: {e}"})
        print(f"[오류] pipeline {set_name} row {row}: {e}", flush=True)
        return
    c = r["case"]
    pipeline[set_name].append({"row": int(row), "true": c["true"], "pattern": c["pattern"], "classifier_pred": c["classifier_pred"],
                               "unknown_flag": c["unknown_flag"], "pass_first": r["pass_first"], "pass_final": r["pass_final"],
                               "seconds": r["seconds"], "card": r["final"]})
    dump_partial()


# 사례 행은 agent_eval.json 에 저장된 것을 그대로 쓴다. (같은 생성기로 다시 뽑아 보니 그 행들이 나오지 않았다 —
# 저장된 실행과 지금 코드의 생성기 소비 순서가 다르다. 그래서 Reader 의 kNN 참조 표본은 원래 실행과 다를 수 있고,
# 경고 켜짐 여부가 저장된 값과 얼마나 같은지를 따로 잰다: summary.reader_flag_agreement)
def run_all():
    rng = np.random.default_rng(args.seed)
    main = Reader("lot_s0", None, X, y, split, device, rng)
    box = Toolbox(main, data, seed=args.seed)
    print("가까움 기준(lot_s0):", box.calibration, flush=True)
    calibration = {"lot_s0": box.calibration}

    # (a) 아는 패턴 96장
    for r in lim(old_known):
        i = r["case"]["row"]
        src = int(data["src_index"][i])
        assert data["all_y"][src] == y[i] == CLASSES.index(r["case"]["true"]) and (np.asarray(data["all_X"][src]) == X[i]).all()
        run_agent(box, src, "a_known", {"labeled_index": i})

    # (b) 깨진 맵 · (c) 낯선 라벨 없는 웨이퍼
    for r in lim(list(broken)):
        run_agent(box, r, "b_broken")
    for r in lim(novel):
        run_agent(box, r, "c_novel_unlabeled")
    for r in lim(list(broken)):
        run_pipeline(main, r, "b_broken")
    for r in lim(novel):
        run_pipeline(main, r, "c_novel_unlabeled")
    del box, main
    torch.cuda.empty_cache()

    # (a) 처음 보는 패턴 24장 — 한 종류를 뺀 모델
    for c in range(1, 9):
        name = CLASSES[c]
        rd = Reader(f"lot_s0_ex-{name}", c, X, y, split, device, rng)
        pick = [r["case"]["row"] for r in old_unseen[name]]
        bx = Toolbox(rd, data, seed=args.seed)
        calibration[f"lot_s0_ex-{name}"] = bx.calibration
        for i in lim(pick):
            run_agent(bx, int(data["src_index"][i]), "a_novel", {"labeled_index": i, "held_out": name})
        del bx, rd
        torch.cuda.empty_cache()
    return calibration


unl_mask = data["all_y"] < 0
broken_pool = np.flatnonzero(unl_mask & (area < 0.5))
broken = np.sort(np.random.default_rng(args.broken_seed if args.broken_seed is not None else args.seed + 1).choice(broken_pool, args.n_broken, replace=False))
unl, far = np.load(ROOT / "runs/unlabeled_novelty.npy")
order = np.argsort(-far)
cand = unl[order].astype(int)
novel = [int(r) for r in cand[area[cand] >= 0.5][:args.n_novel]]
if args.from_partial:   # 이미 돌린 원자료로 집계만 다시 한다
    part = json.loads((ROOT / "reports" / f"{args.from_partial}.partial.json").read_text(encoding="utf-8"))
    results, pipeline, errors = part["agent"], part["pipeline_v3_new"], part["errors"]
    calibration, t0 = {}, time.time()
else:
    t0 = time.time()
    calibration = run_all()
wall = time.time() - t0 if not args.from_partial else None


# ---------------------------------------------------------------- 집계
def mean(v):
    return round(float(np.mean(v)), 3) if len(v) else None


def rate(v):
    return round(float(np.mean(v)), 4) if len(v) else None


def agent_stats(rs):
    firsts = [r["submits"][0]["ok"] for r in rs if r["submits"]]
    return {"n": len(rs), "routes": dict(Counter(r["route"] for r in rs)), "end": dict(Counter(r["end"] for r in rs)),
            "mean_tool_calls": mean([r["n_tool_calls"] for r in rs]), "mean_seconds": mean([r["seconds"] for r in rs]),
            "mean_llm_seconds": mean([r["llm_seconds"] for r in rs]),
            "n_with_submit": len(firsts), "submit_first_pass": rate(firsts), "accepted": rate([r["accepted"] for r in rs]),
            "fabrication_rejections": int(sum(r["fabrication_rejections"] for r in rs)),
            "investigations_with_fabrication": int(sum(r["fabrication_rejections"] > 0 for r in rs)),
            "rejections_total": int(sum(not s["ok"] for r in rs for s in r["submits"])),
            "tool_usage": dict(Counter(s["tool"] for r in rs for s in r["trace"])),
            "called_lot_neighbors": int(sum(any(s["tool"] == "lot_neighbors" for s in r["trace"]) for r in rs))}


def is_correct_verdict(r):
    return r["route"] == "판정" and r["card"].get("pattern") == r["true"]


A, N, B, C = results["a_known"], results["a_novel"], results["b_broken"], results["c_novel_unlabeled"]
PB, PC = pipeline["b_broken"], pipeline["c_novel_unlabeled"]
old_by_row = {r["case"]["row"]: r["case"]["unknown_flag"] for r in old}
agree = [r["look"]["novelty_warning"] == old_by_row[r["labeled_index"]] for r in A + N]
summary = {"date": str(date.today()), "model": args.model, "seed": args.seed, "broken_seed": args.broken_seed if args.broken_seed is not None else args.seed + 1, "prompt_version": PROMPT_VERSION, "wall_seconds": round(wall) if wall else None, "n_errors": len(errors)}
summary["a_known"] = {"agent": {**agent_stats(A), "correct_verdict": rate([is_correct_verdict(r) for r in A]),
                                "verdict_but_wrong_pattern": int(sum(r["route"] == "판정" and r["card"].get("pattern") != r["true"] for r in A)),
                                "classifier_top1_correct": rate([r["look"]["top3"][0]["pattern"] == r["true"] for r in A])},
                      "pipeline_v3": {"n": len(old_known), "correct_verdict": rate([not r["case"]["unknown_flag"] and r["case"]["pattern"] == r["case"]["true"] for r in old_known]),
                                      "pass_first": rate([r["pass_first"] for r in old_known]), "pass_final": rate([r["pass_final"] for r in old_known]),
                                      "mean_seconds": mean([r["seconds"] for r in old_known])}}
old_n = [r for n in CLASSES[1:] for r in old_unseen[n]]
summary["a_novel"] = {"agent": {**agent_stats(N), "not_verdict": rate([r["route"] != "판정" for r in N]),
                                "warning_on": rate([r["look"]["novelty_warning"] for r in N]),
                                "by_class": {n: {"n": sum(r["held_out"] == n for r in N), "not_verdict": int(sum(r["held_out"] == n and r["route"] != "판정" for r in N))} for n in CLASSES[1:]}},
                      "pipeline_v3": {"n": len(old_n), "not_verdict(flag_on)": rate([r["case"]["unknown_flag"] for r in old_n]),
                                      "pass_first": rate([r["pass_first"] for r in old_n]), "pass_final": rate([r["pass_final"] for r in old_n]),
                                      "mean_seconds": mean([r["seconds"] for r in old_n])}}
summary["b_broken"] = {"agent": {**agent_stats(B), "map_anomaly": rate([r["route"] == "맵 이상" for r in B]),
                                 "verdict": int(sum(r["route"] == "판정" for r in B)), "look_quality_not_ok": int(sum(not r["look"]["quality_ok"] for r in B))},
                       "pipeline_v3": {"n": len(PB), "flag_on": rate([p["unknown_flag"] for p in PB]), "verdict(flag_off)": int(sum(not p["unknown_flag"] for p in PB)),
                                       "map_anomaly": 0.0, "pass_first": rate([p["pass_first"] for p in PB]), "pass_final": rate([p["pass_final"] for p in PB]),
                                       "mean_seconds": mean([p["seconds"] for p in PB])}}
summary["c_novel_unlabeled"] = {"ground_truth": "없음 (라벨 없는 웨이퍼 — 분포만 본다)",
                                "agent": {**agent_stats(C), "cited_lot_neighbors": int(sum(any(e.get("tool") == "lot_neighbors" for e in r["card"].get("evidence", []) if isinstance(e, dict)) for r in C)),
                                          "warning_on": int(sum(r["look"]["novelty_warning"] for r in C))},
                                "pipeline_v3": {"n": len(PC), "flag_on": int(sum(p["unknown_flag"] for p in PC)), "pass_first": rate([p["pass_first"] for p in PC]),
                                                "pass_final": rate([p["pass_final"] for p in PC]), "mean_seconds": mean([p["seconds"] for p in PC])}}
allr = A + N + B + C
summary["reader_flag_agreement"] = {"n": len(agree), "same_as_agent_eval_json": rate(agree)}
summary["overall_agent"] = agent_stats(allr)

try:
    ver = json.loads(urllib.request.urlopen(OLLAMA.replace("/api/chat", "/api/version"), timeout=10).read())["version"]
except Exception:
    ver = None
summary["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "ollama": ver,
                          "close_calibration": calibration, "budget": 8, "max_resubmit": 2}
summary["sets"] = {"a": "reports/agent_eval.json 의 120장", "b_rows": [int(r) for r in broken], "c_rows": novel}

out = ROOT / "reports" / f"{args.out}.json"
out.write_text(json.dumps({"summary": summary, "errors": errors, "agent": results, "pipeline_v3_new": pipeline}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
