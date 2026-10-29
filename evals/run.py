"""Run the evaluation set against the real agent (LLM gateway, Pinecone, MCP tools) and score it.

    python -m evals.run                          # every case
    python -m evals.run --category emergency     # one category
    python -m evals.run --case kb-syphilis       # one case, repeatable
    python -m evals.run --min-pass 0.9 --strict-emergencies   # regression gate
    python -m evals.run --rescore evals/results/<run>.json   # score a saved run again, no LLM

Each case runs in a fresh conversation, so memory from one question never helps the next.
Results go to evals/results/: a JSON file per run (not versioned) and latest.md, the report of
the last full run. With LANGSMITH_TRACING=true in .env, runs are traced in LangSmith and tagged
with the run id.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The eval must not write into the app's own conversations, and every case must reach a model.
os.environ["DATABASE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="medguide-eval-"), "eval.db")
os.environ.setdefault("LLM_CACHE_TTL_SECONDS", "0")

from app.core.config import Settings  # noqa: E402
from app.main import build_production_services  # noqa: E402
from evals.evaluators import score, summarize  # noqa: E402

DATASET = Path(__file__).with_name("dataset.json")
RESULTS = Path(__file__).with_name("results")
LABELS = {
    "emergency_recall": "Emergencies detected (recall)",
    "false_alarm_rate": "False emergency alarms",
    "urgency_ok": "Urgency level",
    "source_ok": "Path taken (textbook, research, emergency, refusal)",
    "evidence_ok": "Expected kind of source cited",
    "language_ok": "Answer language",
    "scope_ok": "Out-of-scope refusal",
    "fact_recall": "Expected facts in the answer (mean)",
    "guardrail_ok": "No dosing or diagnosis",
    "passed": "Cases passing every check",
}


def tracing(run_id: str, case: dict):
    """LangSmith tags for this case, or a no-op when langsmith is not installed."""
    try:
        from langsmith import tracing_context
    except ImportError:
        from contextlib import nullcontext

        return nullcontext()
    metadata = {"eval_run": run_id, "eval_case": case["id"], "category": case["category"]}
    return tracing_context(tags=["eval", run_id], metadata=metadata)


def run_case(chat, case: dict, run_id: str) -> dict:
    started = time.perf_counter()
    try:
        with tracing(run_id, case):
            result = chat.ask(case["question"])
        error = None
    except Exception as exc:  # a provider outage or quota is recorded, not hidden
        result, error = {}, f"{type(exc).__name__}: {exc}"
    trace = result.get("trace", [])
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "seconds": round(time.perf_counter() - started, 1),
        "error": error,
        "urgency": result.get("urgency"),
        "expected_urgency": case["expect"].get("urgency"),
        "source_used": result.get("source_used"),
        "language": result.get("language"),
        "in_scope": result.get("in_scope"),
        "grounding": result.get("grounding"),
        "usefulness": result.get("usefulness"),
        "research_calls": result.get("research_calls", []),
        "answer": result.get("answer", ""),
        "llm_steps": sum(1 for step in trace if step.get("model")),
        "models": sorted({step["model"] for step in trace if step.get("model")}),
        "fallbacks": sum("(fallback)" in step.get("detail", "") for step in trace),
        "trace": trace,
        "score": score(case, result),
    }


def saved_result(row: dict) -> dict:
    """The fields the evaluators read, rebuilt from a saved row."""
    in_scope = row.get("in_scope")
    if in_scope is None:  # runs saved before in_scope was stored: the triage trace states it
        triage = next((s["detail"] for s in row["trace"] if s["node"] == "triage"), "")
        in_scope = "out of scope" not in triage
    return {
        "answer": row["answer"],
        "urgency": row["urgency"],
        "source_used": row["source_used"],
        "language": row["language"],
        "in_scope": in_scope,
        "citations": [{"type": kind} for kind in row["score"]["evidence"]],
    }


def rescore(path: Path, dataset: dict) -> dict:
    """Score a saved run against the current dataset. Only for evaluator or expectation fixes:
    the agent's answers stay the recorded ones."""
    run = json.loads(path.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in dataset["cases"]}
    rows = [r for r in run["rows"] if r["id"] in cases]
    for row in rows:
        case = cases[row["id"]]
        row["expected_urgency"] = case["expect"].get("urgency")
        row["score"] = score(case, saved_result(row))
    run.update(rows=rows, summary=summarize(rows), rescored_with_dataset=dataset["version"])
    return run


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def markdown(run: dict) -> str:
    summary, rows = run["summary"], run["rows"]
    lines = [
        f"# Evaluation run {run['run_id']}",
        "",
        f"{run['started']} · {len(rows)} cases · dataset v{run['dataset_version']}"
        + (
            f", rescored with v{run['rescored_with_dataset']}"
            if "rescored_with_dataset" in run
            else ""
        )
        + " · "
        f"models: {', '.join(run['models']) or 'n/a'}",
        "",
        "| Check | Score | Cases |",
        "|---|---|---|",
    ]
    for key, label in LABELS.items():
        value, count = summary[key]
        lines.append(f"| {label} | {percent(value)} | {count} |")

    seconds = sorted(r["seconds"] for r in rows)
    grounded = [r["grounding"] for r in rows if r["grounding"] not in (None, "skipped")]
    lines += [
        "",
        f"Latency per question: median {seconds[len(seconds) // 2]} s, max {seconds[-1]} s. "
        f"LLM calls per question: {sum(r['llm_steps'] for r in rows) / len(rows):.1f} on average. "
        f"Grounding verdicts on generated answers: "
        f"{sum(g == 'supported' for g in grounded)} supported, "
        f"{sum(g == 'partial' for g in grounded)} partial, "
        f"{sum(g == 'unsupported' for g in grounded)} unsupported. "
        f"Malformed structured outputs recovered: {sum(r['fallbacks'] for r in rows)}.",
        "",
        "| Case | Category | Urgency | Path | Sources cited | Facts | Result | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["score"]
        notes = [r["error"]] if r["error"] else []
        notes += [f"missing: {', '.join(s['missing_facts'])}"] if s["missing_facts"] else []
        notes += [f"forbidden: {', '.join(s['violations'])}"] if s["violations"] else []
        notes += ["wrong urgency"] if s["urgency_ok"] is False else []
        notes += ["wrong path"] if s["source_ok"] is False else []
        notes += ["expected source not cited"] if s["evidence_ok"] is False else []
        notes += ["wrong language"] if s["language_ok"] is False else []
        facts = "" if s["fact_recall"] is None else percent(s["fact_recall"])
        lines.append(
            f"| {r['id']} | {r['category']} | {r['urgency'] or ''} | "
            f"{r['source_used'] or 'error'} | "
            f"{', '.join(s['evidence'])} | {facts} | {'pass' if s['passed'] else 'FAIL'} | "
            f"{'; '.join(notes)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--category", help="run one category only")
    parser.add_argument("--case", action="append", help="case id, repeatable")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between cases")
    parser.add_argument("--min-pass", type=float, help="exit 1 below this pass rate")
    parser.add_argument(
        "--strict-emergencies", action="store_true", help="exit 1 if any emergency is missed"
    )
    parser.add_argument("--rescore", type=Path, help="score a saved run JSON again, no model call")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.rescore:
        run = rescore(args.rescore, dataset)
        return save_and_report(run, full=len(run["rows"]) == len(dataset["cases"]), args=args)
    cases = [
        c
        for c in dataset["cases"]
        if (not args.category or c["category"] == args.category)
        and (not args.case or c["id"] in args.case)
    ]
    if not cases:
        print("No case matches.")
        return 1

    chat, _media, cleanup = build_production_services(Settings())
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    rows = []
    try:
        for i, case in enumerate(cases, 1):
            row = run_case(chat, case, run_id)
            rows.append(row)
            verdict = "pass" if row["score"]["passed"] else "FAIL"
            print(
                f"[{i}/{len(cases)}] {verdict:4} {case['id']:28} {row['urgency'] or '':9} "
                f"{row['source_used'] or 'error':12} {row['seconds']:>5} s"
                + (f"  {row['error'][:90]}" if row["error"] else ""),
                flush=True,
            )
            if i < len(cases):
                time.sleep(args.delay)
    finally:
        cleanup()

    run = {
        "run_id": run_id,
        "started": run_id[:15],
        "dataset_version": dataset["version"],
        "models": sorted({m for r in rows for m in r["models"]}),
        "summary": summarize(rows),
        "rows": rows,
    }
    return save_and_report(run, full=not args.category and not args.case, args=args)


def save_and_report(run: dict, full: bool, args: argparse.Namespace) -> int:
    """Save the run, print its report (also latest.md for a full run), apply the gate."""
    RESULTS.mkdir(exist_ok=True)
    rescored = run.get("rescored_with_dataset")
    name = f"{run['run_id']}-rescored-v{rescored}" if rescored else run["run_id"]
    (RESULTS / f"{name}.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = markdown(run)
    if full:
        (RESULTS / "latest.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    failed = False
    pass_rate = run["summary"]["passed"][0] or 0
    if args.min_pass is not None and pass_rate < args.min_pass:
        print(f"Pass rate {pass_rate:.0%} is below the required {args.min_pass:.0%}.")
        failed = True
    # A missed emergency is never averaged away by other cases passing.
    recall = run["summary"]["emergency_recall"][0]
    if args.strict_emergencies and recall is not None and recall < 1:
        print(f"Emergency recall is {recall:.0%}: every emergency must be detected.")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
