"""The evaluation's own checks: a wrong evaluator would make every score meaningless."""

import json
from pathlib import Path

from evals.evaluators import CHECKS, found, score, summarize, unnegated

DATASET = json.loads(
    (Path(__file__).resolve().parents[1] / "evals" / "dataset.json").read_text("utf-8")
)
CATEGORIES = {
    "emergency",
    "no-false-alarm",
    "textbook",
    "everyday",
    "drug",
    "safety",
    "research",
    "scope",
}


def test_facts_match_without_case_accents_or_markdown():
    assert found("Il faut **bien s'hydrater** et consulter un Médecin.", ["medecin"])
    assert found("Caused by *Treponema pallidum*.", ["treponema pallidum"])
    assert not found("Drink fluids.", ["chancre"])


def test_negated_forbidden_phrases_are_not_violations():
    assert not unnegated(
        "Never mix bleach and ammonia: it releases toxic gas.", "bleach and ammonia"
    )
    assert not unnegated("Je ne peux pas dire que vous avez la syphilis.", "vous avez la syphilis")
    assert unnegated("From what you describe, vous avez la syphilis.", "vous avez la syphilis")


def test_emergency_case_checks_urgency_path_language_and_numbers():
    case = {
        "expect": {
            "urgency": ["emergency"],
            "sources": ["emergency"],
            "language": ["fr"],
            "facts": [["15"], ["112"]],
        }
    }
    result = {
        "urgency": "emergency",
        "source_used": "emergency",
        "language": "fr",
        "answer": "Appelez immédiatement le numéro d'urgence (15 ou 112 en France).",
    }
    assert score(case, result)["passed"] is True
    assert score(case, {**result, "urgency": "urgent"})["passed"] is False


def test_evidence_check_looks_at_the_kind_of_source_cited():
    case = {"expect": {"evidence": ["medlineplus"]}}
    assert (
        score(case, {"citations": [{"type": "medlineplus"}, {"type": "kb"}]})["evidence_ok"] is True
    )
    assert score(case, {"citations": [{"type": "web"}]})["evidence_ok"] is False
    assert score(case, {})["evidence_ok"] is False


def test_checks_not_asked_for_are_ignored():
    result = score({"expect": {"sources": ["out_of_scope"]}}, {"source_used": "out_of_scope"})
    assert result["urgency_ok"] is None and result["fact_recall"] is None
    assert result["passed"] is True


def test_emergency_recall_and_false_alarms_are_counted_separately():
    def row(expected, actual):
        return {
            "expected_urgency": expected,
            "urgency": actual,
            "score": {"passed": True, "fact_recall": None},
        }

    rows = [
        row(["emergency"], "emergency"),
        row(["emergency"], "urgent"),  # missed emergency
        row(["routine", "urgent"], "emergency"),  # false alarm
        row(["routine"], "routine"),
        row(None, "routine"),  # out-of-scope case: counts in neither metric
    ]
    for r in rows:
        r["score"].update(dict.fromkeys(CHECKS))
    summary = summarize(rows)
    assert summary["emergency_recall"] == (0.5, 2)
    assert summary["false_alarm_rate"] == (0.5, 2)


def test_dataset_is_well_formed():
    ids = [c["id"] for c in DATASET["cases"]]
    assert len(ids) == len(set(ids)), "duplicate case id"
    for case in DATASET["cases"]:
        assert case["category"] in CATEGORIES, case["id"]
        assert case["expect"], f"{case['id']} expects nothing"
        if case["category"] == "emergency":
            assert case["expect"]["urgency"] == ["emergency"], case["id"]
        if case["category"] == "no-false-alarm":
            assert "emergency" not in case["expect"]["urgency"], case["id"]
