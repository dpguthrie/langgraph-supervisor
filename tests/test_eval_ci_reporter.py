from types import SimpleNamespace

import pytest

from evals.eval_ci_reporter import (
    EvalGateReport,
    GateConfig,
    evaluate_gate,
    report_eval,
)


def make_report(
    *,
    score_name: str = "Combined Score",
    score: float = 0.9,
    regressions: int | None = 0,
    comparison: str | None = "main-baseline",
    errors: tuple[str, ...] = (),
) -> EvalGateReport:
    score_summary = SimpleNamespace(score=score, regressions=regressions)
    summary = SimpleNamespace(
        experiment_name="candidate",
        comparison_experiment_name=comparison,
        scores={score_name: score_summary},
    )
    return EvalGateReport(summary=summary, errors=errors)


def test_gate_passes_when_combined_score_has_no_regressions() -> None:
    decision = evaluate_gate([make_report()], GateConfig())

    assert decision.passed
    assert decision.failures == ()


def test_gate_fails_when_combined_score_regresses() -> None:
    decision = evaluate_gate([make_report(regressions=1)], GateConfig())

    assert not decision.passed
    assert "had 1 regression(s)" in decision.failures[0]


@pytest.mark.parametrize(
    ("report", "expected_failure"),
    [
        (
            make_report(score_name="Routing Accuracy"),
            "required score 'Combined Score' was not present",
        ),
        (make_report(comparison=None, regressions=None), "no baseline comparison"),
        (make_report(errors=("task failed",)), "1 eval error(s)"),
    ],
)
def test_gate_fails_closed(report: EvalGateReport, expected_failure: str) -> None:
    decision = evaluate_gate([report], GateConfig())

    assert not decision.passed
    assert any(expected_failure in failure for failure in decision.failures)


def test_gate_can_enforce_an_absolute_score_floor() -> None:
    config = GateConfig(min_score=0.85)

    decision = evaluate_gate([make_report(score=0.84)], config)

    assert not decision.passed
    assert "minimum required is 85.00%" in decision.failures[0]


def test_report_eval_emits_summary_jsonl_for_eval_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_payload = {
        "project_name": "demo",
        "experiment_name": "candidate",
        "scores": {"Combined Score": {"score": 0.9, "regressions": 0}},
        "metrics": {},
    }
    summary = SimpleNamespace(as_dict=lambda: summary_payload)
    result = SimpleNamespace(summary=summary, results=[])
    evaluator = SimpleNamespace(eval_name="supervisor")

    report = report_eval(evaluator, result, verbose=False, jsonl=True)

    assert report.summary is summary
    assert capsys.readouterr().out.strip() == (
        '{"project_name": "demo", "experiment_name": "candidate", '
        '"scores": {"Combined Score": {"score": 0.9, "regressions": 0}}, '
        '"metrics": {}}'
    )


def test_report_eval_emits_eval_action_compatible_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = SimpleNamespace(as_dict=lambda: {"experiment_name": "candidate"})
    failed_result = SimpleNamespace(
        error=ValueError("bad scorer"),
        exc_info="ValueError: bad scorer",
    )
    result = SimpleNamespace(summary=summary, results=[failed_result])
    evaluator = SimpleNamespace(eval_name="supervisor")

    report = report_eval(evaluator, result, verbose=False, jsonl=True)

    assert report.errors == ("ValueError: bad scorer",)
    assert capsys.readouterr().out.splitlines()[0] == (
        '{"evaluator_name": "supervisor", "errors": ["ValueError: bad scorer"]}'
    )
