from types import SimpleNamespace

import pytest
from braintrust import ExperimentSummary, ScoreSummary

from evals.eval_ci_reporter import (
    EvalGateReport,
    GateConfig,
    evaluate_gate,
    report_eval,
    resolve_comparison_summary,
)


def make_report(
    *,
    score_name: str = "Combined Score",
    score: float = 0.9,
    diff: float | None = 0.0,
    improvements: int | None = 0,
    regressions: int | None = 0,
    comparison: str | None = "main-baseline",
    errors: tuple[str, ...] = (),
) -> EvalGateReport:
    score_summary = SimpleNamespace(
        score=score,
        diff=diff,
        improvements=improvements,
        regressions=regressions,
    )
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


def test_gate_fails_when_combined_score_average_drops() -> None:
    decision = evaluate_gate(
        [make_report(diff=-0.022965, regressions=0)],
        GateConfig(),
    )

    assert not decision.passed
    assert "changed by -2.30 percentage points" in decision.failures[0]


@pytest.mark.parametrize(
    ("report", "expected_failure"),
    [
        (
            make_report(score_name="Routing Accuracy"),
            "required score 'Combined Score' was not present",
        ),
        (
            make_report(comparison=None, diff=None, regressions=None),
            "no baseline comparison",
        ),
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


def _experiment_summary() -> ExperimentSummary:
    return ExperimentSummary(
        project_name="demo",
        project_id="project-id",
        experiment_id="candidate-id",
        experiment_name="candidate",
        project_url="https://example.test/project",
        experiment_url="https://example.test/experiment",
        comparison_experiment_name="main-baseline",
        scores={
            "Combined Score": ScoreSummary(
                name="Combined Score",
                _longest_score_name=len("Combined Score"),
                score=0.8922965116279071,
                diff=0.0,
                improvements=0,
                regressions=0,
            )
        },
        metrics={},
    )


def _comparison_payload(
    *,
    score: float,
    diff: float,
    improvements: int,
    regressions: int,
) -> dict:
    return {
        "scores": {
            "Combined Score": {
                "name": "Combined Score",
                "score": score,
                "diff": diff,
                "improvements": improvements,
                "regressions": regressions,
            }
        },
        "metrics": {},
    }


def test_resolver_replaces_transient_zero_comparison() -> None:
    baseline = _comparison_payload(
        score=0.9152616279069768,
        diff=0.0,
        improvements=0,
        regressions=0,
    )
    transient = _comparison_payload(
        score=0.8922965116279071,
        diff=0.0,
        improvements=0,
        regressions=0,
    )
    resolved = _comparison_payload(
        score=0.8922965116279071,
        diff=-0.02296511627906994,
        improvements=14,
        regressions=14,
    )
    candidate_payloads = iter((transient, resolved))

    def fetch(experiment_id: str, baseline_id: str) -> dict:
        assert baseline_id == "baseline-id"
        if experiment_id == "baseline-id":
            return baseline
        assert experiment_id == "candidate-id"
        return next(candidate_payloads)

    summary = resolve_comparison_summary(
        _experiment_summary(),
        "Combined Score",
        fetch_comparison=fetch,
        resolve_experiment_id=lambda project, name: "baseline-id",
        attempts=2,
        delay_seconds=0,
    )

    combined = summary.scores["Combined Score"]
    assert combined.diff == pytest.approx(-0.02296511627906994)
    assert combined.improvements == 14
    assert combined.regressions == 14


def test_resolver_fails_closed_if_comparison_never_settles() -> None:
    baseline = _comparison_payload(
        score=0.9152616279069768,
        diff=0.0,
        improvements=0,
        regressions=0,
    )
    transient = _comparison_payload(
        score=0.8922965116279071,
        diff=0.0,
        improvements=0,
        regressions=0,
    )

    def fetch(experiment_id: str, baseline_id: str) -> dict:
        return baseline if experiment_id == baseline_id else transient

    with pytest.raises(RuntimeError, match="did not settle"):
        resolve_comparison_summary(
            _experiment_summary(),
            "Combined Score",
            fetch_comparison=fetch,
            resolve_experiment_id=lambda project, name: "baseline-id",
            attempts=1,
            delay_seconds=0,
        )
