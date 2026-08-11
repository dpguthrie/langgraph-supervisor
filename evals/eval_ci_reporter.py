"""Braintrust reporter that turns score regressions into a CI exit status.

The reporter deliberately emits the normal experiment-summary JSONL consumed by
``braintrustdata/eval-action``. Returning ``False`` from ``report_run`` then makes
the Braintrust CLI, and therefore the GitHub Action, exit non-zero.
"""

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from braintrust import MetricSummary, Reporter, ScoreSummary, api_conn, login

DEFAULT_GATE_SCORE = "Combined Score"
GATE_SCORE_ENV = "BRAINTRUST_CI_GATE_SCORE"
MAX_REGRESSIONS_ENV = "BRAINTRUST_CI_MAX_REGRESSIONS"
MAX_SCORE_DROP_ENV = "BRAINTRUST_CI_MAX_SCORE_DROP"
MIN_SCORE_ENV = "BRAINTRUST_CI_MIN_SCORE"
REQUIRE_BASELINE_ENV = "BRAINTRUST_CI_REQUIRE_BASELINE"
COMPARISON_ATTEMPTS_ENV = "BRAINTRUST_CI_COMPARISON_ATTEMPTS"
COMPARISON_DELAY_ENV = "BRAINTRUST_CI_COMPARISON_DELAY_SECONDS"

DEFAULT_COMPARISON_ATTEMPTS = 30
DEFAULT_COMPARISON_DELAY_SECONDS = 2.0
COMPARISON_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GateConfig:
    """Configuration for the score gate."""

    score_name: str = DEFAULT_GATE_SCORE
    max_regressions: int = 0
    max_score_drop: float = 0.0
    min_score: float | None = None
    require_baseline: bool = True

    @classmethod
    def from_env(cls) -> "GateConfig":
        score_name = os.getenv(GATE_SCORE_ENV, DEFAULT_GATE_SCORE).strip()
        if not score_name:
            raise ValueError(f"{GATE_SCORE_ENV} must not be empty")

        max_regressions = _parse_nonnegative_int(MAX_REGRESSIONS_ENV, default=0)
        max_score_drop = _parse_nonnegative_float(MAX_SCORE_DROP_ENV, default=0.0)
        min_score = _parse_optional_score(MIN_SCORE_ENV)
        require_baseline = _parse_bool(REQUIRE_BASELINE_ENV, default=True)
        return cls(
            score_name=score_name,
            max_regressions=max_regressions,
            max_score_drop=max_score_drop,
            min_score=min_score,
            require_baseline=require_baseline,
        )


@dataclass(frozen=True)
class EvalGateReport:
    """The information retained from one evaluator for the run-level gate."""

    summary: Any
    errors: tuple[str, ...]


@dataclass(frozen=True)
class GateDecision:
    """A gate result and the human-readable reasons behind it."""

    passed: bool
    failures: tuple[str, ...]
    observations: tuple[dict[str, Any], ...]


def _parse_nonnegative_int(name: str, *, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _parse_positive_int(name: str, *, default: int) -> int:
    value = _parse_nonnegative_int(name, default=default)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _parse_nonnegative_float(name: str, *, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _parse_optional_score(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _parse_bool(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw_value!r}")


def _fetch_experiment_metadata(experiment_id: str) -> dict[str, Any]:
    """Fetch immutable experiment metadata, including its stored base id."""

    login()
    return api_conn().get_json(f"v1/experiment/{experiment_id}")


def _fetch_experiment_summary(experiment_id: str, baseline_id: str) -> dict[str, Any]:
    """Fetch a server summary against an explicit baseline experiment id."""

    login()
    return api_conn().get_json(
        f"v1/experiment/{experiment_id}/summarize",
        args={
            "summarize_scores": "true",
            "comparison_experiment_id": baseline_id,
        },
    )


def _comparison_is_settled(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    """Reject the transient all-zero comparison returned before aggregates settle."""

    required_candidate = ("score", "diff", "improvements", "regressions")
    if any(candidate.get(field) is None for field in required_candidate):
        return False
    if baseline.get("score") is None:
        return False

    expected_diff = float(candidate["score"]) - float(baseline["score"])
    reported_diff = float(candidate["diff"])
    if math.isclose(
        expected_diff,
        reported_diff,
        rel_tol=0.0,
        abs_tol=COMPARISON_TOLERANCE,
    ):
        return True

    # A comparison can use a matched subset when experiment datasets differ.
    # Non-zero comparison evidence is still authoritative in that case. The
    # problematic transient response has zero diff and zero changed examples.
    return bool(
        reported_diff or int(candidate["improvements"]) or int(candidate["regressions"])
    )


def _summary_from_comparison(summary: Any, payload: dict[str, Any]) -> Any:
    """Replace a summary's scores/metrics with the resolved server response."""

    score_items = payload.get("scores", {})
    metric_items = payload.get("metrics", {})
    longest_score_name = max(map(len, score_items), default=0)
    longest_metric_name = max(map(len, metric_items), default=0)

    scores = {
        name: ScoreSummary(_longest_score_name=longest_score_name, **item)
        for name, item in score_items.items()
    }
    metrics = {
        name: MetricSummary(_longest_metric_name=longest_metric_name, **item)
        for name, item in metric_items.items()
    }
    return replace(
        summary,
        comparison_experiment_name=payload.get("comparison_experiment_name"),
        scores=scores,
        metrics=metrics,
    )


def resolve_comparison_summary(
    summary: Any,
    score_name: str,
    *,
    fetch_experiment_metadata: Callable[
        [str], dict[str, Any]
    ] = _fetch_experiment_metadata,
    fetch_experiment_summary: Callable[
        [str, str], dict[str, Any]
    ] = _fetch_experiment_summary,
    attempts: int | None = None,
    delay_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Wait for project aggregate-score comparison data to become consistent."""

    experiment_id = getattr(summary, "experiment_id", None)
    if not experiment_id:
        return summary

    attempts = attempts or _parse_positive_int(
        COMPARISON_ATTEMPTS_ENV, default=DEFAULT_COMPARISON_ATTEMPTS
    )
    delay_seconds = (
        _parse_nonnegative_float(
            COMPARISON_DELAY_ENV, default=DEFAULT_COMPARISON_DELAY_SECONDS
        )
        if delay_seconds is None
        else delay_seconds
    )

    experiment_metadata = fetch_experiment_metadata(experiment_id)
    baseline_id = experiment_metadata.get("base_exp_id")
    if not baseline_id:
        raise RuntimeError(
            f"experiment {experiment_id!r} does not have a stored base experiment"
        )

    baseline_payload = fetch_experiment_summary(baseline_id, baseline_id)
    baseline_score = baseline_payload.get("scores", {}).get(score_name)
    if baseline_score is None:
        raise RuntimeError(
            f"baseline {baseline_id!r} does not contain score {score_name!r}"
        )

    last_score: dict[str, Any] | None = None
    for attempt in range(attempts):
        if attempt > 0 and delay_seconds:
            sleep(delay_seconds)

        payload = fetch_experiment_summary(experiment_id, baseline_id)
        last_score = payload.get("scores", {}).get(score_name)
        if last_score is not None and _comparison_is_settled(
            last_score, baseline_score
        ):
            return _summary_from_comparison(summary, payload)

    raise RuntimeError(
        f"comparison for {score_name!r} did not settle after {attempts} attempts; "
        f"candidate={last_score!r}, baseline={baseline_score!r}"
    )


def evaluate_gate(
    reports: Iterable[EvalGateReport], config: GateConfig
) -> GateDecision:
    """Evaluate every experiment summary and fail closed on missing gate data."""

    failures: list[str] = []
    observations: list[dict[str, Any]] = []
    report_count = 0

    for report in reports:
        report_count += 1
        summary = report.summary
        experiment_name = getattr(summary, "experiment_name", "<unknown experiment>")
        prefix = f"{experiment_name}:"

        if report.errors:
            failures.append(f"{prefix} {len(report.errors)} eval error(s)")

        scores = getattr(summary, "scores", {})
        score_summary = scores.get(config.score_name)
        if score_summary is None:
            failures.append(
                f"{prefix} required score {config.score_name!r} was not present"
            )
            continue

        comparison_name = getattr(summary, "comparison_experiment_name", None)
        regressions = getattr(score_summary, "regressions", None)
        improvements = getattr(score_summary, "improvements", None)
        score = getattr(score_summary, "score", None)
        diff = getattr(score_summary, "diff", None)

        observations.append(
            {
                "experiment": experiment_name,
                "baseline": comparison_name,
                "score_name": config.score_name,
                "score": score,
                "diff": diff,
                "improvements": improvements,
                "regressions": regressions,
            }
        )

        if config.require_baseline and (
            comparison_name is None or regressions is None or diff is None
        ):
            failures.append(
                f"{prefix} no baseline comparison was available for "
                f"{config.score_name!r}"
            )
        else:
            if regressions is not None and regressions > config.max_regressions:
                failures.append(
                    f"{prefix} {config.score_name!r} had {regressions} "
                    f"regression(s); maximum allowed is {config.max_regressions}"
                )
            if diff is not None and diff < -(
                config.max_score_drop + COMPARISON_TOLERANCE
            ):
                failures.append(
                    f"{prefix} {config.score_name!r} changed by "
                    f"{diff * 100:+.2f} percentage points; maximum allowed "
                    f"drop is {config.max_score_drop * 100:.2f} percentage points"
                )

        if config.min_score is not None:
            if score is None:
                failures.append(
                    f"{prefix} {config.score_name!r} did not contain an average score"
                )
            elif score < config.min_score:
                failures.append(
                    f"{prefix} {config.score_name!r} was {score:.2%}; "
                    f"minimum required is {config.min_score:.2%}"
                )

    if report_count == 0:
        failures.append("no eval results were reported")

    return GateDecision(
        passed=not failures,
        failures=tuple(failures),
        observations=tuple(observations),
    )


def _format_error(result: Any, *, verbose: bool, jsonl: bool) -> str:
    if result.exc_info and (verbose or jsonl):
        return result.exc_info
    if result.error is not None:
        return "".join(
            traceback.format_exception_only(type(result.error), result.error)
        ).strip()
    return "Unknown eval error"


def report_eval(
    evaluator: Any, result: Any, verbose: bool, jsonl: bool
) -> EvalGateReport:
    """Preserve standard output while retaining enough data to enforce the gate."""

    errors = tuple(
        _format_error(eval_result, verbose=verbose, jsonl=jsonl)
        for eval_result in result.results
        if eval_result.error is not None
    )

    summary = result.summary
    comparison_resolved = True
    try:
        summary = resolve_comparison_summary(
            summary,
            GateConfig.from_env().score_name,
        )
    except Exception as exc:
        comparison_resolved = False
        errors += (f"Braintrust comparison resolution failed: {exc}",)

    if errors:
        if jsonl:
            # eval-action@v2 normalizes this key to ``evaluatorName``.
            print(json.dumps({"evaluator_name": evaluator.eval_name, "errors": errors}))
        else:
            print(
                f"Evaluator {evaluator.eval_name} failed with {len(errors)} error(s)",
                file=sys.stderr,
            )
            for error in errors:
                print(error, file=sys.stderr)

    # eval-action@v2 reads this JSONL to build and update its PR comment. Never
    # emit the original summary when comparison resolution failed: it can carry
    # the transient +0pp values that this reporter is designed to reject.
    if comparison_resolved:
        print(json.dumps(summary.as_dict()) if jsonl else summary)
    return EvalGateReport(summary=summary, errors=errors)


def report_run(results: list[EvalGateReport], verbose: bool, jsonl: bool) -> bool:
    """Return False when the configured scorer violates the CI policy."""

    del verbose
    config = GateConfig.from_env()
    decision = evaluate_gate(results, config)

    if jsonl:
        print(
            json.dumps(
                {
                    "braintrust_ci_gate": {
                        "passed": decision.passed,
                        "score": config.score_name,
                        "observations": decision.observations,
                        "failures": decision.failures,
                    }
                }
            )
        )
    elif decision.passed:
        print(f"Braintrust CI gate passed for {config.score_name!r}.")
    else:
        print(f"Braintrust CI gate failed for {config.score_name!r}:", file=sys.stderr)
        for failure in decision.failures:
            print(f"- {failure}", file=sys.stderr)

    return decision.passed


Reporter(
    "combined-score-ci-gate",
    report_eval=report_eval,
    report_run=report_run,
)
