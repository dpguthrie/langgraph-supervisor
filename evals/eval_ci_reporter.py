"""Braintrust reporter that turns score regressions into a CI exit status.

The reporter deliberately emits the normal experiment-summary JSONL consumed by
``braintrustdata/eval-action``. Returning ``False`` from ``report_run`` then makes
the Braintrust CLI, and therefore the GitHub Action, exit non-zero.
"""

import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Iterable

from braintrust import Reporter

DEFAULT_GATE_SCORE = "Combined Score"
GATE_SCORE_ENV = "BRAINTRUST_CI_GATE_SCORE"
MAX_REGRESSIONS_ENV = "BRAINTRUST_CI_MAX_REGRESSIONS"
MIN_SCORE_ENV = "BRAINTRUST_CI_MIN_SCORE"
REQUIRE_BASELINE_ENV = "BRAINTRUST_CI_REQUIRE_BASELINE"


@dataclass(frozen=True)
class GateConfig:
    """Configuration for the score gate."""

    score_name: str = DEFAULT_GATE_SCORE
    max_regressions: int = 0
    min_score: float | None = None
    require_baseline: bool = True

    @classmethod
    def from_env(cls) -> "GateConfig":
        score_name = os.getenv(GATE_SCORE_ENV, DEFAULT_GATE_SCORE).strip()
        if not score_name:
            raise ValueError(f"{GATE_SCORE_ENV} must not be empty")

        max_regressions = _parse_nonnegative_int(MAX_REGRESSIONS_ENV, default=0)
        min_score = _parse_optional_score(MIN_SCORE_ENV)
        require_baseline = _parse_bool(REQUIRE_BASELINE_ENV, default=True)
        return cls(
            score_name=score_name,
            max_regressions=max_regressions,
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


def _parse_nonnegative_int(name: str, *, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value < 0:
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
    if not 0 <= value <= 1:
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


def evaluate_gate(
    reports: Iterable[EvalGateReport], config: GateConfig
) -> GateDecision:
    """Evaluate every experiment summary and fail closed on missing gate data."""

    failures: list[str] = []
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
        score = getattr(score_summary, "score", None)

        if config.require_baseline and (comparison_name is None or regressions is None):
            failures.append(
                f"{prefix} no baseline comparison was available for "
                f"{config.score_name!r}"
            )
        elif regressions is not None and regressions > config.max_regressions:
            failures.append(
                f"{prefix} {config.score_name!r} had {regressions} regression(s); "
                f"maximum allowed is {config.max_regressions}"
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

    return GateDecision(passed=not failures, failures=tuple(failures))


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

    # eval-action@v2 reads this JSONL to build and update its PR comment.
    print(json.dumps(result.summary.as_dict()) if jsonl else result.summary)
    return EvalGateReport(summary=result.summary, errors=errors)


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
