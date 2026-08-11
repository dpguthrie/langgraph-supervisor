# Braintrust PR score gate

The `Braintrust eval gate` workflow runs the supervisor eval for every pull
request and every push to `main`. It uses `braintrustdata/eval-action@v2` for
the live PR comment and a custom Braintrust reporter for the pass/fail policy.

For its Python runtime, `eval-action@v2` currently invokes
`uv run braintrust eval --jsonl`; it does not invoke the standalone `bt eval`
command. The local verification command below intentionally uses the same
Python CLI path as the action so it exercises the reporter-controlled exit
status as well as the eval itself.

## Gate policy

By default, `evals/eval_ci_reporter.py` requires all of the following:

- The experiment completed without task or scorer errors.
- The Braintrust experiment summary contains the project aggregate score named
  `Combined Score`.
- Braintrust found a baseline experiment and produced comparison data.
- `Combined Score` has zero regressed test cases relative to that baseline.
- The average `Combined Score` did not decrease relative to that baseline.

The reporter fails closed: a missing aggregate score or missing baseline fails
the check instead of silently allowing a merge. It also preserves the standard
experiment-summary JSONL, so `eval-action@v2` can continue building its PR
comment.

Project aggregate scores are resolved asynchronously after eval rows are
uploaded. The reporter reads the candidate experiment's stored `base_exp_id`,
passes that id explicitly to Braintrust's experiment-summary endpoint, and
verifies the completed comparison before it decides or emits JSONL. This
prevents an unrelated project baseline or a transient `+0pp` comparison from
producing a false pass or an incorrect PR comment.

The policy can be configured with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRAINTRUST_CI_GATE_SCORE` | `Combined Score` | Exact score or aggregate-score name to gate. |
| `BRAINTRUST_CI_MAX_REGRESSIONS` | `0` | Maximum number of regressed test cases allowed. |
| `BRAINTRUST_CI_MAX_SCORE_DROP` | `0` | Maximum allowed average-score drop as a fraction (`0.01` = one percentage point). |
| `BRAINTRUST_CI_MIN_SCORE` | unset | Optional absolute score floor from `0` to `1`. |
| `BRAINTRUST_CI_REQUIRE_BASELINE` | `true` | Require comparison data before passing. |
| `BRAINTRUST_CI_COMPARISON_ATTEMPTS` | `30` | Maximum server-comparison reads while aggregate scores settle. |
| `BRAINTRUST_CI_COMPARISON_DELAY_SECONDS` | `2` | Delay between comparison reads. |

Keep the default fail-closed behavior for protected branches. A tolerance or
score floor should be an explicit product decision, not an implicit fallback in
the reporter.

## Braintrust setup

1. In the project settings, define an aggregate score with the exact name used
   by `BRAINTRUST_CI_GATE_SCORE`. This repository uses `Combined Score`.
2. Ensure the eval's underlying scorers are selected in that aggregate score.
3. Ensure Braintrust can select a meaningful base experiment. The checkout uses
   full git history (`fetch-depth: 0`) so Braintrust can select the closest
   experiment on `main` when git metadata collection is enabled. A project-wide
   default baseline can be used when a fixed release baseline is preferred.
4. Run an eval on `main` before enabling the required check, so the first pull
   request has a baseline to compare against.

Aggregate scores are calculated by Braintrust and included in experiment
summaries. The reporter reads that server-provided score; it does not duplicate
the aggregate-score formula in repository code.

## GitHub setup

Add these repository or organization secrets:

- `BRAINTRUST_API_KEY`
- `BRAINTRUST_PROJECT_NAME`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Then add `Combined Score regression gate` as a required status check in the
branch ruleset for `main`. A failing workflow only becomes a merge gate after
GitHub is configured to require that check.

The workflow intentionally runs on every pull request rather than using a
workflow-level path filter. Required checks that are skipped by path filtering
can remain pending and block unrelated pull requests without ever producing a
result.

## Local verification

Include both the eval and reporter files in the same CLI invocation:

```bash
set -a && source .env && set +a
uv run braintrust eval \
  evals/eval_supervisor.py \
  evals/eval_ci_reporter.py
```

The command exits `1` if the configured gate fails and `0` if it passes. The
experiment is still uploaded to Braintrust in either case, which keeps the
comparison available for diagnosis.
