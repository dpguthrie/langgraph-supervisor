# CLAUDE.md - langgraph-supervisor

## Project Overview

Multi-agent AI system built with LangGraph using a supervisor pattern. The supervisor routes user queries to specialized agents (Research Agent for web search, Math Agent for calculations) via DeepAgents' `SubAgentMiddleware`.

- **Braintrust project name**: `langgraph-supervisor`
- **Braintrust project ID**: `7052b097-7051-42d9-b7d6-c97ece46b744`
- **Default model**: `gpt-4o-mini` (all agents, routed through Braintrust Gateway)

## Braintrust: Always Use the `bt` CLI

**Never use MCP tools for Braintrust queries.** Always use the `bt` CLI.

### `bt sql` - SQL Queries

Use `bt sql` for data analysis, aggregation, filtering logs by anything: scores, facets, classifications, IDs, metadata, etc.

**Critical syntax rules:**
- Always query from `project_logs('<PROJECT_ID>')` - never use bare table names
- **For facets/classifications:** use `project_logs('<PROJECT_ID>', shape => 'traces')` — these fields are only available on the traces shape, not the default span-level shape
- For scores and other span-level fields, the default shape (no `shape =>`) works fine
- The different shapes you can use are: `spans`, `traces`, and `summary` (the default is `traces`)
- No subqueries, no JOINs, no UNION, no window functions
- Every query on `project_logs()` **must** include a range filter on `created` (or `_xact_id`, `_pagination_key`, or scope to specific `root_span_id`/`id`)
- Use `NOW() - INTERVAL N day` for time ranges (not `INTERVAL 'N days'`)
- Nested fields use dot notation: `scores."Response Quality"`, `facets.Sentiment`, `classifications.Sentiment[0].label`
- Fields with spaces require double quotes: `scores."Response Quality"`, `scores."Routing Accuracy"`
- `ILIKE` for case-insensitive matching, `MATCH` for full-word search

**Example queries:**

```sql
-- Find negative sentiment traces (requires shape => 'traces')
bt sql "SELECT id, facets.Sentiment, facets.Issues, facets.Task, scores.\"Response Quality\", scores.\"Routing Accuracy\" FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744', shape => 'traces') WHERE facets.Sentiment ILIKE '%Negative%' AND created >= NOW() - INTERVAL 7 day"

-- Sentiment distribution with average scores (requires shape => 'traces')
bt sql "SELECT classifications.Sentiment[0].label AS sentiment_label, COUNT(*) AS count, AVG(scores.\"Response Quality\") AS avg_response_quality, AVG(scores.\"Routing Accuracy\") AS avg_routing_accuracy FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744', shape => 'traces') WHERE classifications.Sentiment[0].label IS NOT NULL AND created >= NOW() - INTERVAL 7 day GROUP BY sentiment_label ORDER BY count DESC"

-- Find poor routing accuracy (default shape)
bt sql "SELECT id, scores.\"Routing Accuracy\", scores.\"Response Quality\" FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744') WHERE is_root = true AND scores.\"Routing Accuracy\" = 0 AND created >= NOW() - INTERVAL 7 day LIMIT 10"

-- Find traces with errors (default shape)
bt sql "SELECT id, error, metrics.errors FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744') WHERE is_root = true AND error IS NOT NULL AND created >= NOW() - INTERVAL 2 day LIMIT 20"

-- Low response quality (default shape)
bt sql "SELECT id, scores.\"Response Quality\" FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744') WHERE is_root = true AND scores.\"Response Quality\" <= 0.5 AND created >= NOW() - INTERVAL 7 day LIMIT 10"

-- Combine facets with classifications (requires shape => 'traces')
bt sql "SELECT facets.Sentiment, classifications.Task[0].label AS topic, COUNT(*) AS count FROM project_logs('7052b097-7051-42d9-b7d6-c97ece46b744', shape => 'traces') WHERE facets.Sentiment IS NOT NULL AND created >= NOW() - INTERVAL 7 day GROUP BY facets.Sentiment, topic ORDER BY count DESC"
```

**Available score columns:** `scores."Response Quality"`, `scores."Routing Accuracy"`, `scores."Step Efficiency (Bundled)"`, `scores."Combined Score"`

**Available facet columns:** `facets.Sentiment`, `facets.Task`, `facets.Issues`

**Available classification columns:** `classifications.Sentiment[0].label`, `classifications.Task[0].label`, `classifications.Issues[0].label`

**Useful aggregate functions:** `COUNT(*)`, `AVG()`, `MIN()`, `MAX()`, `SUM()`, `percentile(expr, p)`, `count_distinct()`

### `bt view` - Browsing Logs and Traces

Use `bt view logs` for quick browsing and `bt view trace`/`bt view span` for drilling into specific traces.

```bash
# Browse recent logs (default 1h window)
bt view logs -p langgraph-supervisor --window 48h --json --limit 50 --non-interactive

# Filter logs
bt view logs -p langgraph-supervisor --window 48h --search "error" --non-interactive

# Drill into a specific trace
bt view trace -p langgraph-supervisor --trace-id <ROOT_SPAN_ID> --json --non-interactive

# View a specific span
bt view span -p langgraph-supervisor --id <SPAN_ID> --json --non-interactive
```

Always pass `--non-interactive` to avoid interactive prompts.

### Running Evaluations

The `braintrust eval` CLI authenticates before the eval script's `load_dotenv()` runs, so you must source `.env` into the shell first:

```bash
# Always source .env before running evals (never inline the key in the command)
set -a && source .env && set +a

# Run all evals
.venv/bin/braintrust eval evals/

# Run specific eval
.venv/bin/braintrust eval evals/eval_supervisor.py
```

### Braintrust Python SDK

Use `braintrust.init_dataset()` directly — **not** `bt = braintrust.login(); bt.init_dataset()`. `login()` returns `None`.

```python
import braintrust
from dotenv import load_dotenv
load_dotenv()

braintrust.login(api_key=os.getenv("BRAINTRUST_API_KEY"))
dataset = braintrust.init_dataset(project="langgraph-supervisor", name="Supervisor Agent Dataset")
```

#### Filtering by dataset tag

Eval files support `EVAL_TAG` to run a subset of the dataset filtered by tag. This is useful for fast iteration on specific slices (e.g. production issues, edge cases).

```bash
# Run only production examples (added from log analysis)
EVAL_TAG=production .venv/bin/braintrust eval evals/eval_supervisor.py

# Run only frustrated user examples
EVAL_TAG=frustrated-user .venv/bin/braintrust eval evals/eval_supervisor.py

# Run only the original baseline rows
EVAL_TAG=classic .venv/bin/braintrust eval evals/eval_supervisor.py

# Run the full dataset (default, no filter)
.venv/bin/braintrust eval evals/eval_supervisor.py
```

**Available tags in Supervisor Agent Dataset:** `classic`, `production`, `negative-sentiment`, `frustrated-user`, `edge-case`, `routing-failure`, `low-quality`, `math`, `research`

You can also swap the dataset entirely via `EVAL_DATASET`:

```bash
EVAL_DATASET="My Other Dataset" .venv/bin/braintrust eval evals/eval_supervisor.py
```

## Key Files

- `src/config.py` - AgentConfig with all default prompts and model settings
- `src/agents/deep_agent.py` - Supervisor orchestrator (main entry point: `get_supervisor()`)
- `src/agents/math_agent.py` - Math agent with add/subtract/multiply/divide tools
- `src/agents/research_agent.py` - Research agent with Tavily web search
- `src/llm.py` - LLM creation via Braintrust Gateway
- `evals/eval_supervisor.py` - Main supervisor eval (routing accuracy, response quality, step efficiency)
- `evals/eval_math_agent.py` - Math agent eval (calculation accuracy, tool usage)
- `evals/eval_research_agent.py` - Research agent eval (web search, source attribution)
- `evals/parameters.py` - Braintrust parameter definitions for prompt experiments
- `scorers.py` - Custom scorer (StepEfficiencyScorer) registered with Braintrust

## Eval Scorers

**Supervisor eval:** Routing Accuracy (trace-based), Response Quality (LLM judge), Step Efficiency
**Math eval:** Calculation Accuracy, Tool Usage, Efficiency, Response Format, Calculation Correctness (LLM judge)
**Research eval:** Web Search Usage, Source Attribution, Efficiency, Answer Quality (LLM judge)
