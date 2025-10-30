# Configurable Eval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable prompt and model configuration for the deep_agent supervisor system to support Braintrust remote eval experimentation.

**Architecture:** Introduce an `AgentConfig` Pydantic model for type-safe configuration. Refactor `get_supervisor()` and subagent initialization to accept optional config. Update eval to pass parameters from Braintrust UI through to the supervisor.

**Tech Stack:** Pydantic for config validation, LangGraph for agent orchestration, Braintrust for evaluation framework

---

## Task 1: Create AgentConfig Class

**Files:**
- Create: `src/config.py`
- Test: Test manually by importing in Python REPL

**Step 1: Create the config module with AgentConfig**

File: `src/config.py`

```python
"""Configuration for the deep agent supervisor and subagents."""

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for the deep agent supervisor and subagents.

    All fields are optional with sensible defaults. None means "use the
    existing default from the agent module".
    """

    # Supervisor/System prompt
    system_prompt: str | None = None

    # Subagent prompts
    research_agent_prompt: str | None = None
    math_agent_prompt: str | None = None

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str | None = None
    math_agent_description: str | None = None

    # Model selections
    supervisor_model: str = "gpt-4o-mini"
    research_model: str = "gpt-4o-mini"
    math_model: str = "gpt-4o-mini"

    class Config:
        # Allow usage in eval context
        arbitrary_types_allowed = True
```

**Step 2: Verify config can be instantiated**

Run in Python:
```bash
source .venv/bin/activate
python3 -c "from src.config import AgentConfig; c = AgentConfig(); print(c)"
```

Expected output: Should show the config with default values

**Step 3: Verify config accepts parameters**

Run in Python:
```bash
python3 -c "from src.config import AgentConfig; c = AgentConfig(system_prompt='test', supervisor_model='gpt-4o'); print(c.system_prompt, c.supervisor_model)"
```

Expected output: `test gpt-4o`

**Step 4: Commit**

```bash
git add src/config.py
git commit -m "feat: add AgentConfig for configurable prompts and models

Add Pydantic-based configuration class to enable:
- Custom prompts for supervisor and subagents
- Custom routing descriptions
- Model selection per agent
- Type-safe parameter passing from evals

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Refactor Research Agent to Accept Config

**Files:**
- Modify: `src/agents/research_agent.py`

**Step 1: Update research_agent to accept config parameters**

File: `src/agents/research_agent.py`

Find the function that creates the research agent (around line 18-30). Update it to accept optional prompt and model parameters:

```python
def get_research_agent(
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini"
):
    """Create research agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use (default: gpt-4o-mini)
    """
    # Use provided prompt or fall back to default
    prompt = system_prompt if system_prompt is not None else (
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    tools = [TavilySearchResults(max_results=3)]
    return create_react_agent(
        ChatOpenAI(model=model),
        tools,
        state_modifier=prompt,
    )
```

**Step 2: Verify the changes don't break existing usage**

Run:
```bash
source .venv/bin/activate
python3 -c "from src.agents.research_agent import get_research_agent; agent = get_research_agent(); print('Success')"
```

Expected: `Success` (no errors)

**Step 3: Verify custom parameters work**

Run:
```bash
python3 -c "from src.agents.research_agent import get_research_agent; agent = get_research_agent(system_prompt='Custom prompt', model='gpt-4o'); print('Success')"
```

Expected: `Success` (no errors)

**Step 4: Commit**

```bash
git add src/agents/research_agent.py
git commit -m "refactor: add config params to research agent

Allow custom system prompt and model selection while maintaining
backward compatibility with default parameters.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Refactor Math Agent to Accept Config

**Files:**
- Modify: `src/agents/math_agent.py`

**Step 1: Update math_agent to accept config parameters**

File: `src/agents/math_agent.py`

Find the function that creates the math agent (around line 35-47). Update it to accept optional prompt and model parameters:

```python
def get_math_agent(
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini"
):
    """Create math agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use (default: gpt-4o-mini)
    """
    # Use provided prompt or fall back to default
    prompt = system_prompt if system_prompt is not None else (
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    tools = [add, subtract, multiply, divide]
    return create_react_agent(
        ChatOpenAI(model=model),
        tools,
        state_modifier=prompt,
    )
```

**Step 2: Verify the changes don't break existing usage**

Run:
```bash
source .venv/bin/activate
python3 -c "from src.agents.math_agent import get_math_agent; agent = get_math_agent(); print('Success')"
```

Expected: `Success` (no errors)

**Step 3: Verify custom parameters work**

Run:
```bash
python3 -c "from src.agents.math_agent import get_math_agent; agent = get_math_agent(system_prompt='Custom prompt', model='gpt-4o'); print('Success')"
```

Expected: `Success` (no errors)

**Step 4: Commit**

```bash
git add src/agents/math_agent.py
git commit -m "refactor: add config params to math agent

Allow custom system prompt and model selection while maintaining
backward compatibility with default parameters.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Refactor deep_agent to Accept AgentConfig

**Files:**
- Modify: `src/agents/deep_agent.py`
- Depends on: `src/config.py`, updated research_agent.py, updated math_agent.py

**Step 1: Add import for AgentConfig**

File: `src/agents/deep_agent.py`

Add near the top of the file (after other imports):

```python
from src.config import AgentConfig
```

**Step 2: Update get_supervisor signature and config handling**

Find the `get_supervisor()` function (around line 165). Update it to:

```python
# Module-level cache for default supervisor
_cached_supervisor = None


def get_supervisor(config: AgentConfig | None = None):
    """Get or create a supervisor agent.

    Args:
        config: Optional configuration for prompts and models.
                If None, uses module defaults and caches the result.
                If provided, builds fresh supervisor (no caching).

    Returns:
        Compiled LangGraph agent
    """
    global _cached_supervisor

    # Use cache only when no custom config
    if config is None and _cached_supervisor is not None:
        return _cached_supervisor

    # Create default config if none provided
    if config is None:
        config = AgentConfig()

    # Use config system prompt or fall back to module default
    system_prompt = config.system_prompt if config.system_prompt else (SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT)

    # Rest of supervisor creation...
    # (continue with existing logic, using config values)
```

**Step 3: Update subagent initialization to use config**

In the `get_supervisor()` function, find where subagents are initialized (the lazy initialization section around line 61-90). Update to pass config values:

In the research agent initialization:
```python
research_description = config.research_agent_description if config.research_agent_description else (
    "Research agent with web search capabilities. "
    "Use this agent for: web searches, finding information online, "
    "looking up current events, researching topics, gathering data from the internet, "
    "answering questions that require external knowledge or real-time information."
)

research_agent = NamedSubAgentWrapper(
    get_research_agent(
        system_prompt=config.research_agent_prompt,
        model=config.research_model
    ),
    "Research Agent",
)
```

In the math agent initialization:
```python
math_description = config.math_agent_description if config.math_agent_description else (
    "Math calculation agent with arithmetic tools. "
    "Use this agent for: mathematical calculations, arithmetic operations, "
    "addition, subtraction, multiplication, division, numerical computations, "
    "solving math problems, performing calculations."
)

math_agent = NamedSubAgentWrapper(
    get_math_agent(
        system_prompt=config.math_agent_prompt,
        model=config.math_model
    ),
    "Math Agent",
)
```

**Step 4: Update SubAgentMiddleware to use config descriptions**

Find where SubAgentMiddleware is created (around line 133-140). Update to:

```python
subagent_middleware = SubAgentMiddleware(
    subagents={
        research_description: research_agent,
        math_description: math_agent,
    }
)
```

**Step 5: Update deep agent creation to use config model**

Find where the deep agent LLM is created. Update to use `config.supervisor_model`:

```python
llm = ChatOpenAI(model=config.supervisor_model)
```

**Step 6: Update caching logic at the end**

At the end of `get_supervisor()`, only cache when using default config:

```python
    agent = deep_agent.compile()

    # Only cache when using default config
    if config is None or (
        config.system_prompt is None and
        config.research_agent_prompt is None and
        config.math_agent_prompt is None and
        config.research_agent_description is None and
        config.math_agent_description is None and
        config.supervisor_model == "gpt-4o-mini" and
        config.research_model == "gpt-4o-mini" and
        config.math_model == "gpt-4o-mini"
    ):
        _cached_supervisor = agent

    return agent
```

**Step 7: Verify backward compatibility**

Run:
```bash
source .venv/bin/activate
python3 -c "from src.agents.deep_agent import get_supervisor; s = get_supervisor(); print('Success')"
```

Expected: `Success` (no errors, uses defaults)

**Step 8: Verify custom config works**

Run:
```bash
python3 -c "from src.agents.deep_agent import get_supervisor; from src.config import AgentConfig; c = AgentConfig(supervisor_model='gpt-4o'); s = get_supervisor(c); print('Success')"
```

Expected: `Success` (no errors, uses custom config)

**Step 9: Commit**

```bash
git add src/agents/deep_agent.py
git commit -m "refactor: make deep_agent configurable with AgentConfig

- Accept optional AgentConfig parameter in get_supervisor()
- Pass config values to subagent initialization
- Use custom prompts, models, and routing descriptions
- Maintain backward compatibility (None = use defaults)
- Only cache supervisor when using default config

Enables Braintrust remote evals to experiment with different
prompts and models without code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Update Eval to Support Parameters

**Files:**
- Modify: `evals/eval_simple.py`

**Step 1: Add AgentConfig import**

File: `evals/eval_simple.py`

Add near the top of the file (after other imports from src):

```python
from src.config import AgentConfig  # noqa: E402
```

**Step 2: Update run_supervisor_task to accept parameters**

Find the `run_supervisor_task` function (around line 27). Update the signature:

```python
def run_supervisor_task(input_data: dict, hooks: Any, parameters: dict = None) -> dict[str, list]:
    """Run a single task through the supervisor and return the final response.

    Args:
        input_data: Input data containing messages
        hooks: Braintrust hooks for metadata
        parameters: Optional parameters from Braintrust remote eval UI
    """
```

**Step 3: Build AgentConfig from parameters**

At the start of `run_supervisor_task`, add:

```python
    try:
        # Extract parameters (provided in remote evals, None in local runs)
        params = parameters or {}

        # Build AgentConfig from parameters, filtering out None values
        config_dict = {
            k: v for k, v in params.items()
            if k in [
                "system_prompt",
                "research_agent_prompt",
                "math_agent_prompt",
                "research_agent_description",
                "math_agent_description",
                "supervisor_model",
                "research_model",
                "math_model"
            ] and v is not None
        }

        # Create config (empty dict = all defaults)
        config = AgentConfig(**config_dict) if config_dict else None

        # Get supervisor with config
        supervisor = get_supervisor(config)

        # Extract the human message content from the input structure
        # ... (rest of existing code)
```

**Step 4: Update supervisor import**

Make sure you're importing `get_supervisor` (around line 22):

```python
from src.agents.deep_agent import get_supervisor  # noqa: E402
```

**Step 5: Verify eval runs locally without parameters**

Run:
```bash
source .venv/bin/activate
python3 evals/eval_simple.py
```

Expected: Eval should run successfully with default configuration

**Step 6: Test with mock parameters**

Create a test script `test_eval_params.py`:

```python
from evals.eval_simple import run_supervisor_task

input_data = {
    "messages": [
        {"role": "user", "content": "What is 2 + 2?"}
    ]
}

class MockHooks:
    def __init__(self):
        self.metadata = {}

hooks = MockHooks()
parameters = {
    "system_prompt": "You are a test supervisor",
    "supervisor_model": "gpt-4o-mini"
}

result = run_supervisor_task(input_data, hooks, parameters)
print(f"Success! Got {len(result['messages'])} messages")
```

Run:
```bash
python3 test_eval_params.py
```

Expected: Should run and print message count

**Step 7: Clean up test script**

```bash
rm test_eval_params.py
```

**Step 8: Commit**

```bash
git add evals/eval_simple.py
git commit -m "feat: add parameter support to eval for remote evals

- Accept optional parameters dict in run_supervisor_task()
- Build AgentConfig from parameters
- Pass config to get_supervisor()
- Maintain backward compatibility (local runs work unchanged)

Enables Braintrust remote eval UI to override prompts and models
for rapid experimentation without code changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Add Parameter Definitions for Braintrust UI

**Files:**
- Modify: `evals/eval_simple.py`

**Step 1: Add parameter schema for Braintrust**

File: `evals/eval_simple.py`

Add a comment section before the `Eval()` call (around line 194) documenting the available parameters:

```python
# Parameter definitions for Braintrust remote evals
# When running with `braintrust eval eval_simple.py --dev`, these parameters
# will be exposed in the playground UI for experimentation.
#
# Available parameters:
# - system_prompt: str | None - Custom supervisor/system prompt
# - research_agent_prompt: str | None - Custom research agent prompt
# - math_agent_prompt: str | None - Custom math agent prompt
# - research_agent_description: str | None - Custom routing description for research agent
# - math_agent_description: str | None - Custom routing description for math agent
# - supervisor_model: str - Model for supervisor (default: gpt-4o-mini)
# - research_model: str - Model for research agent (default: gpt-4o-mini)
# - math_model: str - Model for math agent (default: gpt-4o-mini)
#
# All parameters are optional and default to the values in the source code.

# Basic evaluation
Eval(
    "langgraph-supervisor",
    data=init_dataset("langgraph-supervisor", "Supervisor Agent Dataset"),
    task=run_supervisor_task,
    scores=[
        response_quality_scorer,
        routing_accuracy_scorer,
        step_efficiency_scorer,
        source_attribution_scorer,
    ],  # type: ignore
)
```

**Step 2: Verify eval still runs**

Run:
```bash
source .venv/bin/activate
python3 evals/eval_simple.py
```

Expected: Should run successfully

**Step 3: Commit**

```bash
git add evals/eval_simple.py
git commit -m "docs: add parameter documentation for remote evals

Document available parameters for Braintrust remote eval UI.
Lists all configurable prompts and models with descriptions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Update README with Remote Eval Usage

**Files:**
- Modify: `README.md`

**Step 1: Add remote eval section to README**

File: `README.md`

Find the section about running evals (or add near the end). Add:

```markdown
### Remote Evaluation with Braintrust

The evaluation system supports remote execution with configurable parameters:

```bash
# Run as a remote eval with dev server
braintrust eval evals/eval_simple.py --dev
```

This exposes the eval in the Braintrust playground where you can:
- Override system, research, and math agent prompts
- Change routing descriptions for better agent selection
- Test different models (gpt-4o, gpt-4o-mini, etc.)
- Experiment without changing code
- Compare results side-by-side

**Available parameters:**
- `system_prompt` - Custom supervisor prompt
- `research_agent_prompt` - Custom research agent instructions
- `math_agent_prompt` - Custom math agent instructions
- `research_agent_description` - Routing description for research agent
- `math_agent_description` - Routing description for math agent
- `supervisor_model` - Model for supervisor (default: gpt-4o-mini)
- `research_model` - Model for research agent (default: gpt-4o-mini)
- `math_model` - Model for math agent (default: gpt-4o-mini)

All parameters are optional. Omitted parameters use defaults from the source code.
```

**Step 2: Verify README renders correctly**

View the README in GitHub or a markdown preview to ensure formatting is correct.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add remote eval usage guide

Document how to run Braintrust remote evals with configurable
parameters. Lists all available options for prompt experimentation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Integration Testing

**Files:**
- Test: Full integration test of the configurable eval system

**Step 1: Run eval locally with defaults**

```bash
source .venv/bin/activate
python3 evals/eval_simple.py
```

Expected: Eval runs successfully, uses default prompts and models

**Step 2: Test with BRAINTRUST_API_KEY**

Ensure you have your Braintrust API key:
```bash
echo $BRAINTRUST_API_KEY
```

If not set, add to `.env`:
```
BRAINTRUST_API_KEY=your_key_here
```

**Step 3: Run a single eval case manually**

Create test script `test_manual_eval.py`:

```python
import os
from dotenv import load_dotenv
from evals.eval_simple import run_supervisor_task

load_dotenv()

class MockHooks:
    def __init__(self):
        self.metadata = {}

# Test with default config
print("Testing with defaults...")
result = run_supervisor_task(
    {"messages": [{"role": "user", "content": "What is 5 + 3?"}]},
    MockHooks(),
    None
)
print(f"✓ Default config: {len(result['messages'])} messages")

# Test with custom config
print("\nTesting with custom config...")
result = run_supervisor_task(
    {"messages": [{"role": "user", "content": "What is 10 - 4?"}]},
    MockHooks(),
    {
        "system_prompt": "You are a helpful supervisor who routes tasks efficiently.",
        "supervisor_model": "gpt-4o-mini"
    }
)
print(f"✓ Custom config: {len(result['messages'])} messages")

print("\n✅ All manual tests passed!")
```

Run:
```bash
python3 test_manual_eval.py
```

Expected: Both tests pass and show message counts

**Step 4: Clean up test script**

```bash
rm test_manual_eval.py
```

**Step 5: Document test results**

No commit needed, just verification step.

---

## Task 9: Final Verification and Documentation

**Files:**
- Verify: All changes work together
- Update: Any final documentation

**Step 1: Run full eval suite**

```bash
source .venv/bin/activate
python3 evals/eval_simple.py
```

Expected: Complete eval run with all scorers working

**Step 2: Verify all commits are clean**

```bash
git log --oneline -10
```

Expected: Should see all 7 commits with clear messages

**Step 3: Check for any uncommitted changes**

```bash
git status
```

Expected: Clean working directory (or only docs/plans files)

**Step 4: Push branch if ready**

```bash
git push -u origin feature/configurable-eval
```

Expected: Branch pushed successfully

**Step 5: Create summary of changes**

Create a summary in your notes or prepare for PR:

**Summary:**
- Added `AgentConfig` class for type-safe configuration
- Refactored research and math agents to accept custom prompts and models
- Updated `deep_agent.get_supervisor()` to accept optional config
- Modified eval to support parameter passing from Braintrust remote evals
- Maintained full backward compatibility
- Documented remote eval usage in README

**Benefits:**
- Experiment with prompts in Braintrust UI without code changes
- Compare different models and routing strategies
- Type-safe configuration with Pydantic
- Zero impact on existing local eval runs

---

## Completion Checklist

- [ ] Task 1: AgentConfig class created and tested
- [ ] Task 2: Research agent refactored with config support
- [ ] Task 3: Math agent refactored with config support
- [ ] Task 4: deep_agent refactored to use AgentConfig
- [ ] Task 5: Eval updated to accept and use parameters
- [ ] Task 6: Parameter documentation added for Braintrust
- [ ] Task 7: README updated with remote eval guide
- [ ] Task 8: Integration testing completed successfully
- [ ] Task 9: Final verification and documentation complete

## Key Principles Applied

- **DRY**: Config class centralizes all parameter definitions
- **YAGNI**: No unnecessary features, just what's needed for remote evals
- **TDD**: Verify each change immediately after implementation
- **Frequent commits**: Each task gets its own focused commit
- **Backward compatibility**: All existing code continues to work

## Next Steps After Implementation

1. Test the remote eval in Braintrust UI:
   ```bash
   braintrust eval evals/eval_simple.py --dev
   ```

2. Experiment with different prompts to improve routing accuracy

3. Compare model performance (gpt-4o vs gpt-4o-mini)

4. Document any winning prompt configurations back in source code

5. Consider creating preset configs for common experimentation scenarios
