# Configurable Eval Design for deep_agent Architecture

**Date:** 2025-10-30
**Status:** Approved

## Overview

Update the evaluation system to support the new deep_agent architecture and enable prompt experimentation through Braintrust remote evals. The design allows a single eval file to work both locally (with defaults) and remotely (with configurable parameters).

## Goals

1. Update `eval_simple.py` to work with the new deep_agent/SubAgentMiddleware architecture
2. Add configurable parameters for prompts and model selection
3. Enable remote eval support for prompt engineering experiments in Braintrust UI
4. Maintain backward compatibility - local runs work without changes

## Architecture

### 1. AgentConfig Class

**Location:** `src/config.py` (new file)

A Pydantic-based configuration class that encapsulates all configurable aspects of the agent system:

```python
class AgentConfig(BaseModel):
    # Supervisor/System prompts
    system_prompt: str | None = None

    # Subagent prompts
    research_agent_prompt: str | None = None
    math_agent_prompt: str | None = None

    # Subagent routing descriptions
    research_agent_description: str | None = None
    math_agent_description: str | None = None

    # Model selections
    supervisor_model: str = "gpt-4o-mini"
    research_model: str = "gpt-4o-mini"
    math_model: str = "gpt-4o-mini"
```

**Design principles:**
- All fields optional or have defaults
- `None` means "use existing default from deep_agent.py"
- Type-safe with Pydantic validation
- Can be instantiated from dict (for remote eval parameters)

### 2. deep_agent.py Refactoring

**Location:** `src/agents/deep_agent.py` (modified)

Update `get_supervisor()` to accept optional configuration:

```python
def get_supervisor(config: AgentConfig | None = None):
    if config is None:
        config = AgentConfig()  # Use all defaults

    # Use config values or fall back to module constants
    system_prompt = config.system_prompt if config.system_prompt else (SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT)

    # Pass config values to subagent initialization
    # Pass model selections from config
    # Pass routing descriptions to SubAgentMiddleware

    # Caching: only cache when config is None (default behavior)
```

**Key changes:**
- Add `config` parameter to `get_supervisor()`
- Conditionally use config values or existing defaults
- Pass prompts to subagent constructors
- Pass model names to agent initialization
- Pass routing descriptions to SubAgentMiddleware
- Disable caching when custom config provided

**Backward compatibility:**
Existing code calling `get_supervisor()` without arguments continues to work exactly as before.

### 3. Eval Structure

**Location:** `evals/eval_simple.py` (modified)

Update eval to support both local and remote execution:

```python
class EvalParameters(BaseModel):
    system_prompt: str | None = None
    research_agent_prompt: str | None = None
    math_agent_prompt: str | None = None
    research_agent_description: str | None = None
    math_agent_description: str | None = None
    supervisor_model: str = "gpt-4o-mini"
    research_model: str = "gpt-4o-mini"
    math_model: str = "gpt-4o-mini"

def run_supervisor_task(input_data: dict, hooks: Any, parameters: dict = None) -> dict[str, list]:
    # Extract parameters (remote eval) or use empty dict (local eval)
    params = parameters or {}

    # Build AgentConfig from parameters
    config = AgentConfig(**{k: v for k, v in params.items() if v is not None})

    # Get supervisor with config
    supervisor = get_supervisor(config)

    # Execute task and return results
```

**Parameter handling:**
- Local run: `parameters` is None, all defaults used
- Remote run: `parameters` dict provided by Braintrust UI
- Only non-None parameters override defaults

### 4. Routing Detection

**Strategy:** Defer to implementation phase

The deep_agent/SubAgentMiddleware likely already annotates which subagent handled each message in the message metadata. During implementation, we'll examine the actual message structure from `supervisor.stream()` to find the cleanest way to extract agent routing information.

**Fallback:** If needed, we can detect by tool names (Tavily = research, arithmetic = math), but there's likely a better approach built into the framework.

## Usage Workflows

### Local Evaluation

```bash
python evals/eval_simple.py
```

- Uses all default prompts from source code
- Uses default models (gpt-4o-mini)
- Runs against local dataset

### Remote Evaluation (Development Server)

```bash
braintrust eval evals/eval_simple.py --dev
```

- Exposes eval in Braintrust playground UI
- All parameters become interactive UI controls
- Experiment with different prompts/models in real-time
- Results displayed immediately
- Copy winning prompts back to source code

### Benefits

1. **Rapid iteration:** Test prompt variations without code changes
2. **A/B testing:** Compare different prompt strategies side-by-side
3. **Model comparison:** Easily test gpt-4o vs gpt-4o-mini vs other models
4. **Routing experimentation:** Tune subagent descriptions for better routing accuracy
5. **No deployment friction:** Changes in UI, no git commits needed for experiments

## Implementation Checklist

- [ ] Create `src/config.py` with `AgentConfig` class
- [ ] Refactor `src/agents/deep_agent.py` to accept config parameter
- [ ] Update subagent initialization to use config values
- [ ] Update `evals/eval_simple.py` with parameter support
- [ ] Test local execution (no parameters)
- [ ] Test remote execution (with parameters)
- [ ] Investigate and fix routing detection for deep_agent
- [ ] Update scorers if needed based on new message structure
- [ ] Verify all existing scorers work correctly

## Non-Goals

- Changing the evaluation dataset
- Modifying the scoring functions (routing accuracy, response quality, etc.)
- Adding new agent types
- Changing the deep_agent architecture itself

## Trade-offs

**Chosen approach:** Config class with dependency injection

**Pros:**
- Type-safe with Pydantic
- Clear, explicit parameter passing
- Easy to test and reason about
- Plays well with Braintrust parameter system

**Cons:**
- Requires refactoring deep_agent.py
- More code than monkey-patching approach

**Alternative approaches considered:**
1. Factory function: Would require more extensive refactoring
2. Monkey-patching: Less maintainable, harder to debug

The config class approach balances cleanliness with implementation effort.
