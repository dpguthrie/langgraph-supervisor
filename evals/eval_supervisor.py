"""
Simple evaluation script for the Agent Assistant.
Run this file to execute basic evaluations.
"""

import os
import sys
from pathlib import Path
from typing import Any, Literal

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval, init_dataset  # noqa: E402
from braintrust.oai import wrap_openai  # noqa: E402
from braintrust_langchain import BraintrustCallbackHandler  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.braintrust_parameter_patch import apply_parameter_patch  # noqa: E402
from evals.parameters import (  # noqa: E402
    MathAgentPromptParam,
    MathModelParam,
    ResearchAgentPromptParam,
    ResearchModelParam,
    SupervisorModelParam,
    SystemPromptParam,
)

# Import our supervisor system
from src.agents.deep_agent import get_supervisor  # noqa: E402
from src.config import (  # noqa: E402
    AgentConfig,
)

load_dotenv()

# Apply the parameter patch for both local dev server and remote Modal deployment
# This fixes the Braintrust SDK's missing default value extraction for Pydantic parameters
apply_parameter_patch()


client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def unwrap_parameters(params: dict) -> dict:
    """Extract parameter values from hooks.parameters.

    Braintrust parameters can be either:
    - Parameter model instances (single-field Pydantic models) - extract 'value' field
    - Parameter classes (when running locally) - instantiate then extract 'value'
    - Plain values (fallback) - use directly

    Args:
        params: Dict of parameter names to Parameter classes/instances/values

    Returns:
        Dict of parameter names to actual values (filters out None)

    Example:
        Input:  {"system_prompt": SystemPromptParam(value="Hello"), "other": "value"}
        Output: {"system_prompt": "Hello", "other": "value"}
    """
    import inspect

    from pydantic import BaseModel

    result = {}
    for key, param in params.items():
        if param is None:
            continue

        # If it's a Pydantic model class (not an instance), instantiate it with defaults
        if inspect.isclass(param) and issubclass(param, BaseModel):
            param_instance = param()  # Instantiate with default values
            # Extract the 'value' field (single-field model pattern)
            if hasattr(param_instance, "value"):
                result[key] = param_instance.value  # type: ignore
            else:
                # Fallback: use the whole instance
                result[key] = param_instance
        # If it's already a Pydantic model instance, extract the 'value' field
        elif isinstance(param, BaseModel):
            if hasattr(param, "value"):
                result[key] = param.value  # type: ignore
            else:
                # Fallback: use the whole instance
                result[key] = param
        # Otherwise use the param directly (fallback for plain values)
        else:
            result[key] = param
    return result


def serialize_message(msg: Any) -> dict:
    """Convert a LangChain message object to a JSON-serializable dict.

    Args:
        msg: LangChain message object (AIMessage, HumanMessage, etc.)

    Returns:
        Dict with message content and metadata
    """
    # Handle different message types
    if hasattr(msg, "content"):
        result = {
            "content": msg.content,
            "role": getattr(msg, "role", getattr(msg, "type", "unknown")),
        }

        # Add tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in msg.tool_calls
            ]

        # Add additional response metadata if present
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            result["response_metadata"] = msg.response_metadata

        return result
    else:
        # Fallback for dict-like objects
        return msg if isinstance(msg, dict) else {"content": str(msg)}


async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, list]:
    """Run a single task through the supervisor and return the final response.

    Args:
        input_data: Input data containing messages
        hooks: Optional Braintrust hooks for metadata tracking and parameters.
               When running remotely, hooks.parameters contains the configurable
               parameters defined in the Eval() constructor.

    Returns:
        Dict containing messages from the supervisor execution
    """
    try:
        # Build AgentConfig from parameters (if provided)
        # When running locally: hooks is None, params is empty dict
        # When running remotely: hooks.parameters contains the config values
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}

        config_params = unwrap_parameters(params)
        config = AgentConfig(**config_params) if config_params else None

        supervisor = get_supervisor(config, force_rebuild=True)

        # Use hooks.span as the parent so LangChain spans nest under the eval trace
        span = hooks.span if hooks and hasattr(hooks, "span") and hooks.span else None
        callback = BraintrustCallbackHandler(logger=span)
        result = await supervisor.ainvoke(
            {"messages": input["messages"]},
            config={"callbacks": [callback]},
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []

        # Serialize messages to JSON-serializable format
        serialized_messages = [serialize_message(m) for m in messages]
        return {"messages": serialized_messages}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


# LLM-as-a-Judge Scoring functions

## Routing Accuracy - Trace Scorer


class RoutingAccuracyOutput(BaseModel):
    """Structured output for routing accuracy evaluation."""

    choice: Literal["A", "B", "C", "D"]
    reasoning: str


ROUTING_ACCURACY_PROMPT = """
You are an expert evaluator of AI agent routing systems. Your task is to determine whether a user question was correctly routed to the appropriate agents.

The system has the following specialized agents:
1. **MathAgent**: Should handle mathematical calculations, arithmetic, equations, numerical problems, and any query requiring computation with specific numbers.
2. **ResearchAgent**: Should handle factual questions, information lookup, current events, geography, history, statistics, and any query requiring external knowledge or web search.

The supervisor can:
- Route to a single agent
- Route to multiple agents (if the query requires both research and math)
- Answer directly without routing (for simple greetings, conversational queries, or ambiguous questions)

**User Question**: {input}

**Agents Called**: {agents_called}

**Evaluation Criteria**:

Math queries (e.g., "What is 25 * 4?", "Calculate 100 + 50"):
- SHOULD route to MathAgent only
- Should NOT route to ResearchAgent unless additional context/research is needed

Research queries (e.g., "Who is the president?", "What is the capital of France?"):
- SHOULD route to ResearchAgent only
- Should NOT route to MathAgent unless calculation is involved

Hybrid queries (e.g., "What year was the Eiffel Tower built? Multiply that by 2."):
- SHOULD route to BOTH ResearchAgent (for the fact) AND MathAgent (for the calculation)
- Order may vary

Simple conversational queries (e.g., "hello", "help me understand this"):
- CAN be answered directly by supervisor (no routing)
- Routing is acceptable but not required

**Task**: Evaluate the routing decision and respond with your reasoning, then select ONE of these options:

(A) CORRECT - All routing decisions were appropriate. This includes:
    - Correct agent(s) called for the query type
    - No routing when direct answer is appropriate (simple greetings, chat)
    - Multiple agents called when query requires both research and calculation

(B) MOSTLY_CORRECT - Routing was generally correct but with minor issues:
    - Correct agents called but could have answered directly
    - Correct primary agent but missed a secondary agent for optimal answer

(C) PARTIALLY_WRONG - Significant routing issues:
    - Wrong agent called but got lucky with the answer
    - Correct agent plus unnecessary additional agent(s)
    - Missing critical agent for the query type

(D) INCORRECT - Routing was wrong:
    - Wrong agent(s) called for the query type
    - No routing when specialized agent was clearly needed
    - Multiple wrong agents called
"""


async def routing_accuracy_scorer(input, output, expected, metadata, trace):
    choice_map = {
        "A": 1.0,
        "B": 0.7,
        "C": 0.3,
        "D": 0.0,
    }
    spans = await trace.get_spans(span_type=["task"])
    agents_called_str = "None (supervisor answered directly)"
    agents_called = []
    for span in spans:
        span_name = span.span_attributes.get("name", None)
        if span_name in ["MathAgent", "ResearchAgent"]:
            agents_called.append(span_name)

    if agents_called:
        agents_called_str = ", ".join(agents_called)

    prompt = ROUTING_ACCURACY_PROMPT.format(
        input=input, agents_called=agents_called_str
    )
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}],
        text_format=RoutingAccuracyOutput,
    )
    output = response.output_parsed
    if output is None:
        return {
            "name": "Routing Accuracy",
            "score": 0.0,
            "metadata": {
                "agents_called": agents_called_str,
                "reasoning": "No output",
                "choice": "D",
            },
        }

    return {
        "name": "Routing Accuracy",
        "score": choice_map.get(output.choice, 0.0),
        "metadata": {
            "agents_called": agents_called_str,
            "reasoning": output.reasoning,
            "choice": output.choice,
        },
    }


# Response Quality LLM Judge
response_quality_prompt = """
You are an expert evaluator of AI assistant responses. Your task is to assess the quality, accuracy, and completeness of responses.

User Question: {{input}}
AI Response: {{output}}

Evaluate the response based on:
1. ACCURACY: Is the information provided correct?
2. COMPLETENESS: Does it fully answer the question?
3. CLARITY: Is the response clear and well-structured?
4. RELEVANCE: Does it directly address what was asked?

For math questions, check if the calculation is correct.
For factual questions, assess if the information appears accurate and complete.

Respond with:
EXCELLENT - Response is accurate, complete, clear, and highly relevant
GOOD - Response is mostly accurate and complete with minor issues
FAIR - Response has some accuracy or completeness issues
POOR - Response is inaccurate, incomplete, or irrelevant
"""

response_quality_scorer = LLMClassifier(
    name="Response Quality",
    prompt_template=response_quality_prompt,
    choice_scores={"EXCELLENT": 1.0, "GOOD": 0.75, "FAIR": 0.5, "POOR": 0.0},
    use_cot=True,
    model="gpt-4o",
)


class StepEfficiencyScorer(BaseModel):
    output: list[dict]
    max_steps: int = 8


async def step_efficiency_scorer(output):
    """
    Scores based on the number of steps (messages/tool calls) taken.
    - output: dict containing the 'messages' list.
    - max_steps: maximum reasonable number of steps for full score.
    Returns a score between 0 and 1.
    """
    MAX_STEPS = 8
    messages = output.get("messages", [])
    num_steps = len(messages)
    if num_steps <= MAX_STEPS:
        return 1.0
    # Linearly penalize extra steps
    return max(0.0, 1.0 - (num_steps - MAX_STEPS) / MAX_STEPS)


# Basic evaluation
Eval(
    "langgraph-supervisor",
    data=init_dataset("langgraph-supervisor", "Supervisor Agent Dataset"),
    task=run_supervisor_task,
    scores=[
        response_quality_scorer,
        routing_accuracy_scorer,
        step_efficiency_scorer,
    ],  # type: ignore
    parameters={
        # Prompt parameters
        "system_prompt": SystemPromptParam,
        "research_agent_prompt": ResearchAgentPromptParam,
        "math_agent_prompt": MathAgentPromptParam,
        # Model selection parameters
        "supervisor_model": SupervisorModelParam,
        "research_model": ResearchModelParam,
        "math_model": MathModelParam,
    },
)
