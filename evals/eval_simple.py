"""
Simple evaluation script for the Agent Assistant.
Run this file to execute basic evaluations.
"""

import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval, init_dataset  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# Import our supervisor system
from src.agents.deep_agent import get_supervisor  # noqa: E402
from src.config import AgentConfig  # noqa: E402

load_dotenv()


# Parameter definitions for remote evals
class SystemPromptParam(BaseModel):
    system_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for the supervisor agent. If None, uses the default prompt.",
    )


class ResearchAgentPromptParam(BaseModel):
    research_agent_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for the research agent. If None, uses the default prompt.",
    )


class MathAgentPromptParam(BaseModel):
    math_agent_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for the math agent. If None, uses the default prompt.",
    )


class ResearchAgentDescriptionParam(BaseModel):
    research_agent_description: str | None = Field(
        default=None,
        description="Custom routing description for the research agent. Used by the supervisor to decide when to route to this agent.",
    )


class MathAgentDescriptionParam(BaseModel):
    math_agent_description: str | None = Field(
        default=None,
        description="Custom routing description for the math agent. Used by the supervisor to decide when to route to this agent.",
    )


class SupervisorModelParam(BaseModel):
    supervisor_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for the supervisor agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class ResearchModelParam(BaseModel):
    research_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for the research agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class MathModelParam(BaseModel):
    math_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for the math agent (e.g., gpt-4o-mini, gpt-4o).",
    )


def unwrap_parameters(params: dict) -> dict:
    """Unwrap Pydantic parameter models to extract actual field values.

    Braintrust wraps each parameter in its Pydantic model instance. This function
    extracts the actual field values from those models.

    Args:
        params: Dict of parameter names to Pydantic model instances

    Returns:
        Dict of parameter names to unwrapped field values (filters out None)

    Example:
        Input:  {"system_prompt": SystemPromptParam(system_prompt="Hello")}
        Output: {"system_prompt": "Hello"}
    """
    config_params = {}
    for key, value in params.items():
        # Check if this is a Pydantic model instance that needs unwrapping
        if isinstance(value, BaseModel):
            # Get the actual field value from the model
            # The field name matches the key name in our parameter definitions
            field_value = getattr(value, key, None)
            if field_value is not None:
                config_params[key] = field_value
        elif value is not None:
            # Direct value (shouldn't happen with our setup, but handle it)
            config_params[key] = value
    return config_params


def run_supervisor_task(input_data: dict, hooks: Any = None) -> dict[str, list]:
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

        # Unwrap Pydantic parameter models to extract actual field values
        config_params = unwrap_parameters(params)
        config = AgentConfig(**config_params) if config_params else None

        # Get supervisor with config (or default if config is None)
        supervisor = get_supervisor(config)

        # Extract the human message content from the input structure
        messages = input_data.get("messages", [])
        if not messages:
            return {"messages": [{"error": "No messages in input"}]}

        # Get the first human message content
        first_message = messages[0]
        user_content = first_message.get("content", "")
        if not user_content:
            return {"messages": [{"error": "No content in first message"}]}

        history = [{"role": "user", "content": user_content}]

        # Collect all events and extract routing info
        all_messages = []
        agent_used = []
        tool_calls = []

        for event in supervisor.stream({"messages": history}):
            # Event structure: {"node_name": {"messages": [...]}}
            for node_name, node_update in event.items():
                if "messages" in node_update and node_update["messages"]:
                    messages = node_update["messages"]
                    all_messages.extend(messages)

                    # Extract tool calls for routing analysis
                    for msg in messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_calls.append(
                                    {
                                        "node": node_name,
                                        "tool_name": tool_call["name"],
                                    }
                                )

                                # Determine which agent was used
                                if "research_agent" in node_name:
                                    agent_used.append("research_agent")
                                elif "math_agent" in node_name:
                                    agent_used.append("math_agent")

        # Add metadata for evaluation
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "agent_used": agent_used,
                    "tool_calls": tool_calls,
                    "total_messages": len(all_messages),
                }
            )

        return {"messages": all_messages}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


# LLM-as-a-Judge Scoring functions

# Routing Accuracy LLM Judge
routing_accuracy_prompt = """
You are an expert evaluator of AI agent routing systems. Your task is to determine whether a user question was correctly routed to the appropriate agent.

The system has two agents:
1. MATH_AGENT: Should handle mathematical calculations, arithmetic, equations, and numerical problems
2. RESEARCH_AGENT: Should handle factual questions, information lookup, current events, geography, history, etc.

Question: {{input}}
Agent Used: {{metadata.agent_used}}

Evaluate whether the question was routed to the correct agent. Consider:
- Math questions (calculations, arithmetic, "what is X + Y") should go to MATH_AGENT
- Factual/research questions (who, what, where, when questions about real-world information) should go to RESEARCH_AGENT

Respond with:
CORRECT - if the routing was appropriate
INCORRECT - if the routing was wrong
"""

routing_accuracy_scorer = LLMClassifier(
    name="Routing Accuracy",
    prompt_template=routing_accuracy_prompt,
    choice_scores={"CORRECT": 1, "INCORRECT": 0},
    use_cot=True,
    model="gpt-4o",
)

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


class SourceAttributionScorer(BaseModel):
    output: list[dict]


async def source_attribution_scorer(output):
    """
    Checks if the final answer includes a credible source (URL).
    Returns 1.0 if a URL is present, else 0.0.
    """
    messages = output.get("messages", [])
    # Find the last non-empty content message from an AI
    for msg in reversed(messages):
        content = msg.content
        if content and msg.role == "assistant":
            if re.search(r"https?://", content):
                return 1.0
            break
    return 0.0


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
    parameters={
        # Prompt parameters
        "system_prompt": SystemPromptParam,
        "research_agent_prompt": ResearchAgentPromptParam,
        "math_agent_prompt": MathAgentPromptParam,
        # Routing description parameters
        "research_agent_description": ResearchAgentDescriptionParam,
        "math_agent_description": MathAgentDescriptionParam,
        # Model selection parameters
        "supervisor_model": SupervisorModelParam,
        "research_model": ResearchModelParam,
        "math_model": MathModelParam,
    },
)
