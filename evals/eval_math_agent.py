"""
Math Agent evaluation - focused on calculation accuracy and tool usage.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evals.parameters import MathAgentPromptParam, MathModelParam  # noqa: E402
from src.agents.math_agent import get_math_agent  # noqa: E402

load_dotenv()


def serialize_message(msg: Any) -> dict:
    """Convert a message to JSON-serializable dict."""
    if hasattr(msg, "content"):
        result = {
            "content": msg.content,
            "role": getattr(msg, "role", getattr(msg, "type", "unknown")),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]
        return result
    return msg if isinstance(msg, dict) else {"content": str(msg)}


async def run_math_task(input: dict, hooks: Any = None) -> dict:
    """Run a math calculation through the math agent.

    Args:
        input_data: Dict with 'query' and 'expected_answer' fields

    Returns:
        Dict with 'messages' containing the conversation history
    """
    try:
        # Get math agent
        agent = get_math_agent()

        # Run the agent
        result = await agent.ainvoke({"messages": input["messages"]})

        # Extract messages
        messages = result.get("messages", []) if isinstance(result, dict) else []
        serialized = [serialize_message(m) for m in messages]

        # Track metadata for scoring
        tool_calls = []
        for msg in serialized:
            if "tool_calls" in msg:
                tool_calls.extend(
                    [
                        {"name": tc["name"], "args": tc.get("args", {})}
                        for tc in msg["tool_calls"]
                    ]
                )

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "tool_calls": tool_calls,
                    "total_messages": len(serialized),
                }
            )

        return {"messages": serialized}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


# Inline test dataset with expected answers
MATH_TEST_DATA = [
    {
        "input": {"query": "What is 25 + 17?", "expected_answer": 42},
    },
    {
        "input": {"query": "Calculate 100 - 37", "expected_answer": 63},
    },
    {
        "input": {"query": "What is 12 * 8?", "expected_answer": 96},
    },
    {
        "input": {"query": "Divide 144 by 12", "expected_answer": 12},
    },
    {
        "input": {"query": "What's 15 * 7 + 3?", "expected_answer": 108},
    },
    {
        "input": {"query": "Calculate (50 + 30) / 4", "expected_answer": 20},
    },
]


# Custom Scorers


async def calculation_accuracy_scorer(input, output, expected):
    """Check if the calculated answer matches the expected value."""
    if not expected or "expected_answer" not in expected:
        return 0.5

    expected_answer = expected["expected_answer"]

    # Extract the final assistant message
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""

        if content and role == "assistant":
            # Check if expected answer appears in the response
            if str(expected_answer) in str(content):
                return 1.0
            break

    return 0.0


async def tool_usage_scorer(output, metadata=None):
    """Check if the agent used math tools appropriately."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    if not tool_calls:
        return 0.0

    # Check that math tools were used
    tool_names = [tc["name"] for tc in tool_calls]
    valid_tools = {"add", "subtract", "multiply", "divide"}

    used_valid_tools = any(name in valid_tools for name in tool_names)
    return 1.0 if used_valid_tools else 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal unnecessary tool calls."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    num_calls = len(tool_calls)

    # Ideal: 1-3 tool calls for most operations
    if num_calls <= 2:
        return 1.0
    elif num_calls <= 4:
        return 0.8
    elif num_calls <= 6:
        return 0.6
    else:
        return 0.4


async def response_format_scorer(output):
    """Check if the response is clear and includes the final answer."""
    messages = output.get("messages", [])

    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""

        if content and role == "assistant":
            # Check for clear numerical answer
            import re

            # Look for numbers in the final response
            if re.search(r"\d+", content):
                return 1.0
            break

    return 0.0


# LLM Judge for calculation correctness
calculation_correctness_prompt = """
You are evaluating a math agent's calculation.

Question: {{input}}
Agent's Response: {{output}}
Expected Answer: {{expected}}

Evaluate whether:
1. The calculation is mathematically correct
2. The final answer matches the expected result
3. The reasoning (if shown) is sound

Respond with:
CORRECT - Calculation and answer are correct
INCORRECT - Calculation or answer is wrong
"""

calculation_correctness_scorer = LLMClassifier(
    name="Calculation Correctness",
    prompt_template=calculation_correctness_prompt,
    choice_scores={"CORRECT": 1.0, "INCORRECT": 0.0},
    use_cot=True,
    model="gpt-4o",
)


# Evaluation
Eval(
    "langgraph-supervisor",
    experiment_name="math-agent",
    data=MATH_TEST_DATA,
    task=run_math_task,
    scores=[
        calculation_accuracy_scorer,
        tool_usage_scorer,
        efficiency_scorer,
        response_format_scorer,
        calculation_correctness_scorer,
    ],  # type: ignore
    parameters={
        "math_agent_prompt": MathAgentPromptParam,
        "math_model": MathModelParam,
    },
)
