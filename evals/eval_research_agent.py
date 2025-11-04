"""
Research Agent evaluation - focused on web search and information retrieval.
"""

import re
import sys
from pathlib import Path
from typing import Any

from evals.parameters import ResearchAgentPromptParam, ResearchModelParam  # noqa: E402

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.agents.research_agent import get_research_agent  # noqa: E402

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


async def run_research_task(input: dict, hooks: Any = None) -> dict:
    """Run a research query through the research agent.

    Args:
        input_data: Dict with 'query' field containing the research question

    Returns:
        Dict with 'messages' containing the conversation history
    """
    try:
        # Get research agent
        agent = get_research_agent()

        # Run the agent
        result = await agent.ainvoke({"messages": input["messages"]})

        # Extract messages
        messages = result.get("messages", []) if isinstance(result, dict) else []
        serialized = [serialize_message(m) for m in messages]

        # Track metadata for scoring
        tool_calls = []
        for msg in serialized:
            if "tool_calls" in msg:
                tool_calls.extend([tc["name"] for tc in msg["tool_calls"]])

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "tool_calls": tool_calls,
                    "used_web_search": "tavily_search_results_json" in tool_calls,
                    "total_messages": len(serialized),
                }
            )

        return {"messages": serialized}

    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


# Inline test dataset
RESEARCH_TEST_DATA = [
    {
        "input": {"query": "Who is the current president of France?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "What is the capital of Japan?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "When was the Eiffel Tower built?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
    {
        "input": {"query": "What are the main causes of climate change?"},
        "expected": {
            "should_use_search": True,
            "should_have_url": True,
        },
    },
]


# Custom Scorers


async def web_search_usage_scorer(output, metadata=None):
    """Check if the agent used web search when appropriate."""
    if metadata and metadata.get("used_web_search"):
        return 1.0
    return 0.0


async def source_attribution_scorer(output):
    """Check if the response includes a URL citation."""
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if content and role == "assistant":
            if re.search(r"https?://", content):
                return 1.0
            break
    return 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal tool calls (should use search efficiently)."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    num_searches = tool_calls.count("tavily_search_results_json")

    # Ideal: 1-2 searches, penalize excessive searching
    if num_searches == 1:
        return 1.0
    elif num_searches == 2:
        return 0.9
    elif num_searches <= 4:
        return 0.7
    else:
        return 0.5


# LLM Judge for answer quality
answer_quality_prompt = """
You are evaluating a research agent's response to a factual question.

Question: {{input}}
Response: {{output}}

Evaluate the response on:
1. ACCURACY: Is the information correct and factual?
2. COMPLETENESS: Does it answer the question fully?
3. CLARITY: Is it well-structured and clear?
4. RELEVANCE: Does it address what was asked?

Respond with:
EXCELLENT - Accurate, complete, clear, and highly relevant
GOOD - Mostly accurate and complete with minor issues
FAIR - Some accuracy or completeness issues
POOR - Inaccurate, incomplete, or irrelevant
"""

answer_quality_scorer = LLMClassifier(
    name="Answer Quality",
    prompt_template=answer_quality_prompt,
    choice_scores={"EXCELLENT": 1.0, "GOOD": 0.75, "FAIR": 0.5, "POOR": 0.0},
    use_cot=True,
    model="gpt-4o",
)


# Evaluation
Eval(
    "langgraph-supervisor",
    experiment_name="research-agent",
    data=RESEARCH_TEST_DATA,
    task=run_research_task,
    scores=[
        web_search_usage_scorer,
        source_attribution_scorer,
        efficiency_scorer,
        answer_quality_scorer,
    ],  # type: ignore
    parameters={
        "research_agent_prompt": ResearchAgentPromptParam,
        "research_model": ResearchModelParam,
    },
)
