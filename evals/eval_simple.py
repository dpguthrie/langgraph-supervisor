"""
Simple evaluation script for the LangGraph supervisor system.
Run this file to execute basic evaluations.
"""

import sys
from pathlib import Path
from typing import Any

# Add the src directory to Python path to resolve imports
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

# Import our supervisor system
from src.app import supervisor  # noqa: E402

load_dotenv()


def run_supervisor_task(input_data: dict, hooks: Any) -> str:
    """Run a single task through the supervisor and return the final response."""
    try:
        # Extract the human message content from the input structure
        messages = input_data.get("messages", [])
        if not messages:
            return "Error: No messages in input"

        # Get the first human message content
        first_message = messages[0]
        user_content = first_message.get("content", "")
        if not user_content:
            return "Error: No content in first message"

        history = [HumanMessage(content=user_content)]

        # Collect all events and extract routing info
        all_messages = []
        agent_used = "unknown"
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
                                    agent_used = "research_agent"
                                elif "math_agent" in node_name:
                                    agent_used = "math_agent"

        # Add metadata for evaluation
        if hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "agent_used": agent_used,
                    "tool_calls": tool_calls,
                    "total_messages": len(all_messages),
                }
            )

        # Return the final supervisor response as a string
        final_response = ""
        for message in reversed(all_messages):
            if (
                hasattr(message, "name")
                and message.name == "supervisor"
                and hasattr(message, "content")
                and message.content
                and not message.content.startswith("Transferring")
            ):
                final_response = message.content
                break

        return final_response or "No response generated"

    except Exception as e:
        if hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return f"Error: {str(e)}"


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


# Basic evaluation
Eval(
    "langgraph-supervisor",
    data=[
        {
            "input": {
                "messages": [
                    {
                        "content": "What is 15 + 27?",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "Calculate 8 * 6",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "Divide 100 by 5",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "Who is the mayor of Denver?",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "What is the capital of Japan?",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "What year was Python created?",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "Multiply 12 by 7",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {
                        "content": "Who wrote the novel 1984?",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        },
    ],
    task=run_supervisor_task,
    scores=[response_quality_scorer, routing_accuracy_scorer],  # type: ignore
)
