"""Deep agent orchestrator with subagent routing."""

import os
from typing import Any, TypedDict

from braintrust import init_logger
from braintrust_langchain import set_global_handler
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda

from src.agents.math_agent import get_math_agent
from src.agents.research_agent import get_research_agent
from src.agents.state import AgentState
from src.agents.tracing import ImprovedBraintrustCallbackHandler


class CompiledSubAgent(TypedDict):
    """Definition of a subagent with routing metadata."""

    name: str
    description: str
    runnable: Runnable


def create_named_subagent_wrapper(subagent_name: str, subagent: Runnable) -> Runnable:
    """Wrap a subagent to add proper run_name for better Braintrust tracing.

    This adds a run_name and tags to the subagent invocation config, which helps
    Braintrust identify the specific subagent in traces instead of showing generic
    'tools' spans.

    Args:
        subagent_name: Name of the subagent (e.g., "Research Agent")
        subagent: The compiled subagent runnable

    Returns:
        Wrapped runnable with metadata injection
    """

    def invoke_with_name(
        state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        config = config or {}
        config["run_name"] = subagent_name  # type: ignore
        config["tags"] = config.get("tags", []) + [f"subagent:{subagent_name}"]  # type: ignore
        return subagent.invoke(state, config)

    async def ainvoke_with_name(
        state: dict[str, Any], config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        config = config or {}
        config["run_name"] = subagent_name  # type: ignore
        config["tags"] = config.get("tags", []) + [f"subagent:{subagent_name}"]  # type: ignore
        return await subagent.ainvoke(state, config)

    return RunnableLambda(invoke_with_name, ainvoke_with_name)


def _get_sub_agents() -> list[CompiledSubAgent]:
    """Lazily initialize subagents at runtime (not import time).

    This is important for Modal deployment where environment variables
    and secrets aren't available until the function actually runs.
    """
    return [
        CompiledSubAgent(
            name="Research Agent",
            description=(
                "Research agent with web search capabilities. "
                "Use this agent for: web searches, finding information online, "
                "looking up current events, researching topics, gathering data from the internet, "
                "answering questions that require external knowledge or real-time information."
            ),
            runnable=create_named_subagent_wrapper("Research Agent", get_research_agent()),
        ),
        CompiledSubAgent(
            name="Math Agent",
            description=(
                "Math calculation agent with arithmetic tools. "
                "Use this agent for: mathematical calculations, arithmetic operations, "
                "addition, subtraction, multiplication, division, numerical computations, "
                "solving math problems, performing calculations."
            ),
            runnable=create_named_subagent_wrapper("Math Agent", get_math_agent()),
        ),
    ]

SYSTEM_PROMPT = """
You are a helpful AI assistant that can delegate tasks to specialized agents when needed.

You have access to the following specialized agents:
- Research Agent: For web searches and finding information online
- Math Agent: For mathematical calculations and arithmetic

IMPORTANT INSTRUCTIONS:
- For simple greetings, small talk, or general conversational responses, respond directly yourself
- ALWAYS delegate to the Research Agent for:
  * Factual questions about real-world events, people, places, or statistics
  * Questions asking "who", "what", "when", "where" about specific facts
  * Historical records, achievements, or data points
  * ANY question where accurate, verified information is important
  * Questions that could benefit from current or verified information
- ONLY delegate to the Math Agent for queries requiring calculations with specific numbers
- When delegating, assign work to one agent at a time, do not call agents in parallel
- When in doubt about whether to research something, USE THE RESEARCH AGENT - it's better to verify facts than to rely on potentially outdated information
"""


def get_deep_agent():
    """Create the main deep agent with subagent routing.

    This creates an orchestrator agent that automatically routes user queries
    to specialized subagents based on their descriptions.

    Returns:
        Compiled agent with middleware stack
    """
    # Initialize model
    model = init_chat_model(
        model="openai:gpt-4o-mini", api_key=os.environ.get("OPENAI_API_KEY")
    )

    # Get subagents at runtime (not import time)
    sub_agents = _get_sub_agents()

    # Define middleware stack
    deepagent_middleware = [
        SubAgentMiddleware(
            default_model=model,
            default_tools=None,
            subagents=sub_agents,  # type: ignore
            default_middleware=[
                SummarizationMiddleware(
                    model=model,
                    max_tokens_before_summary=100000,
                    messages_to_keep=6,
                ),
            ],
            default_interrupt_on=None,
            general_purpose_agent=True,
        ),
        SummarizationMiddleware(
            model=model,
            max_tokens_before_summary=100000,
            messages_to_keep=6,
        ),
    ]

    # Create agent with state schema
    BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to specialized agents."
    agent = create_agent(
        model,
        system_prompt=SYSTEM_PROMPT + "\n\n" + BASE_AGENT_PROMPT,
        tools=None,
        middleware=deepagent_middleware,
        state_schema=AgentState,
    ).with_config({"recursion_limit": 1000})

    # Initialize tracing
    logger = init_logger(
        project="langgraph-supervisor", api_key=os.environ.get("BRAINTRUST_API_KEY")
    )
    set_global_handler(ImprovedBraintrustCallbackHandler(logger=logger))

    return agent


# Cache the agent for reuse
_cached_deep_agent = None


def get_supervisor(force_rebuild: bool = False):
    """Get the deep agent (maintains compatibility with old API).

    Args:
        force_rebuild: If True, rebuild the agent even if cached

    Returns:
        The cached or newly built deep agent
    """
    global _cached_deep_agent
    if force_rebuild or _cached_deep_agent is None:
        _cached_deep_agent = get_deep_agent()
    return _cached_deep_agent


if __name__ == "__main__":
    print("Initializing deep agent...\n")
    deep_agent = get_deep_agent()

    test_question = "What is 25 * 4?"
    print(f"Question: {test_question}\n")
    print("=" * 80)

    for chunk in deep_agent.stream(
        {
            "messages": [{"role": "user", "content": test_question}],
        }
    ):
        print(chunk)
        print("-" * 80)
