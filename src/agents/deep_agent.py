"""Deep agent orchestrator with subagent routing."""

import os
from typing import Any, TypedDict

from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable

from src.agents.math_agent import get_math_agent
from src.agents.research_agent import get_research_agent
from src.agents.state import AgentState
from src.config import (
    AgentConfig,
)


class CompiledSubAgent(TypedDict):
    """Definition of a subagent with routing metadata."""

    name: str
    description: str
    runnable: Runnable


def _get_sub_agents(config: AgentConfig | None = None) -> list[CompiledSubAgent]:
    """Lazily initialize subagents at runtime (not import time).

    This is important for Modal deployment where environment variables
    and secrets aren't available until the function actually runs.

    Args:
        config: Optional configuration for agent prompts, models, and descriptions.
                If None, uses defaults.
    """
    if config is None:
        config = AgentConfig()

    return [
        CompiledSubAgent(
            name="Research Agent",
            description=config.research_agent_description,
            runnable=get_research_agent(
                system_prompt=config.research_agent_prompt,
                model=config.research_model,
            ),
        ),
        CompiledSubAgent(
            name="Math Agent",
            description=config.math_agent_description,
            runnable=get_math_agent(
                system_prompt=config.math_agent_prompt,
                model=config.math_model,
            ),
        ),
    ]


def get_deep_agent(config: AgentConfig | None = None):
    """Create the main deep agent with subagent routing.

    This creates an orchestrator agent that automatically routes user queries
    to specialized subagents based on their descriptions.

    Args:
        config: Optional configuration for agent prompts, models, and descriptions.
                If None, uses defaults from module constants.

    Returns:
        Compiled agent with middleware stack
    """
    if config is None:
        config = AgentConfig()

    # Initialize supervisor model with configured model
    model = init_chat_model(
        model=f"openai:{config.supervisor_model}",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Get subagents with config at runtime (not import time)
    sub_agents = _get_sub_agents(config)

    # Define middleware stack
    deepagent_middleware = [
        SubAgentMiddleware(
            default_model=model,
            default_tools=None,
            subagents=sub_agents,  # type: ignore
            default_interrupt_on=None,
            general_purpose_agent=True,
        ),
    ]

    # Use config system prompt
    system_prompt = config.system_prompt

    # Create agent with state schema
    agent = create_agent(
        model,
        system_prompt=system_prompt,
        tools=None,
        middleware=deepagent_middleware,
        state_schema=AgentState,
    ).with_config({
        "recursion_limit": 25,
        "run_name": "Supervisor Agent"
    })

    return agent


# Cache the agent for reuse
_cached_deep_agent = None


def get_supervisor(config: AgentConfig | None = None, force_rebuild: bool = False):
    """Get the deep agent (maintains compatibility with old API).

    Args:
        config: Optional configuration for agent prompts, models, and descriptions.
                If provided, caching is disabled and a new agent is always built.
                If None, uses cached agent with defaults.
        force_rebuild: If True, rebuild the agent even if cached. Only applies when
                      config is None.

    Returns:
        The cached or newly built deep agent
    """
    global _cached_deep_agent

    # If custom config is provided, always build fresh (no caching)
    if config is not None:
        return get_deep_agent(config)

    # Otherwise use caching for default config
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
