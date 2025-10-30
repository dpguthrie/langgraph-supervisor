"""Research agent with web search capabilities."""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


def get_research_agent(
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini"
):
    """Create research agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use (default: gpt-4o-mini)

    This agent is specialized for:
    - Web searches and finding information online
    - Looking up current events
    - Researching topics
    - Gathering data from the internet
    """
    # Use provided prompt or fall back to default
    prompt = system_prompt if system_prompt is not None else (
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    web_search = TavilySearch(max_results=3)

    return create_agent(
        model=init_chat_model(f"openai:{model}"),
        tools=[web_search],
        system_prompt=prompt,
    )
