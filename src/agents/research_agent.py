"""Research agent with web search capabilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.config import DEFAULT_RESEARCH_AGENT_PROMPT, DEFAULT_RESEARCH_MODEL
from src.llm import get_gateway_chat_model


@tool
def get_timezone_info(timezone: str, date: str = "") -> str:
    """Deterministic IANA timezone lookup (no web search).

    Args:
        timezone: IANA zone name, e.g. "Europe/Berlin" or "America/New_York".
        date: Optional anchor date YYYY-MM-DD (noon local time); defaults to now.

    Returns the local date/time, UTC offset and DST status on that date.
    """
    try:
        zone = ZoneInfo(timezone)
        if date:
            when = datetime.strptime(date, "%Y-%m-%d").replace(hour=12, tzinfo=zone)
        else:
            when = datetime.now(zone)
    except Exception as e:
        return f"Error: {e}. Use an IANA zone name and YYYY-MM-DD date."
    offset = when.utcoffset()
    hours = offset.total_seconds() / 3600 if offset else 0
    return (
        f"zone={timezone}; local={when.isoformat()}; "
        f"utc_offset={hours:+g}h ({when.strftime('%z')}); "
        f"abbr={when.tzname()}; dst={'yes' if when.dst() else 'no'}"
    )


def get_research_agent(
    system_prompt: str | None = None, model: str = DEFAULT_RESEARCH_MODEL
):
    """Create research agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use.

    This agent is specialized for:
    - Web searches and finding information online
    - Looking up current events
    - Researching topics
    - Gathering data from the internet
    """
    # Use provided prompt or fall back to default
    prompt = (
        system_prompt if system_prompt is not None else DEFAULT_RESEARCH_AGENT_PROMPT
    )

    web_search = TavilySearch(max_results=3)

    return create_agent(
        model=get_gateway_chat_model(model),
        tools=[web_search, get_timezone_info],
        system_prompt=prompt,
        name="ResearchAgent",
    )
