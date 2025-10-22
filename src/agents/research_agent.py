"""Research agent with web search capabilities."""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


def get_research_agent():
    """Create research agent with web search tool.

    This agent is specialized for:
    - Web searches and finding information online
    - Looking up current events
    - Researching topics
    - Gathering data from the internet
    """
    web_search = TavilySearch(max_results=3)

    return create_agent(
        model=init_chat_model("openai:gpt-4o-mini"),
        tools=[web_search],
        system_prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
    )
