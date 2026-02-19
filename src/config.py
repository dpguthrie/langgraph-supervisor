"""Configuration for the deep agent supervisor and subagents."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

# Default prompts and descriptions
DEFAULT_SYSTEM_PROMPT = f"""
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

IMPORTANT INFORMATION:
- The current date is {datetime.now().strftime("%Y-%m-%d")}.

In order to complete the objective that the user asks of you, you have access to specialized agents.
"""

DEFAULT_RESEARCH_AGENT_DESCRIPTION = (
    "Research agent with web search capabilities. "
    "Use this agent for: web searches, finding information online, "
    "looking up current events, researching topics, gathering data from the internet, "
    "answering questions that require external knowledge or real-time information."
)

DEFAULT_MATH_AGENT_DESCRIPTION = (
    "Math calculation agent with arithmetic tools. "
    "Use this agent for: mathematical calculations, arithmetic operations, "
    "addition, subtraction, multiplication, division, numerical computations, "
    "solving math problems, performing calculations."
)

DEFAULT_RESEARCH_AGENT_PROMPT = (
    "You are a research agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with research-related tasks, DO NOT do any math\n"
    "- Provide links to sources of your information in the response\n"
    "- After you're done with your tasks, respond to the supervisor directly\n"
    "- Respond ONLY with the results of your work, do NOT include ANY other text."
)

DEFAULT_MATH_AGENT_PROMPT = (
    "You are a math agent.\n\n"
    "INSTRUCTIONS:\n"
    "- Assist ONLY with math-related tasks\n"
    "- After you're done with your tasks, respond to the supervisor directly\n"
    "- Respond ONLY with the results of your work, do NOT include ANY other text."
)

# Default model names
DEFAULT_SUPERVISOR_MODEL = "gpt-4o-mini"
DEFAULT_RESEARCH_MODEL = "gpt-4o-mini"
DEFAULT_MATH_MODEL = "gpt-4o-mini"


class AgentConfig(BaseModel):
    """Configuration for the deep agent supervisor and subagents.

    All fields are optional with sensible defaults.
    """

    # Supervisor/System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Subagent prompts
    research_agent_prompt: str = DEFAULT_RESEARCH_AGENT_PROMPT
    math_agent_prompt: str = DEFAULT_MATH_AGENT_PROMPT

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str = DEFAULT_RESEARCH_AGENT_DESCRIPTION
    math_agent_description: str = DEFAULT_MATH_AGENT_DESCRIPTION

    # Model selections
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    research_model: str = DEFAULT_RESEARCH_MODEL
    math_model: str = DEFAULT_MATH_MODEL

    model_config = ConfigDict(arbitrary_types_allowed=True)
