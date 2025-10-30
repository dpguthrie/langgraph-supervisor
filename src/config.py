"""Configuration for the deep agent supervisor and subagents."""

from pydantic import BaseModel, ConfigDict


class AgentConfig(BaseModel):
    """Configuration for the deep agent supervisor and subagents.

    All fields are optional with sensible defaults. None means "use the
    existing default from the agent module".
    """

    # Supervisor/System prompt
    system_prompt: str | None = None

    # Subagent prompts
    research_agent_prompt: str | None = None
    math_agent_prompt: str | None = None

    # Subagent routing descriptions (used by SubAgentMiddleware)
    research_agent_description: str | None = None
    math_agent_description: str | None = None

    # Model selections
    supervisor_model: str = "gpt-4o-mini"
    research_model: str = "gpt-4o-mini"
    math_model: str = "gpt-4o-mini"

    model_config = ConfigDict(arbitrary_types_allowed=True)
