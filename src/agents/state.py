"""Shared state schema for all agents."""

from langchain.agents import AgentState as BaseAgentState


class AgentState(BaseAgentState):
    """Shared state for all agents.

    Extends MessagesState with any custom fields needed.
    Currently just uses the base state (messages only).
    Add custom fields here as needed, e.g.:
        user_id: str | None = None
        session_id: str | None = None
    """
    pass
