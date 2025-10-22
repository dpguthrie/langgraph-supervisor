"""Agent graph module - migrated to deep agent framework.

This module now imports from the new deep agent implementation.
The old supervisor-based implementation has been archived to agent_graph_OLD.py.
"""

# Import the new deep agent implementation
from src.agents.deep_agent import get_supervisor

# Export for backward compatibility
__all__ = ["get_supervisor"]
