"""
Simple evaluation script for the Agent Assistant.
Run this file to execute basic evaluations.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

# Import our supervisor system
from src.config import (  # noqa: E402
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

load_dotenv()


# Parameter definitions for remote evals
class SystemPromptParam(BaseModel):
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="Custom system prompt for the supervisor agent.",
    )


class ResearchAgentPromptParam(BaseModel):
    research_agent_prompt: str = Field(
        default=DEFAULT_RESEARCH_AGENT_PROMPT,
        description="Custom system prompt for the research agent.",
    )


class MathAgentPromptParam(BaseModel):
    math_agent_prompt: str = Field(
        default=DEFAULT_MATH_AGENT_PROMPT,
        description="Custom system prompt for the math agent.",
    )


class SupervisorModelParam(BaseModel):
    supervisor_model: str = Field(
        default=DEFAULT_SUPERVISOR_MODEL,
        description="Model to use for the supervisor agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class ResearchModelParam(BaseModel):
    research_model: str = Field(
        default=DEFAULT_RESEARCH_MODEL,
        description="Model to use for the research agent (e.g., gpt-4o-mini, gpt-4o).",
    )


class MathModelParam(BaseModel):
    math_model: str = Field(
        default=DEFAULT_MATH_MODEL,
        description="Model to use for the math agent (e.g., gpt-4o-mini, gpt-4o).",
    )
