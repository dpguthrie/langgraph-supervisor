"""Saved Braintrust parameter definitions for evals."""

from typing import cast

from braintrust import EvalParameters, projects
from braintrust.parameters import ModelParameter
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

PROJECT_NAME = "langgraph-supervisor"
SUPERVISOR_EVAL_PARAMETERS_NAME = "Supervisor Eval Config"
SUPERVISOR_EVAL_PARAMETERS_SLUG = "supervisor-eval-config"

SUPERVISOR_MODEL_PARAM = "01_supervisor_model"
SYSTEM_PROMPT_PARAM = "02_system_prompt"
RESEARCH_MODEL_PARAM = "03_research_model"
RESEARCH_AGENT_PROMPT_PARAM = "04_research_agent_prompt"
MATH_MODEL_PARAM = "05_math_model"
MATH_AGENT_PROMPT_PARAM = "06_math_agent_prompt"

PARAM_TO_CONFIG_KEY = {
    SUPERVISOR_MODEL_PARAM: "supervisor_model",
    SYSTEM_PROMPT_PARAM: "system_prompt",
    RESEARCH_MODEL_PARAM: "research_model",
    RESEARCH_AGENT_PROMPT_PARAM: "research_agent_prompt",
    MATH_MODEL_PARAM: "math_model",
    MATH_AGENT_PROMPT_PARAM: "math_agent_prompt",
}


class SystemPromptParam(BaseModel):
    value: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="Supervisor system prompt.",
    )


class ResearchAgentPromptParam(BaseModel):
    value: str = Field(
        default=DEFAULT_RESEARCH_AGENT_PROMPT,
        description="Research agent system prompt.",
    )


class MathAgentPromptParam(BaseModel):
    value: str = Field(
        default=DEFAULT_MATH_AGENT_PROMPT,
        description="Math agent system prompt.",
    )


SUPERVISOR_EVAL_PARAMETERS: EvalParameters = {
    SYSTEM_PROMPT_PARAM: SystemPromptParam,
    RESEARCH_AGENT_PROMPT_PARAM: ResearchAgentPromptParam,
    MATH_AGENT_PROMPT_PARAM: MathAgentPromptParam,
    SUPERVISOR_MODEL_PARAM: cast(
        ModelParameter,
        {
            "type": "model",
            "default": DEFAULT_SUPERVISOR_MODEL,
            "description": "Model to use for the supervisor agent.",
        },
    ),
    RESEARCH_MODEL_PARAM: cast(
        ModelParameter,
        {
            "type": "model",
            "default": DEFAULT_RESEARCH_MODEL,
            "description": "Model to use for the research agent.",
        },
    ),
    MATH_MODEL_PARAM: cast(
        ModelParameter,
        {
            "type": "model",
            "default": DEFAULT_MATH_MODEL,
            "description": "Model to use for the math agent.",
        },
    ),
}


project = projects.create(name=PROJECT_NAME)
saved_supervisor_eval_parameters = project.parameters.create(
    name=SUPERVISOR_EVAL_PARAMETERS_NAME,
    slug=SUPERVISOR_EVAL_PARAMETERS_SLUG,
    description="Saved parameter configuration for the supervisor eval.",
    schema=SUPERVISOR_EVAL_PARAMETERS,
)
