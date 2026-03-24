"""Saved Braintrust parameter definitions for evals."""

from typing import cast

from braintrust import EvalParameters, projects
from braintrust.logger import Prompt
from braintrust.parameters import ModelParameter, PromptParameter

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

def prompt_to_text(prompt: Prompt) -> str:
    """Convert a Braintrust prompt parameter into the instruction string used by agents."""

    prompt_block = prompt.prompt
    if prompt_block is None:
        raise ValueError(f"Prompt parameter '{prompt.name}' is empty")

    if getattr(prompt_block, "type", None) == "completion":
        return prompt_block.content

    messages = getattr(prompt_block, "messages", None) or []
    if not messages:
        raise ValueError(f"Prompt parameter '{prompt.name}' has no messages")

    content = messages[0].content
    if isinstance(content, str):
        return content

    text_parts = []
    for part in content:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    return "\n".join(text_parts)


SUPERVISOR_EVAL_PARAMETERS: EvalParameters = {
    "system_prompt": cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Supervisor system prompt.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
                }
            },
        },
    ),
    "research_agent_prompt": cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Research agent system prompt.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": DEFAULT_RESEARCH_AGENT_PROMPT}
                    ],
                }
            },
        },
    ),
    "math_agent_prompt": cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Math agent system prompt.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "system", "content": DEFAULT_MATH_AGENT_PROMPT}],
                }
            },
        },
    ),
    "supervisor_model": cast(
        ModelParameter,
        {
            "type": "model",
            "default": DEFAULT_SUPERVISOR_MODEL,
            "description": "Model to use for the supervisor agent.",
        },
    ),
    "research_model": cast(
        ModelParameter,
        {
            "type": "model",
            "default": DEFAULT_RESEARCH_MODEL,
            "description": "Model to use for the research agent.",
        },
    ),
    "math_model": cast(
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
