"""Saved Braintrust parameter definitions for evals."""

from typing import cast

from braintrust import EvalParameters, projects
from braintrust.logger import Prompt
from braintrust.parameters import PromptParameter

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

SYSTEM_PROMPT_PARAM = "system_prompt"
RESEARCH_AGENT_PROMPT_PARAM = "research_agent_prompt"
MATH_AGENT_PROMPT_PARAM = "math_agent_prompt"

def parse_prompt_param(prompt: Prompt) -> tuple[str, str | None]:
    """Return the instruction text and default model from a Braintrust prompt parameter."""

    prompt_block = prompt.prompt
    if prompt_block is None:
        raise ValueError(f"Prompt parameter '{prompt.name}' is empty")

    if getattr(prompt_block, "type", None) == "completion":
        content = prompt_block.content
    else:
        messages = getattr(prompt_block, "messages", None) or []
        if not messages:
            raise ValueError(f"Prompt parameter '{prompt.name}' has no messages")

        message_content = messages[0].content
        if isinstance(message_content, str):
            content = message_content
        else:
            text_parts = []
            for part in message_content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            content = "\n".join(text_parts)

    model = prompt.options.get("model")
    return content, model if isinstance(model, str) else None


SUPERVISOR_EVAL_PARAMETERS: EvalParameters = {
    SYSTEM_PROMPT_PARAM: cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Supervisor system prompt and model.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
                },
                "options": {"model": DEFAULT_SUPERVISOR_MODEL},
            },
        },
    ),
    RESEARCH_AGENT_PROMPT_PARAM: cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Research agent system prompt and model.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [
                        {"role": "system", "content": DEFAULT_RESEARCH_AGENT_PROMPT}
                    ],
                },
                "options": {"model": DEFAULT_RESEARCH_MODEL},
            },
        },
    ),
    MATH_AGENT_PROMPT_PARAM: cast(
        PromptParameter,
        {
            "type": "prompt",
            "description": "Math agent system prompt and model.",
            "default": {
                "prompt": {
                    "type": "chat",
                    "messages": [{"role": "system", "content": DEFAULT_MATH_AGENT_PROMPT}],
                },
                "options": {"model": DEFAULT_MATH_MODEL},
            },
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
