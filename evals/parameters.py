"""Saved Braintrust parameter definitions for evals."""

from typing import Any, cast

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

MATH_AGENT_EVAL_PARAMETERS_NAME = "Math Agent Eval Config"
MATH_AGENT_EVAL_PARAMETERS_SLUG = "math-agent-eval-config"

RESEARCH_AGENT_EVAL_PARAMETERS_NAME = "Research Agent Eval Config"
RESEARCH_AGENT_EVAL_PARAMETERS_SLUG = "research-agent-eval-config"

SYSTEM_PROMPT_PARAM = "system_prompt"
RESEARCH_AGENT_PROMPT_PARAM = "research_agent_prompt"
MATH_AGENT_PROMPT_PARAM = "math_agent_prompt"


def _extract_message_content(message: Any) -> str:
    """Extract text content from a prompt message payload."""
    message_content = (
        message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    )
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        text_parts: list[str] = []
        for part in message_content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        return "\n".join(text_parts)
    return ""


def parse_prompt_param(prompt: Prompt | dict[str, Any]) -> tuple[str, str | None]:
    """Return the instruction text and default model from a Braintrust prompt parameter."""

    if isinstance(prompt, Prompt):
        prompt_name = prompt.name
        prompt_block = prompt.prompt
        options = prompt.options
    elif isinstance(prompt, dict):
        prompt_name = prompt.get("name", "<unnamed>")
        prompt_block = prompt.get("prompt")
        options = prompt.get("options", {})
    else:
        raise TypeError(f"Unsupported prompt parameter type: {type(prompt)!r}")

    if prompt_block is None:
        raise ValueError(f"Prompt parameter '{prompt_name}' is empty")

    prompt_type = (
        prompt_block.get("type") if isinstance(prompt_block, dict) else getattr(prompt_block, "type", None)
    )
    if prompt_type == "completion":
        content = (
            prompt_block.get("content")
            if isinstance(prompt_block, dict)
            else getattr(prompt_block, "content", None)
        )
    else:
        messages = (
            prompt_block.get("messages", [])
            if isinstance(prompt_block, dict)
            else getattr(prompt_block, "messages", None) or []
        )
        if not messages:
            raise ValueError(f"Prompt parameter '{prompt_name}' has no messages")
        content = _extract_message_content(messages[0])

    if not isinstance(content, str):
        raise ValueError(f"Prompt parameter '{prompt_name}' has invalid content")

    model = options.get("model") if isinstance(options, dict) else None
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
                    "messages": [
                        {"role": "system", "content": DEFAULT_MATH_AGENT_PROMPT}
                    ],
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

saved_math_agent_eval_parameters = project.parameters.create(
    name=MATH_AGENT_EVAL_PARAMETERS_NAME,
    slug=MATH_AGENT_EVAL_PARAMETERS_SLUG,
    description="Saved parameter configuration for the math agent eval.",
    schema={
        k: v
        for k, v in SUPERVISOR_EVAL_PARAMETERS.items()
        if k == MATH_AGENT_PROMPT_PARAM
    },
)

saved_research_agent_eval_parameters = project.parameters.create(
    name=RESEARCH_AGENT_EVAL_PARAMETERS_NAME,
    slug=RESEARCH_AGENT_EVAL_PARAMETERS_SLUG,
    description="Saved parameter configuration for the research agent eval.",
    schema={
        k: v
        for k, v in SUPERVISOR_EVAL_PARAMETERS.items()
        if k == RESEARCH_AGENT_PROMPT_PARAM
    },
)
