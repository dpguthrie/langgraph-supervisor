"""Math agent with arithmetic capabilities."""

from langchain.agents import create_agent

from src.config import DEFAULT_MATH_AGENT_PROMPT, DEFAULT_MATH_MODEL
from src.llm import get_gateway_chat_model


def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient. Raises if b is zero."""
    return a / b


def get_math_agent(
    system_prompt: str | None = None, model: str = DEFAULT_MATH_MODEL
):
    """Create math agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use.
    """
    # Use provided prompt or fall back to default
    prompt = system_prompt if system_prompt is not None else DEFAULT_MATH_AGENT_PROMPT

    tools = [add, subtract, multiply, divide]
    return create_agent(
        model=get_gateway_chat_model(model),
        tools=tools,
        system_prompt=prompt,
        name="MathAgent",
    )
