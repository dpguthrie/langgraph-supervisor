"""Math agent with arithmetic capabilities."""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


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
    system_prompt: str | None = None,
    model: str = "gpt-4o-mini"
):
    """Create math agent with optional custom prompt and model.

    Args:
        system_prompt: Custom system prompt. If None, uses default.
        model: Model name to use (default: gpt-4o-mini)
    """
    # Use provided prompt or fall back to default
    prompt = system_prompt if system_prompt is not None else (
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    )

    tools = [add, subtract, multiply, divide]
    return create_agent(
        model=init_chat_model(f"openai:{model}"),
        tools=tools,
        system_prompt=prompt,
    )
