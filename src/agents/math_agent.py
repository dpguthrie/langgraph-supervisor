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


def get_math_agent():
    """Create math agent with calculator tools.

    This agent is specialized for:
    - Mathematical calculations
    - Arithmetic operations
    - Addition, subtraction, multiplication, division
    - Numerical computations
    """
    return create_agent(
        model=init_chat_model("openai:gpt-4o-mini"),
        tools=[add, subtract, multiply, divide],
        system_prompt=(
            "You are a math agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
    )
