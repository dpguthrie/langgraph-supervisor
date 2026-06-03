"""Math agent with arithmetic capabilities."""

import ast
import math
import operator
from collections.abc import Callable

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


_ALLOWED_OPERATORS: dict[type[ast.operator | ast.unaryop], Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_numeric_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_numeric_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _eval_numeric_ast(node.left)
        right = _eval_numeric_ast(node.right)
        return float(_ALLOWED_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _eval_numeric_ast(node.operand)
        return float(_ALLOWED_OPERATORS[type(node.op)](operand))
    raise ValueError("expression must contain only numbers and +, -, *, /, **")


def calculate_expression(expression: str) -> float | str:
    """Safely evaluate a numeric arithmetic expression."""
    try:
        parsed = ast.parse(expression.replace("^", "**"), mode="eval")
        return _eval_numeric_ast(parsed)
    except Exception as exc:
        return f"Unsupported numeric expression: {exc}"


def power(base: float, exponent: float) -> float:
    """Raise base to exponent and return the result."""
    return base**exponent


def square_root(value: float) -> float:
    """Return the square root of a non-negative number."""
    if value < 0:
        raise ValueError("square_root requires a non-negative value")
    return math.sqrt(value)


def ascii_value(character: str) -> int:
    """Return the ASCII/Unicode code point for a single character."""
    if len(character) != 1:
        raise ValueError("ascii_value requires exactly one character")
    return ord(character)


def prime_factorization(value: int) -> str:
    """Return the prime factorization of a positive integer in exponential form."""
    if value < 1:
        raise ValueError("prime_factorization requires a positive integer")

    remaining = value
    factor = 2
    factors: list[str] = []
    while factor * factor <= remaining:
        exponent = 0
        while remaining % factor == 0:
            remaining //= factor
            exponent += 1
        if exponent:
            factors.append(f"{factor}^{exponent}")
        factor += 1 if factor == 2 else 2

    if remaining > 1:
        factors.append(f"{remaining}^1")

    return " * ".join(factors) if factors else f"{value}^1"


def solve_sum_product(total: float, product: float) -> list[float]:
    """Find two real numbers with the given sum and product.

    For x + y = total and x * y = product, solve x^2 - total*x + product = 0.
    Returns an empty list when there is no real-valued solution.
    """
    discriminant = total * total - 4 * product
    if discriminant < 0:
        return []

    root = math.sqrt(discriminant)
    first = (total + root) / 2
    second = (total - root) / 2
    return [first, second]


def convert_time_between_offsets(
    hour: int,
    minute: int,
    source_utc_offset: float,
    target_utc_offset: float,
) -> str:
    """Convert a clock time between numeric UTC offsets.

    Offsets are expressed in hours, for example London during BST is +1 and
    Tokyo is +9. Returns a 24-hour HH:MM time string.
    """
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be between 0 and 59")

    total_minutes = int(round((hour + target_utc_offset - source_utc_offset) * 60))
    total_minutes = (total_minutes + minute) % (24 * 60)
    converted_hour, converted_minute = divmod(total_minutes, 60)
    return f"{converted_hour:02d}:{converted_minute:02d}"


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

    tools = [
        add,
        subtract,
        multiply,
        divide,
        calculate_expression,
        power,
        square_root,
        ascii_value,
        prime_factorization,
        solve_sum_product,
        convert_time_between_offsets,
    ]
    return create_agent(
        model=get_gateway_chat_model(model),
        tools=tools,
        system_prompt=prompt,
        name="MathAgent",
    )
