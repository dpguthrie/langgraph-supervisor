"""Helpers for constructing LLM clients."""

import os

from langchain_openai import ChatOpenAI

DEFAULT_BRAINTRUST_GATEWAY_URL = "https://gateway.braintrust.dev"


def get_gateway_chat_model(model: str) -> ChatOpenAI:
    """Create a LangChain chat model that routes through the Braintrust gateway."""

    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise ValueError("BRAINTRUST_API_KEY is required to use the Braintrust gateway")

    return ChatOpenAI(
        model=model,
        base_url=os.environ.get("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL),
        api_key=api_key,
    )
