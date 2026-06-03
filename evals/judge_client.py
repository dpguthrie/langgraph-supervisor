"""OpenAI client for LLM-as-judge scorers (autoevals).

Autoevals defaults to OPENAI_API_KEY when no client is passed. Use this client so
judges authenticate via BRAINTRUST_API_KEY and the Braintrust gateway.
"""

import os

from braintrust.oai import wrap_openai
from openai import AsyncOpenAI

from src.llm import DEFAULT_BRAINTRUST_GATEWAY_URL

judge_client = wrap_openai(
    AsyncOpenAI(
        api_key=os.getenv("BRAINTRUST_API_KEY"),
        base_url=os.getenv("BRAINTRUST_GATEWAY_URL", DEFAULT_BRAINTRUST_GATEWAY_URL),
    )
)
