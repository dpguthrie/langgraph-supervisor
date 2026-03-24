"""
Minimal supervisor eval that compares multiple Gateway-routed models.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, init_dataset, init_function  # noqa: E402
from braintrust_langchain import BraintrustCallbackHandler  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.agents.deep_agent import get_supervisor  # noqa: E402
from src.config import AgentConfig  # noqa: E402

load_dotenv()

PROJECT_NAME = "langgraph-supervisor"
DATASET_NAME = "Supervisor Agent Dataset"

GATEWAY_BASE_URL = os.getenv(
    "BRAINTRUST_GATEWAY_BASE_URL", "https://gateway.braintrust.dev/v1"
)
GATEWAY_API_KEY = os.getenv("BRAINTRUST_API_KEY")

ROUTING_SCORER_SLUG = os.getenv(
    "BT_ROUTING_SCORER_SLUG", "routing-correctness-validator-9558"
)
RESPONSE_QUALITY_SCORER_SLUG = os.getenv(
    "BT_RESPONSE_QUALITY_SCORER_SLUG", "response-quality"
)
STEP_EFFICIENCY_SCORER_SLUG = os.getenv(
    "BT_STEP_EFFICIENCY_SCORER_SLUG", "step-efficiency-bundled"
)


def _require_env() -> None:
    if not GATEWAY_API_KEY:
        raise RuntimeError("BRAINTRUST_API_KEY is required")


def _experiment_name(model: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", model).strip("-").lower()
    return f"gateway-matrix-{normalized}"


def _serialize_message(msg: Any) -> dict:
    if hasattr(msg, "content"):
        result = {
            "content": msg.content,
            "role": getattr(msg, "role", getattr(msg, "type", "unknown")),
        }
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            result["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in msg.tool_calls
            ]
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            result["response_metadata"] = msg.response_metadata
        return result
    if isinstance(msg, dict):
        return msg
    return {"content": str(msg)}


def _build_task(model: str):
    async def _task(input: dict, hooks: Any = None) -> dict[str, list]:
        try:
            os.environ["BRAINTRUST_API_KEY"] = GATEWAY_API_KEY or ""
            os.environ["BRAINTRUST_GATEWAY_URL"] = GATEWAY_BASE_URL

            config = AgentConfig(
                supervisor_model=model,
                research_model=model,
                math_model=model,
            )
            supervisor = get_supervisor(config, force_rebuild=True)

            span = (
                hooks.span if hooks and hasattr(hooks, "span") and hooks.span else None
            )
            callback = BraintrustCallbackHandler(logger=span)
            result = await supervisor.ainvoke(
                {"messages": input["messages"]},
                config={"callbacks": [callback]},
            )

            messages = result.get("messages", []) if isinstance(result, dict) else []
            serialized_messages = [_serialize_message(m) for m in messages]

            if hooks and hasattr(hooks, "metadata"):
                hooks.metadata.update(
                    {
                        "gateway_base_url": GATEWAY_BASE_URL,
                        "model": model,
                    }
                )

            return {"messages": serialized_messages}
        except Exception as e:
            if hooks and hasattr(hooks, "metadata"):
                hooks.metadata.update(
                    {
                        "gateway_base_url": GATEWAY_BASE_URL,
                        "model": model,
                        "error": str(e),
                    }
                )
            return {"messages": [{"error": str(e)}]}

    return _task


def _remote_scores() -> list:
    return [
        init_function(project_name=PROJECT_NAME, slug=RESPONSE_QUALITY_SCORER_SLUG),
        init_function(project_name=PROJECT_NAME, slug=ROUTING_SCORER_SLUG),
    ]


DEFAULT_MODEL_MATRIX = [
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.1",
    "google/gemini-2.5-flash",
    "moonshotai/Kimi-K2.5",
]


def _models() -> list[str]:
    configured = os.getenv("GATEWAY_MODELS")
    if not configured:
        return DEFAULT_MODEL_MATRIX
    return [model.strip() for model in configured.split(",") if model.strip()]


def main() -> None:
    _require_env()

    dataset = init_dataset(
        PROJECT_NAME,
        DATASET_NAME,
        _internal_btql={"filter": {"btql": "metadata.key = 'value'"}},
    )

    for model in _models():
        Eval(
            PROJECT_NAME,
            experiment_name=_experiment_name(model),
            data=dataset,
            task=_build_task(model),
            scores=_remote_scores(),
            metadata={
                "gateway_base_url": GATEWAY_BASE_URL,
                "model": model,
            },
        )


if __name__ == "__main__":
    main()
