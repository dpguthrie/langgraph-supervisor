import modal  # type: ignore
from fastapi import Request  # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage

from src.agent_graph import get_supervisor


modal_image = (
    modal.Image.debian_slim()
    .uv_sync()
    .add_local_python_source("src")
)
app = modal.App("langgraph-supervisor-web", image=modal_image)

# Always read secrets from local .env and send them as a Secret
_secrets = [modal.Secret.from_dotenv()]


def _normalize_content(content):
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p)
    return content


def _serialize_message(message: BaseMessage):
    role = getattr(message, "type", message.__class__.__name__.lower())
    return {"role": role, "content": _normalize_content(message.content)}


@app.function(secrets=_secrets)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
async def chat(
    _request: Request,
    payload: dict,
):
    """HTTP endpoint: {"message": "..."} -> run supervisor and return messages.

    Returns a JSON-serializable transcript of messages.
    """

    user_text = (payload or {}).get("q")
    if not user_text or not str(user_text).strip():
        return {"error": "Missing 'q' in request body"}

    try:
        # Initialize supervisor inside the function so Modal secrets are available
        supervisor = get_supervisor()

        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=str(user_text))]}
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []
        serialized = [
            _serialize_message(m) for m in messages if isinstance(m, BaseMessage)
        ]
        # Fallback: if nothing serialized, return string form
        if not serialized and isinstance(result, dict):
            return {"result": {k: str(v) for k, v in result.items()}}
        return {"messages": serialized}
    except Exception as e:
        return {"error": str(e)}
