import hmac
import os

import modal  # type: ignore
from fastapi import Depends, HTTPException, Request, status  # type: ignore
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage

from src.agent_graph import get_supervisor

# Build or fetch the cached supervisor graph
supervisor = get_supervisor()

auth_scheme = HTTPBearer()


modal_image = (
    modal.Image.debian_slim()
    .uv_pip_install(requirements=["requirements.txt"])
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
@modal.fastapi_endpoint(method="POST")
async def chat(
    request: Request,
    payload: dict,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """HTTP endpoint: {"message": "..."} -> run supervisor and return messages.

    Returns a JSON-serializable transcript of messages.
    """
    # Simple Bearer token using constant-time comparison
    expected_token = os.environ.get("ENDPOINT_AUTH_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server not configured",
        )
    if not token or not hmac.compare_digest(token.credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_text = (payload or {}).get("q")
    if not user_text or not str(user_text).strip():
        return {"error": "Missing 'q' in request body"}

    try:
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
