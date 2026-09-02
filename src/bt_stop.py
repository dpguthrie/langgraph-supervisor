"""Cooperative cancellation for Braintrust remote eval servers.

Why this exists
---------------
The Playground's Stop button does not send anything to a remote eval server.
For remote evals the browser POSTs directly to ``{your_endpoint}/eval`` and
Stop only calls ``abortController.abort()`` on that browser fetch. On the
server side, ``braintrust.devserver.server`` launches the run as a detached
task (``asyncio.create_task(EvalAsync(...))``) whose lifetime is not tied to
the request, so the eval keeps running to completion -- and in-flight calls to
Anthropic/OpenAI keep burning tokens -- long after the user clicked Stop.

This module closes that gap on the customer side, without forking the SDK:

1. ``stop_aware(app)`` wraps the ASGI app returned by
   ``braintrust.devserver.server.create_app``. For each ``POST /eval`` it
   creates a run handle, stashes it in a ``ContextVar`` (inherited by the
   detached eval task and every row task beneath it), and trips the handle's
   stop event when the ASGI server delivers ``http.disconnect`` -- i.e. when
   the user clicks Stop. It also serves two extra routes outside Braintrust's
   auth middleware, so you have an out-of-band kill switch:

       GET  /runs   -> list in-flight runs
       POST /stop   -> {"run_id": "..."} | {"object_id": "..."} | {"all": true}

   Both require the ``X-Stop-Token`` header to match ``EVAL_STOP_TOKEN`` when
   that env var is set.

2. ``@stoppable`` decorates your eval task function. It fast-fails rows that
   have not started yet, and registers the row's asyncio task so a stop can
   hard-cancel work already in flight.

3. ``cancellable(awaitable)`` races a single await against the stop event and
   cancels it on stop, which is what actually tears down an open HTTP request
   to a model provider mid-generation.

Notes / limits
--------------
* ``http.disconnect`` is only delivered while something is reading the ASGI
  receive channel. Starlette's ``StreamingResponse`` polls it for the duration
  of the SSE stream, which covers the run itself. Behind some proxies and
  serverless ASGI runtimes disconnect delivery is unreliable -- that is what
  ``POST /stop`` is for.
* Cancelling an in-flight provider call stops the request; tokens already
  generated before the cancel may still be billed.
* Cancelled rows surface as errored rows in the experiment, which is accurate:
  they did not produce an output.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger("bt_stop")

T = TypeVar("T")


class EvalStopped(Exception):
    """Raised inside a task when the run it belongs to has been stopped."""


class RunHandle:
    """Tracks one in-flight ``POST /eval`` so it can be stopped."""

    def __init__(self, run_id: str, *, eval_name: str | None, object_id: str | None):
        self.run_id = run_id
        self.eval_name = eval_name
        self.object_id = object_id
        self.started_at = time.time()
        self.event = asyncio.Event()
        self.reason: str | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def stopped(self) -> bool:
        return self.event.is_set()

    def stop(self, reason: str) -> int:
        """Trip the stop event and hard-cancel any registered row tasks."""
        if not self.event.is_set():
            self.reason = reason
            self.event.set()
        cancelled = 0
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
                cancelled += 1
        logger.warning(
            "stopping run %s (%s): cancelled %d in-flight task(s)",
            self.run_id,
            reason,
            cancelled,
        )
        return cancelled

    def track(self, task: asyncio.Task[Any]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "eval_name": self.eval_name,
            "object_id": self.object_id,
            "age_seconds": round(time.time() - self.started_at, 1),
            "in_flight_rows": sum(1 for t in self._tasks if not t.done()),
            "stopped": self.stopped,
            "reason": self.reason,
        }


_ACTIVE: dict[str, RunHandle] = {}
_current_run: ContextVar[RunHandle | None] = ContextVar("bt_stop_current_run", default=None)


# --------------------------------------------------------------------------
# Task-side API
# --------------------------------------------------------------------------


def current_run() -> RunHandle | None:
    """The run handle for the row currently executing, if any."""
    return _current_run.get()


def is_stopped() -> bool:
    """True if the run this row belongs to has been stopped."""
    handle = _current_run.get()
    return handle is not None and handle.stopped


def raise_if_stopped() -> None:
    """Cooperative checkpoint -- call between steps of a long task."""
    handle = _current_run.get()
    if handle is not None and handle.stopped:
        raise EvalStopped(f"run {handle.run_id} stopped ({handle.reason})")


async def cancellable(awaitable: Awaitable[T]) -> T:
    """Await ``awaitable``, cancelling it if the run is stopped.

    Wrap the calls that actually cost money::

        result = await cancellable(agent.ainvoke(...))
    """
    handle = _current_run.get()
    if handle is None:
        return await awaitable
    if handle.stopped:
        # Don't even start the work.
        if asyncio.isfuture(awaitable) or asyncio.iscoroutine(awaitable):
            with contextlib.suppress(Exception):
                awaitable.close() if asyncio.iscoroutine(awaitable) else awaitable.cancel()
        raise EvalStopped(f"run {handle.run_id} stopped ({handle.reason})")

    work: asyncio.Future[T] = asyncio.ensure_future(awaitable)
    waiter = asyncio.ensure_future(handle.event.wait())
    try:
        done, _ = await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return work.result()
        work.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await work
        raise EvalStopped(f"run {handle.run_id} stopped ({handle.reason})")
    finally:
        if not waiter.done():
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter


def stoppable(task_fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorate an async eval task so the run can be stopped.

    Fast-fails rows that have not started when the stop arrives, and registers
    the row's asyncio task so ``stop()`` can hard-cancel work already running.
    """

    @functools.wraps(task_fn)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        handle = _current_run.get()
        if handle is None:
            # Running under `bt eval` (offline) or without the wrapper -- no-op.
            return await task_fn(*args, **kwargs)
        if handle.stopped:
            raise EvalStopped(f"run {handle.run_id} stopped ({handle.reason})")
        task = asyncio.current_task()
        if task is not None:
            handle.track(task)
        try:
            return await task_fn(*args, **kwargs)
        except asyncio.CancelledError:
            # A hard cancel from stop() -- report it as a stop, not a crash.
            if handle.stopped:
                raise EvalStopped(
                    f"run {handle.run_id} stopped ({handle.reason})"
                ) from None
            raise

    return wrapper


# --------------------------------------------------------------------------
# Server-side API
# --------------------------------------------------------------------------


def stop_all(reason: str = "stop_all") -> int:
    """Stop every in-flight run. Returns the number of runs stopped."""
    stopped = 0
    for handle in list(_ACTIVE.values()):
        if not handle.stopped:
            handle.stop(reason)
            stopped += 1
    return stopped


def _authorized(scope: dict[str, Any]) -> bool:
    expected = os.environ.get("EVAL_STOP_TOKEN")
    if not expected:
        return True
    # Some ASGI runtimes (Modal) hand back bytearray pairs, which are unhashable.
    headers = {bytes(k).lower(): bytes(v) for k, v in scope.get("headers", [])}
    supplied = headers.get(b"x-stop-token", b"").decode("utf-8", "replace")
    return supplied == expected


async def _read_body(receive: Callable[[], Awaitable[dict[str, Any]]]) -> bytes:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"") or b""
        if not message.get("more_body"):
            return body


async def _send_json(
    send: Callable[[dict[str, Any]], Awaitable[None]],
    status: int,
    payload: dict[str, Any],
) -> None:
    data = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(data)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": data})


def stop_aware(app: Any) -> Any:
    """Wrap a Braintrust dev-server ASGI app with stop support.

    Usage::

        app = create_app(evaluators, org_name=None)
        app = stop_aware(app)
    """

    async def wrapped(scope, receive, send):  # noqa: ANN001, ANN202
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return

        path = (scope.get("path") or "/").rstrip("/") or "/"
        method = scope.get("method", "GET").upper()

        # --- out-of-band control plane, in front of Braintrust's auth ---
        if path.endswith("/runs") and method == "GET":
            if not _authorized(scope):
                await _send_json(send, 401, {"error": "bad stop token"})
                return
            await _send_json(
                send, 200, {"runs": [h.as_dict() for h in _ACTIVE.values()]}
            )
            return

        if path.endswith("/stop") and method == "POST":
            if not _authorized(scope):
                await _send_json(send, 401, {"error": "bad stop token"})
                return
            raw = await _read_body(receive)
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                await _send_json(send, 400, {"error": "body must be JSON"})
                return
            if not isinstance(body, dict):
                await _send_json(send, 400, {"error": "body must be a JSON object"})
                return

            targets: list[RunHandle]
            if body.get("all"):
                targets = list(_ACTIVE.values())
            elif body.get("run_id"):
                handle = _ACTIVE.get(str(body["run_id"]))
                targets = [handle] if handle else []
            elif body.get("object_id"):
                wanted = str(body["object_id"])
                targets = [h for h in _ACTIVE.values() if h.object_id == wanted]
            else:
                await _send_json(
                    send,
                    400,
                    {"error": "pass one of: all, run_id, object_id"},
                )
                return

            cancelled = sum(h.stop("stop endpoint") for h in targets)
            await _send_json(
                send,
                200,
                {
                    "stopped": [h.run_id for h in targets],
                    "cancelled_rows": cancelled,
                },
            )
            return

        if not (path.endswith("/eval") and method == "POST"):
            await app(scope, receive, send)
            return

        # --- /eval: buffer the body so we can label the run, then proxy ---
        body = await _read_body(receive)
        eval_name: str | None = None
        object_id: str | None = None
        with contextlib.suppress(Exception):
            parsed = json.loads(body)
            eval_name = parsed.get("name")
            parent = parsed.get("parent") or {}
            if isinstance(parent, dict):
                object_id = parent.get("object_id")
            elif isinstance(parent, str):
                object_id = parent

        handle = RunHandle(
            uuid.uuid4().hex[:12], eval_name=eval_name, object_id=object_id
        )
        _ACTIVE[handle.run_id] = handle
        token = _current_run.set(handle)
        logger.info(
            "run %s started (eval=%s object_id=%s)",
            handle.run_id,
            eval_name,
            object_id,
        )

        replayed = False
        response_complete = False

        async def send_shim(message: dict[str, Any]) -> None:
            nonlocal response_complete
            if message.get("type") == "http.response.body" and not message.get(
                "more_body"
            ):
                response_complete = True
            await send(message)

        async def receive_shim() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            message = await receive()
            if message.get("type") == "http.disconnect" and not response_complete:
                # A disconnect before the response finished is the Playground's
                # Stop button (or a closed tab). A disconnect after it is just
                # the client hanging up on a completed run -- ignore that one.
                handle.stop("client disconnected")
            return message

        try:
            await app(scope, receive_shim, send_shim)
        finally:
            _current_run.reset(token)
            _ACTIVE.pop(handle.run_id, None)
            logger.info(
                "run %s request finished (stopped=%s)", handle.run_id, handle.stopped
            )

    return wrapped
