# Making the Playground Stop button work for remote evals

Status: a known gap in Braintrust's remote eval dev servers, tracked by Braintrust. No fix
released as of 2026-09-02.

## The problem

The Playground's **Stop** button does not reach a remote eval server.

For a remote eval task, the browser POSTs **directly** to `{your_endpoint}/eval`, and the
request body deliberately omits the `stop_token` that Braintrust-hosted tasks carry. Stop
therefore only calls `abortController.abort()` on the browser's own fetch — you see
`AbortError: BodyStreamBuffer was aborted` and `Playground eval stream disconnected` in the
console, and nothing else happens.

Server-side, `braintrust/devserver/server.py` launches the run detached:

```python
eval_task = asyncio.create_task(EvalAsync(...))   # not tied to the request
...
return StreamingResponse(event_generator(), media_type="text/event-stream")
```

When the browser disconnects, the ASGI server cancels the response generator. `eval_task`
isn't referenced by it, so the eval runs to completion — and in-flight calls to Anthropic /
OpenAI keep burning tokens for the full run. There is no `/stop` route (`/`, `/list`,
`POST /eval` only), no `request.is_disconnected()` check anywhere in `devserver/`, and the
Python SDK's `EvalAsync` has no cancellation parameter at all. The same gap exists in the
JS, Java, Ruby and `bt` (Rust) dev servers.

Until the SDKs ship a real fix, this repo carries a customer-side workaround.

## The workaround

Three pieces, in [`src/bt_stop.py`](../src/bt_stop.py):

| Piece | Role |
| --- | --- |
| `stop_aware(app)` | ASGI wrapper around `create_app(...)`. Per `POST /eval` it creates a run handle, puts it in a `ContextVar` (inherited by the detached eval task and every row task under it), and trips the stop event when the ASGI server delivers `http.disconnect` — i.e. when the user clicks Stop. Also serves `GET /runs` and `POST /stop` in front of Braintrust's auth middleware. |
| `@stoppable` | Decorates the eval task fn. Fast-fails rows that haven't started, and registers the row's asyncio task so a stop can hard-cancel work already running. |
| `cancellable(aw)` | Races one await against the stop event and cancels it. This is what actually tears down an open provider request mid-generation. |

Wiring is already done — [`src/eval_server.py`](../src/eval_server.py):

```python
app = create_app(evaluators, org_name=None)
app = stop_aware(app)
```

and [`evals/supervisor_common.py`](../evals/supervisor_common.py):

```python
@stoppable
async def run_supervisor_task(input, hooks=None):
    ...
    raise_if_stopped()
    result = await cancellable(supervisor.ainvoke(...))
```

All three degrade to transparent no-ops outside the stop-aware server, so `bt eval` and
local runs are unaffected.

## Out-of-band kill switch

Set `EVAL_STOP_TOKEN` to require a shared secret (unset = no auth check).

```bash
# what's running right now
curl -H "X-Stop-Token: $EVAL_STOP_TOKEN" https://your-server/runs

# stop everything
curl -X POST -H "X-Stop-Token: $EVAL_STOP_TOKEN" -H 'content-type: application/json' \
     -d '{"all": true}' https://your-server/stop

# stop one run, or every run from one playground
curl -X POST ... -d '{"run_id": "1d44daa18868"}' https://your-server/stop
curl -X POST ... -d '{"object_id": "<prompt_session_id>"}' https://your-server/stop
```

## Verified behavior

Against a harness that reproduces the dev server's detached-task shape (6 rows, 2s of
"paid work" each, Stop clicked at 0.4s):

| Scenario | Rows that did paid work | Rows stopped |
| --- | --- | --- |
| No stop support (today's SDK), user clicks Stop | **6** | 0 |
| `stop_aware` + user clicks Stop (disconnect) | **0** | 6 |
| `stop_aware` + `POST /stop` | **0** | 6 |
| `stop_aware`, nobody clicks Stop | 6 | 0 |

## Limits

- `http.disconnect` is only delivered while something reads the ASGI receive channel.
  Starlette's `StreamingResponse` polls it for the life of the SSE stream, which covers the
  run. Behind some proxies and serverless ASGI runtimes (including Modal) disconnect
  delivery is not guaranteed — that is what `POST /stop` is for. Test the disconnect path on
  your own deployment before relying on it.
- Cancelling an in-flight provider call stops the request; tokens already generated before
  the cancel may still be billed.
- Stopped rows appear as errored rows in the experiment. That's accurate — they produced no
  output — but it does mean a stopped run leaves error rows behind.
- Only rows are cancelled, not the parent eval coroutine, so the run still finalizes and
  flushes its summary rather than being killed mid-write.
