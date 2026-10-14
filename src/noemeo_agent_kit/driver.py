"""The Lambda-handler shape every capability driver repeats: decode the
event, call the agent, time it, emit one structured JSON log line, return
a JSON-safe response body.

Deliberately thin and deliberately un-opinionated about what "the event"
looks like beyond the image case -- a text-input capability's handler
reads its own event fields directly (there's no generic "the text is
always under this key" contract to enforce) and calls run_capability the
same way an image capability does.
"""

import base64
import json
import time
from typing import Any, Callable

from pydantic import BaseModel

from noemeo_agent_kit.token_usage import TokenUsage


def parse_image_event(event: dict[str, Any]) -> bytes:
    """event["image_base64"] -> decoded bytes. Raises ValueError (not a
    KeyError/TypeError) on a missing/empty field, so a misconfigured
    caller gets a clear message instead of a stack trace pointing at
    base64 internals."""
    image_base64 = event.get("image_base64")
    if not image_base64:
        raise ValueError("event must include 'image_base64'")
    return base64.b64decode(image_base64)


def log_invocation(
    *,
    capability: str,
    model: str,
    prompt_hash: str | None = None,
    token_usage: TokenUsage | None = None,
    latency_ms: float | None = None,
    request_id: str | None = None,
) -> None:
    """One structured JSON line per invocation, to stdout -- CloudWatch
    picks this up automatically from a Lambda's stdout, and a metric
    filter or Embedded Metric Format extraction on `token_usage`/
    `latency_ms` turns this into cost/latency dashboards without a
    separate logging service. Never includes the capability's actual
    result -- this is bookkeeping, not a response mirror."""
    record: dict[str, Any] = {
        "capability": capability,
        "model": model,
        "prompt_hash": prompt_hash,
        "latency_ms": round(latency_ms, 1) if latency_ms is not None else None,
        "request_id": request_id,
    }
    if token_usage is not None:
        record["token_usage"] = token_usage.model_dump(mode="json")
    print(json.dumps(record))


def run_capability(
    fn: Callable[[], tuple[BaseModel, TokenUsage | None]],
    *,
    capability: str,
    model: str,
    prompt_hash: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Call `fn` (a zero-arg closure that runs the actual agent and
    returns (result, token_usage)), log one structured invocation line,
    and return `result.model_dump(mode="json")` -- the Lambda response
    body. `token_usage` may be None for a capability that doesn't call an
    LLM at all; it never appears in the returned body regardless (see
    log_invocation) -- internal cost tracking must not be visible to
    whoever calls the Lambda.

    Exceptions from `fn` propagate uncaught -- a driver failure should
    fail the Lambda invocation loudly (visible in CloudWatch/the caller's
    error path), not be swallowed here.
    """
    start = time.monotonic()
    result, token_usage = fn()
    latency_ms = (time.monotonic() - start) * 1000

    log_invocation(
        capability=capability,
        model=model,
        prompt_hash=prompt_hash,
        token_usage=token_usage,
        latency_ms=latency_ms,
        request_id=request_id,
    )
    return result.model_dump(mode="json")
