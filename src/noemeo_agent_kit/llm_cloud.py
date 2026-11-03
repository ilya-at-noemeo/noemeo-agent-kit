"""Cloud-backed structured-output construction -- calls a deployed
endpoint implementing the cloud chat contract (README, "Cloud chat
contract v1") instead of talking to a model provider directly. Which
endpoint, and what's behind it, is the calling product's business: each
product's own infra implements the contract server-side and chooses the
model, provider and guardrails. Sibling to llm_openai.py, not a shared
dispatcher with it: per that module's own docstring, two providers
don't yet justify a shared abstraction -- callers pick which module's
build_structured_llm to import.

This module's job is only to shape a contract request and parse the
response back into the same {"raw", "parsed", "parsing_error"} shape
llm_openai.build_structured_llm's return value satisfies, so
structured_output.invoke_structured needs no changes to accept either.

`session_id` (optional) is sent as the `X-Session-Id` header when set: a
UUID the caller picks to group the calls of one session of its own, so
the server can log it (e.g. to sum cost per session). It's opaque to
the server, which scopes it to the authenticated caller, and must never
be derived from content.

`client_version` (optional) is sent as the `X-Client-Version` header
when set: `<client>/<major>.<minor>.<patch>`, the calling application
and its release, which the driver knows and this module doesn't. A
server uses it to decide whether it still supports that client and
may refuse one it doesn't with the `client_outdated` error.

`on_notice` (optional) is called with a `Notice` whenever a response --
a success or an error -- says that what this client calls is deprecated
or will stop being served (`Deprecation`/`Sunset`, README). It's the
driver's, so it can warn its user; an exception it raises is swallowed,
so a bug there never fails the call. It runs on the calling thread
after every such response, so concurrent calls invoke it concurrently
and repeatedly: it must be thread-safe and should deduplicate. `parse_notice` reads the same
headers off any response, for a driver's own calls to the same server.

`CONTRACT_VERSION` is the contract this module speaks, `<major>.<minor>`:
the major is the path's (`/v1`), the minor counts additive revisions.
It's not this package's version, which also covers the other modules.

`model` is forwarded only when set; leave it unset so the server's own
default wins -- translating between model naming schemes (e.g. an
OpenAI-native name vs. a provider-specific one) is the server's job,
not this module's. `reasoning_effort` is accepted for interface parity
and not forwarded.
"""

import email.utils
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
from langchain_core.messages import convert_to_openai_messages

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS

# README, "Cloud chat contract v1": 1.0 the original contract (with the
# reserved client_outdated code), 1.1 the deprecation notice.
CONTRACT_VERSION = "1.1"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Notice:
    """What a server said about the lifetime of what this client calls:
    `deprecated` since when (None if it only announced a sunset),
    `sunset` when it stops being served (None if not yet decided), and
    `link`, where to read about it and get an update."""

    deprecated: datetime | None
    sunset: datetime | None
    link: str | None


def parse_notice(headers: Mapping[str, str]) -> Notice | None:
    """A Notice from a response's `Deprecation` (RFC 9745, `@<unix
    time>`), `Sunset` (RFC 8594, an HTTP date) and `Link` (`rel=
    "sunset"` or `"deprecation"`) headers; None when it carries neither
    of the first two in a readable form. Malformed values are ignored,
    never raised: a notice is advice, not part of the answer. Header
    names are matched in any case, from any mapping."""
    try:
        headers = httpx.Headers(headers)
    except (TypeError, ValueError):  # UnicodeEncodeError: non-ASCII in a hand-built mapping
        return None
    deprecated = _deprecation(headers.get("deprecation"))
    sunset = _http_date(headers.get("sunset"))
    if deprecated is None and sunset is None:
        return None
    return Notice(deprecated=deprecated, sunset=sunset, link=_lifecycle_link(headers.get("link")))


class CloudChatError(httpx.HTTPStatusError):
    """A non-2xx response, with the contract's error envelope parsed
    (README, "Cloud chat contract v1"): `error_code` is the server's
    stable, product-defined code (None if the body didn't carry one),
    `server_message` is safe to show a user, `request_id` correlates
    with the server's logs. Subclasses httpx.HTTPStatusError, so code
    that catches that keeps working."""

    def __init__(self, response: httpx.Response):
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            payload = {}
        self.status_code = response.status_code
        self.error_code: str | None = payload.get("error")
        self.server_message: str | None = payload.get("message")
        self.request_id: str | None = payload.get("request_id")
        super().__init__(
            f"Cloud chat endpoint returned {response.status_code}"
            + (f" ({self.error_code})" if self.error_code else ""),
            request=response.request,
            response=response,
        )


class GuardrailBlockedError(Exception):
    """Raised (via invoke_structured's parsing_error re-raise path) when
    the server's input guardrail intervened before the request ever
    reached a model -- distinct from a malformed/unparseable model
    response, which is a real pydantic.ValidationError instead."""


def build_structured_llm(
    output_schema: type,
    *,
    use_case: str,
    activation_key: str,
    endpoint: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    session_id: str | None = None,
    client_version: str | None = None,
    on_notice: Callable[[Notice], None] | None = None,
):
    """Construct a runnable bound to `output_schema`, invoked the same
    way llm_openai.build_structured_llm's return value is
    (`llm.invoke(messages)` -> {"raw", "parsed", "parsing_error"}).

    `use_case` is an opaque, product-defined label the server maps to
    its own per-use-case policy (guardrail, limits, model); this module
    doesn't interpret it. `activation_key`/`endpoint` identify the
    caller's account and the deployed chat endpoint respectively; both
    required, since there's no local fallback for either once this
    transport is selected. `session_id` is sent as `X-Session-Id` and
    `client_version` as `X-Client-Version` when set (README, "Cloud chat
    contract v1"). `on_notice` receives any deprecation notice a response
    carries.
    """
    return _CloudStructuredLLM(
        output_schema,
        use_case=use_case,
        activation_key=activation_key,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        session_id=session_id,
        client_version=client_version,
        on_notice=on_notice,
    )


class _CloudStructuredLLM:
    def __init__(
        self,
        output_schema,
        *,
        use_case,
        activation_key,
        endpoint,
        model,
        timeout,
        session_id,
        client_version,
        on_notice,
    ):
        self._output_schema = output_schema
        self._schema = output_schema.model_json_schema()
        self._use_case = use_case
        self._activation_key = activation_key
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout
        self._session_id = session_id
        self._client_version = client_version
        self._on_notice = on_notice

    def invoke(self, messages):
        body = {
            "use_case": self._use_case,
            "messages": convert_to_openai_messages(messages),
            "schema": self._schema,
        }
        if self._model is not None:
            body["model"] = self._model

        headers = {"Authorization": f"Bearer {self._activation_key}"}
        if self._session_id is not None:
            headers["X-Session-Id"] = self._session_id
        if self._client_version is not None:
            headers["X-Client-Version"] = self._client_version
        response = httpx.post(
            self._endpoint,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )
        self._report_notice(response)
        if response.is_error:
            raise CloudChatError(response)
        payload = response.json()

        if payload["blocked"]:
            return {
                "raw": None,
                "parsed": None,
                "parsing_error": GuardrailBlockedError(payload.get("reason")),
            }

        result = payload["result"]
        content = result["choices"][0]["message"]["content"]
        raw = SimpleNamespace(usage_metadata=_usage_metadata(result.get("usage")))
        try:
            parsed = self._output_schema.model_validate_json(content)
        except (
            Exception
        ) as exc:  # pydantic.ValidationError, matching with_structured_output's own contract
            return {"raw": raw, "parsed": None, "parsing_error": exc}
        return {"raw": raw, "parsed": parsed, "parsing_error": None}

    def _report_notice(self, response: httpx.Response) -> None:
        if self._on_notice is None:
            return
        try:
            notice = parse_notice(response.headers)
            if notice is not None:
                self._on_notice(notice)
        except Exception:  # noqa: BLE001 -- the caller's bug mustn't fail a paid call
            logger.exception("on_notice failed")


def _deprecation(value: str | None) -> datetime | None:
    # isascii(): isdigit() alone also accepts digits such as "٣".
    if not value or not value.startswith("@") or not (value[1:].isascii() and value[1:].isdigit()):
        return None
    try:
        return datetime.fromtimestamp(int(value[1:]), UTC)
    except (ValueError, OverflowError, OSError):
        return None


def _http_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _lifecycle_link(value: str | None) -> str | None:
    """The first `Link` target whose rel is "sunset" or "deprecation"."""
    for part in (value or "").split(","):
        target, _, params = part.partition(";")
        target = target.strip()
        if not (target.startswith("<") and target.endswith(">")):
            continue
        rels = {
            rel
            for param in params.split(";")
            if param.strip().lower().startswith("rel=")
            for rel in param.split("=", 1)[1].strip().strip('"').lower().split()
        }
        if rels & {"sunset", "deprecation"}:
            return target[1:-1]
    return None


def _usage_metadata(usage: dict | None) -> dict | None:
    """Map an OpenAI Chat Completions `usage` object to the shape
    token_usage.extract_token_usage() reads off `.usage_metadata`."""
    if usage is None:
        return None
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "output_token_details": {"reasoning": reasoning_tokens},
    }
