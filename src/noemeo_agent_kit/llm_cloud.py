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

`model` is forwarded only when set; leave it unset so the server's own
default wins -- translating between model naming schemes (e.g. an
OpenAI-native name vs. a provider-specific one) is the server's job,
not this module's. `reasoning_effort` is accepted for interface parity
and not forwarded.
"""

from types import SimpleNamespace

import httpx
from langchain_core.messages import convert_to_openai_messages

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS


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
):
    """Construct a runnable bound to `output_schema`, invoked the same
    way llm_openai.build_structured_llm's return value is
    (`llm.invoke(messages)` -> {"raw", "parsed", "parsing_error"}).

    `use_case` is an opaque, product-defined label the server maps to
    its own per-use-case policy (guardrail, limits, model); this module
    doesn't interpret it. `activation_key`/`endpoint` identify the
    caller's account and the deployed chat endpoint respectively; both
    required, since there's no local fallback for either once this
    transport is selected.
    """
    return _CloudStructuredLLM(
        output_schema,
        use_case=use_case,
        activation_key=activation_key,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
    )


class _CloudStructuredLLM:
    def __init__(self, output_schema, *, use_case, activation_key, endpoint, model, timeout):
        self._output_schema = output_schema
        self._schema = output_schema.model_json_schema()
        self._use_case = use_case
        self._activation_key = activation_key
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout

    def invoke(self, messages):
        body = {
            "use_case": self._use_case,
            "messages": convert_to_openai_messages(messages),
            "schema": self._schema,
        }
        if self._model is not None:
            body["model"] = self._model

        response = httpx.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self._activation_key}"},
            json=body,
            timeout=self._timeout,
        )
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
