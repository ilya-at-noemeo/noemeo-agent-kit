"""Anthropic-specific structured-output construction -- the second
provider module llm_openai.py's header anticipated. Kept separate rather
than merged into a common builder: the two providers' knobs don't line up
(OpenAI's reasoning_effort "none" has no Anthropic equivalent on every
model; Anthropic splits it into `thinking` and `output_config.effort`),
so callers pass Anthropic's own vocabulary here and any translation from
another provider's settings is the caller's (driver's) business.

Needs the `anthropic` extra (`noemeo-agent-kit[anthropic]`), so products
that only use OpenAI or the cloud contract don't pull in the Anthropic
SDK.
"""

from typing import Any

from langchain_anthropic import ChatAnthropic

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS

# Same as llm_openai's: the SDK's own default, stated explicitly.
DEFAULT_MAX_RETRIES = 2

DEFAULT_MAX_TOKENS = 16_000
"""Generous for a non-streaming structured answer: hitting the cap
truncates the JSON mid-object (a parsing error, then a retry), which
costs more than unused headroom. Thinking tokens count against it too."""


def build_structured_llm(
    output_schema: type,
    *,
    model: str,
    effort: str | None = None,
    thinking: dict[str, Any] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """Construct a ChatAnthropic runnable bound to `output_schema` via
    with_structured_output(..., include_raw=True, method="json_schema").

    method="json_schema" (Claude's structured outputs,
    `output_config.format`) rather than langchain-anthropic's default
    "function_calling": the latter forces a tool call, which current
    models (Sonnet 5.5, Opus 5.5, Fable 5.1) reject with a 400.

    `effort` becomes `output_config.effort` ("low" .. "max"); None
    leaves the model's default. `thinking` is sent as given, e.g.
    {"type": "between_tools"} (Sonnet 5.5's thinking-off setting, effort
    "high" or below only) or {"type": "adaptive"}; None leaves the
    model's default (adaptive on current models). Neither is validated
    here -- accepted values differ per model and change across
    generations, so the API is the authority, as in llm_openai.

    No temperature: current Claude models reject non-default sampling
    parameters, and with thinking on they aren't meaningful anyway.

    Reads ANTHROPIC_API_KEY itself (via the Anthropic SDK), as ChatOpenAI
    reads OPENAI_API_KEY.
    """
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if effort is not None:
        kwargs["effort"] = effort
    if thinking is not None:
        kwargs["thinking"] = thinking
    return ChatAnthropic(**kwargs).with_structured_output(
        output_schema, include_raw=True, method="json_schema"
    )
