"""OpenAI-specific structured-output construction.

Deliberately named llm_openai.py, not llm.py -- more providers are coming
(Anthropic under consideration, AWS Bedrock planned), and each has its
own construction quirks that don't generalize. Once a second provider
module exists, whatever's genuinely common between them gets extracted
then, from two real implementations -- not guessed at now from one. See
structured_output.py for the part of this that already *is*
provider-agnostic (it operates on LangChain's own standardized
with_structured_output contract, not on anything OpenAI-specific).
"""

from langchain_openai import ChatOpenAI

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS

# max_retries=2 is the openai SDK's own default -- stated explicitly here
# rather than left implicit, since a caller reading this signature should
# see the retry behavior they're getting.
DEFAULT_MAX_RETRIES = 2


def build_structured_llm(
    output_schema: type,
    *,
    model: str,
    reasoning_effort: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """Construct a ChatOpenAI runnable bound to `output_schema` via
    with_structured_output(..., include_raw=True) -- include_raw is not
    optional here, structured_output.invoke_structured requires it.

    `reasoning_effort` (e.g. "high", "none") is for reasoning models
    (gpt-5.6-* family and similar) -- those reject `temperature` outright,
    so passing a value here switches the call to reasoning_effort instead
    of temperature=0, rather than sending both and letting the API error.
    Left as a plain str, not a Literal -- accepted values have changed
    across model generations, so validation is left to the API itself
    rather than encoding a possibly-stale set here.

    History: gpt-5.6 used to reject any `reasoning_effort` other than
    "none" (and temperature=0) when with_structured_output went through
    tool calling. langchain-openai 1.x defaults to method="json_schema",
    and with it gpt-5.6 accepts "none", "low", "medium" and
    temperature=0 (checked against the real API 2026-10-04). Which effort
    to use is now a quality/cost/latency choice for the caller to
    measure, not a constraint to work around.

    `timeout` / `max_retries` are passed straight through to ChatOpenAI
    (langchain-openai forwards them to the underlying openai/httpx
    client) -- see config.DEFAULT_LLM_TIMEOUT_SECONDS for why every
    caller gets a bound by default instead of opting in.
    """
    kwargs: dict = {"model": model, "timeout": timeout, "max_retries": max_retries}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs).with_structured_output(output_schema, include_raw=True)
