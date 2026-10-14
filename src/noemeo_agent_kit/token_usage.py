"""Real LLM token-usage capture, shared across every agent (in every
product) that calls a model via `with_structured_output(...,
include_raw=True)`.

Extracted here once more than one agent, and then more than one
product, needed the identical logic -- "reusable components, not
duplicated logic per caller," one repo up.
"""

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Real token counts read off the LLM provider's response at call
    time -- never estimated or reconstructed after the fact (that was
    tried once, ad hoc, and isn't good enough to price a product on). A
    field left None means the provider's response didn't report that
    figure -- treat that as "unknown," not as zero, anywhere this feeds
    cost accounting.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    """Populated when the provider bills reasoning separately (e.g.
    gpt-5.6 with reasoning_effort above 'none') -- distinct from
    output_tokens because reasoning tokens are often priced differently.
    """


def extract_token_usage(raw_message) -> TokenUsage:
    """Pull real token counts off the raw AIMessage's usage_metadata.

    Requires the caller to have requested `include_raw=True` from
    `with_structured_output` -- with the default `include_raw=False`,
    `.invoke()` returns only the parsed Pydantic object and this
    information is unrecoverable after the fact.
    """
    usage = raw_message.usage_metadata
    if usage is None:
        return TokenUsage()

    output_details = usage.get("output_token_details") or {}
    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
        reasoning_tokens=output_details.get("reasoning"),
    )
