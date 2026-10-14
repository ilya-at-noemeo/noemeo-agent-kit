"""Provider-agnostic handling of a with_structured_output(...,
include_raw=True) result: check for a parsing error, extract token usage,
return the parsed object.

This is genuinely provider-agnostic, not just conveniently reused across
one provider so far: `with_structured_output` and its `{"raw", "parsed",
"parsing_error"}` return shape are part of LangChain's own BaseChatModel
interface, implemented consistently by ChatOpenAI, ChatAnthropic, and
ChatBedrock (each either supports structured output natively or falls
back to a parsing wrapper -- either way the *shape* returned to the
caller is the same). Likewise `AIMessage.usage_metadata` is a
LangChain-standardized field populated by every major integration, not
something specific to OpenAI's response format.

That said, "implemented consistently" is a claim about today's LangChain
integrations, not a law of nature -- when Anthropic or Bedrock is actually
wired up (see llm_openai.py's header), verify usage_metadata actually
populates the same way there before trusting cost accounting on it.
"""

from noemeo_agent_kit.token_usage import TokenUsage, extract_token_usage


def invoke_structured(llm, messages) -> tuple[object, TokenUsage]:
    """Invoke a with_structured_output(..., include_raw=True) runnable
    (from any provider's construction helper, e.g. llm_openai's
    build_structured_llm) and return (parsed, token_usage). Re-raises the
    parsing error explicitly -- include_raw=True otherwise suppresses the
    exception with_structured_output would raise on a malformed response,
    which would silently hide a bad LLM output instead of failing loudly.
    """
    output = llm.invoke(messages)
    if output["parsing_error"] is not None:
        raise output["parsing_error"]
    return output["parsed"], extract_token_usage(output["raw"])
