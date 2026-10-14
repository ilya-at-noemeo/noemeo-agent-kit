# noemeo-agent-kit

Shared, product-neutral library used by Noemeo products' agents and by
their Lambda handlers (`lambda_handler.py` in each agent). Keep that
dual audience in mind -- a change here can affect both a local
LangGraph agent and a packaged Lambda.

Don't name or link any product or product-specific repo from code or
docs here: products depend on this library, never the reverse.

## The one rule that matters most: driver/agent boundary

`src/noemeo_agent_kit/config.py`'s own docstring states it directly:
*"only drivers read the environment. Agents receive a `CapabilityConfig`
(or its individual fields) as plain function parameters and never touch
`os.environ` themselves."* This is load-bearing, not a style preference
— it's what lets a cloud chat endpoint be added as a transport option
without touching any agent's own logic.
When adding any new capability/config knob, it goes through
`env_str`/`env_choice` in `config.py`, read once by the driver
(a product's app or Lambda handler), passed down as a
plain parameter — never read from `os.environ` inside agent code.

## The LLM client seam

There are two paths to an LLM, as sibling modules behind the same
return contract: `llm_cloud.py`'s `build_structured_llm()` (HTTP to an
endpoint implementing the README's "Cloud chat contract v1"; the
server picks provider and model) and `llm_openai.py`'s (`ChatOpenAI`
with an `OPENAI_API_KEY`). Which one a product uses, and when, is the
product driver's choice, not this library's. The contract is defined
here and implemented by product infra; change it here first, versioned,
and keep it product-neutral.
`structured_output.py`'s `invoke_structured()` is provider-agnostic: it
only consumes a `{"raw", "parsed", "parsing_error"}` shape from
`llm.invoke(messages)`, never constructs a client itself, so either
module works with `invoke_structured()` and every caller unchanged.

## Testing pattern

Existing tests patch `noemeo_agent_kit.llm_openai.ChatOpenAI` directly
(`@patch("noemeo_agent_kit.llm_openai.ChatOpenAI")` in both this repo's
own `tests/test_llm_openai.py` and consuming products' agent tests) — there's
no factory/wrapper indirection to patch instead. A new provider module
needs an equivalently patchable, module-level import for its own tests
to hook into the same way.
