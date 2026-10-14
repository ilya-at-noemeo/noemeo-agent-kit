# noemeo-agent-kit

Shared Lambda/agent **driver** toolkit for Noemeo products with LLM
agents. Generic by design: it must not depend on, or point to, any
product or product-specific repo -- dependencies run the other way.
Extracted once the same boilerplate showed up independently in more
than one product's `lambda_handler.py` files.

## What this is

The plumbing every Lambda-deployed agent needs and none of it should
write twice:

- `token_usage.py` -- `TokenUsage` + `extract_token_usage`: pull real
  token counts off a raw `AIMessage.usage_metadata` (requires
  `include_raw=True` from `with_structured_output`).
- `structured_output.py` -- `invoke_structured`: the `parsing_error`
  re-raise + token-usage-extraction pattern every `evaluate`/`respond`
  node in every agent graph repeats. **Provider-agnostic** -- it operates
  on LangChain's own standardized `with_structured_output(...,
  include_raw=True)` contract, not on anything OpenAI-specific (see the
  module docstring for the caveat on that claim).
- `llm_openai.py` -- `build_structured_llm`: the OpenAI-specific
  `reasoning_effort` vs. `temperature=0` construction switch. (An older
  gpt-5.6 restriction to `reasoning_effort="none"` applied only when
  `with_structured_output` used tool calling; langchain-openai 1.x's
  `json_schema` default accepts any effort -- see the function
  docstring.) The local-development path (OpenAI directly from localhost).
- `llm_cloud.py` -- `build_structured_llm` with the same return
  contract, calling a deployed endpoint that implements the cloud chat
  contract below (activation key, server-side model choice). The
  deployed path, for products whose shipped builds never use a user's
  own provider key.
- `config.py` -- `CapabilityConfig` and small env-var helpers
  (`env_str`, `env_choice`) for reading model/reasoning-effort/detail
  configuration the same way in every driver.
- `prompt_hash.py` -- `hash_prompts`: content-hash a prompt template (or
  set of templates) for traceability, so a saved result can be traced to
  the exact prompt version that produced it without a hand-bumped
  version number.
- `driver.py` -- `parse_image_event`, `run_capability`, `log_invocation`:
  the Lambda-handler shape (decode input, call the agent, time it, emit
  one structured JSON log line, return `model_dump(mode="json")`).

## Cloud chat contract v1

The wire contract `llm_cloud.py` speaks. This library defines it; each
product's own infrastructure implements it server-side and decides
which provider, model, guardrails and quotas sit behind it. Nothing
here knows which product or endpoint it's talking to.

**Versioning:** the version is part of the endpoint path the driver
supplies (e.g. `https://<host>/v1/chat`); the request carries no
version field. A breaking change means a new path (`/v2/chat`) served
alongside `/v1` until clients move, never a changed `/v1`.

Request: `POST <endpoint>` with `Authorization: Bearer <activation key>`
and a JSON body:

```json
{
  "use_case": "<product-defined label>",
  "messages": [ "...OpenAI Chat Completions message format..." ],
  "schema": { "...JSON Schema of the expected output (strict-mode subset, see below)..." },
  "model": "<optional; omit to use the server default>"
}
```

Response (HTTP 200):

```json
{"blocked": true, "reason": "<optional>"}
```

when the server's input guardrail refused the request, otherwise

```json
{"blocked": false, "result": { "...OpenAI Chat Completions response..." }}
```

where `result.choices[0].message.content` is JSON matching `schema`, and
`result.usage` (optional) follows Chat Completions' `usage` shape (token
counts, optional `completion_tokens_details.reasoning_tokens`).

Non-2xx is an error. Its body should be

```json
{"error": "<stable code>", "message": "<safe to show a user>", "request_id": "<id>"}
```

where `error` is a product-defined code a client branches on (e.g.
`quota_exceeded`) and `message` never echoes request content.
`llm_cloud` raises `CloudChatError` (an `httpx.HTTPStatusError`
subclass) carrying `error_code`, `server_message` and `request_id`,
each None when the body doesn't follow this shape.

Rules both sides rely on:

- **`schema` must fit OpenAI structured outputs' strict mode** (servers
  may pass it with `"strict": true`): every property listed in
  `required`, `additionalProperties: false` on every object, and only
  the supported JSON Schema subset. A Pydantic model with no default
  values and `model_config = ConfigDict(extra="forbid")` produces this.
  A server may reject a non-conforming schema with a 4xx.
- **Clients ignore unknown response fields**, so a server can add
  diagnostics (e.g. latency) without a version bump. Adding a request
  field or changing a field's meaning is breaking.
- **`use_case` is opaque here:** product rules (guardrail, limits,
  model) belong in the server's handling of it, not in new contract
  fields.

## What this is *not*

No LangGraph graphs, no prompts, no Pydantic schemas describing what an
agent actually does, no product-specific event/response contracts. Every
agent's actual reasoning, rubric, and domain schema stays in its own
product repo. If you're about to add something here that mentions
photography or any other product concept by name, it belongs
in that product's repo instead -- this library should never need to
change to ship a product feature.

## Using it from a product repo

Poetry git dependency, no private package index needed:

```toml
[tool.poetry.dependencies]
noemeo-agent-kit = { git = "https://github.com/ilya-at-noemeo/noemeo-agent-kit.git", branch = "main" }
```

Pin to a tag instead of `branch = "main"` once more than one product
depends on this and you want to protect one from the other's in-flight
changes.

## Development

`poetry.toml` gives this repo its own isolated `.venv/` (overriding a
shared-environment machine-level poetry default) -- `poetry install`
below only ever installs what this repo actually declares.

```
poetry install
poetry run pytest
```

Editing this repo alongside a consumer that depends on it via the git
reference above: the consumer's own `poetry install` won't see local
edits here until they're pushed. Workaround: in the consumer's
environment, `pip install -e <path-to-noemeo-agent-kit>` (re-run
`poetry install` afterwards to go back to the pinned git version).
