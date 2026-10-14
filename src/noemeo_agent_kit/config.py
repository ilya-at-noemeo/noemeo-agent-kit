"""Driver-level configuration: reading model / reasoning-effort / detail
knobs from the environment the same way in every Lambda handler, across
every product.

Per the driver/agent boundary this library assumes (see README.md and
CLAUDE.md): only drivers read the environment.
Agents receive a `CapabilityConfig` (or its individual fields) as plain
function parameters and never touch os.environ themselves.
"""

import os
from dataclasses import dataclass

DEFAULT_REASONING_EFFORT = "none"

DEFAULT_LLM_TIMEOUT_SECONDS = 120
"""Shared default for every LLM client (cloud and OpenAI). 120s rather
than something tighter: gpt-5.6 at reasoning_effort above "none" can
legitimately take a while, and a false-positive timeout is worse than
waiting for a call that was never going to fail on its own."""
"""Confirmed-safe default for gpt-5.6 combined with structured/tool
output -- see llm.py's build_structured_llm docstring for the full
gotcha. This is a reasonable fallback for *reasoning models generally*,
not a guarantee for every model/version; a caller targeting a different
model should pass its own default rather than assume this one."""


@dataclass(frozen=True)
class CapabilityConfig:
    """The three knobs every image/text capability driver exposes.
    `detail` is only meaningful for vision calls -- leave it None for a
    text-only capability."""

    model: str
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT
    detail: str | None = None


def env_str(name: str, default: str) -> str:
    """Read a plain string env var, falling back to `default` if unset."""
    return os.environ.get(name, default)


def env_choice(name: str, default: str, valid: tuple[str, ...]) -> str:
    """Read an env var constrained to a fixed set of values -- raises
    ValueError immediately (at driver startup, not deep inside an agent
    call) if the operator set it to something invalid."""
    value = os.environ.get(name, default)
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
    return value
