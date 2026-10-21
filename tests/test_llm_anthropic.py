from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS
from noemeo_agent_kit.llm_anthropic import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    build_structured_llm,
)


class _Schema:
    pass


class _Answer(BaseModel):
    verdict: str


@patch("noemeo_agent_kit.llm_anthropic.ChatAnthropic")
def test_build_structured_llm_leaves_thinking_and_effort_to_the_model_by_default(mock_chat):
    mock_instance = MagicMock()
    mock_chat.return_value = mock_instance

    build_structured_llm(_Schema, model="claude-sonnet-5-5")

    mock_chat.assert_called_once_with(
        model="claude-sonnet-5-5",
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )


@patch("noemeo_agent_kit.llm_anthropic.ChatAnthropic")
def test_build_structured_llm_uses_json_schema_not_forced_tool_calls(mock_chat):
    # Forced tool_choice (langchain-anthropic's default "function_calling")
    # is a 400 on Sonnet 5.5 / Opus 5.5 / Fable 5.1.
    mock_instance = MagicMock()
    mock_chat.return_value = mock_instance

    build_structured_llm(_Schema, model="claude-sonnet-5-5")

    mock_instance.with_structured_output.assert_called_once_with(
        _Schema, include_raw=True, method="json_schema"
    )


@patch("noemeo_agent_kit.llm_anthropic.ChatAnthropic")
def test_build_structured_llm_passes_effort_and_thinking_through(mock_chat):
    build_structured_llm(
        _Schema, model="claude-sonnet-5-5", effort="high", thinking={"type": "between_tools"}
    )

    _, kwargs = mock_chat.call_args
    assert kwargs["effort"] == "high"
    assert kwargs["thinking"] == {"type": "between_tools"}
    assert "temperature" not in kwargs


@patch("noemeo_agent_kit.llm_anthropic.ChatAnthropic")
def test_build_structured_llm_lets_callers_override_limits(mock_chat):
    build_structured_llm(
        _Schema, model="claude-sonnet-5-5", max_tokens=512, timeout=5, max_retries=0
    )

    _, kwargs = mock_chat.call_args
    assert (kwargs["max_tokens"], kwargs["timeout"], kwargs["max_retries"]) == (512, 5, 0)


def test_real_client_builds_the_request_we_expect(monkeypatch):
    # Unmocked, against the installed langchain-anthropic: the kwargs land
    # where the Messages API expects them. Builds the payload only -- no
    # network call.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    llm = build_structured_llm(
        _Answer, model="claude-sonnet-5-5", effort="low", thinking={"type": "between_tools"}
    )
    raw = llm.first.steps__["raw"]
    payload = raw.bound._get_request_payload([("user", "hi")], **raw.kwargs)

    assert payload["model"] == "claude-sonnet-5-5"
    assert payload["thinking"] == {"type": "between_tools"}
    assert payload["output_config"]["effort"] == "low"
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in payload and "tools" not in payload
