from unittest.mock import MagicMock, patch

from noemeo_agent_kit.config import DEFAULT_LLM_TIMEOUT_SECONDS
from noemeo_agent_kit.llm_openai import (
    DEFAULT_MAX_RETRIES,
    build_structured_llm,
)


class _Schema:
    pass


@patch("noemeo_agent_kit.llm_openai.ChatOpenAI")
def test_build_structured_llm_uses_temperature_when_no_reasoning_effort(mock_chat_openai):
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    build_structured_llm(_Schema, model="gpt-4o")

    mock_chat_openai.assert_called_once_with(
        model="gpt-4o",
        temperature=0,
        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    mock_instance.with_structured_output.assert_called_once_with(_Schema, include_raw=True)


@patch("noemeo_agent_kit.llm_openai.ChatOpenAI")
def test_build_structured_llm_uses_reasoning_effort_when_given(mock_chat_openai):
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance

    build_structured_llm(_Schema, model="gpt-5.6", reasoning_effort="none")

    mock_chat_openai.assert_called_once_with(
        model="gpt-5.6",
        reasoning_effort="none",
        timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
    )
    mock_instance.with_structured_output.assert_called_once_with(_Schema, include_raw=True)


@patch("noemeo_agent_kit.llm_openai.ChatOpenAI")
def test_build_structured_llm_every_call_gets_a_timeout_by_default(mock_chat_openai):
    # The regression this guards: before DEFAULT_LLM_TIMEOUT_SECONDS existed,
    # a stalled API response hung the calling request forever with no
    # error anywhere -- see the module-level comment on
    # DEFAULT_LLM_TIMEOUT_SECONDS for the real incident that surfaced this.
    build_structured_llm(_Schema, model="gpt-4o")

    _, kwargs = mock_chat_openai.call_args
    assert kwargs["timeout"] == DEFAULT_LLM_TIMEOUT_SECONDS
    assert kwargs["max_retries"] == DEFAULT_MAX_RETRIES


@patch("noemeo_agent_kit.llm_openai.ChatOpenAI")
def test_build_structured_llm_lets_callers_override_timeout_and_retries(mock_chat_openai):
    build_structured_llm(_Schema, model="gpt-4o", timeout=5, max_retries=0)

    _, kwargs = mock_chat_openai.call_args
    assert kwargs["timeout"] == 5
    assert kwargs["max_retries"] == 0
