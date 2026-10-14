from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from noemeo_agent_kit.llm_cloud import CloudChatError, GuardrailBlockedError, build_structured_llm
from noemeo_agent_kit.structured_output import invoke_structured


class _Schema(BaseModel):
    verdict: str


def _mock_response(json_body, status_ok=True, status_code=None):
    status_code = status_code or (200 if status_ok else 500)
    return httpx.Response(
        status_code, json=json_body, request=httpx.Request("POST", "https://example.com/v1/chat")
    )


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_invoke_structured_parses_a_successful_response(mock_post):
    mock_post.return_value = _mock_response(
        {
            "blocked": False,
            "result": {
                "choices": [{"message": {"content": '{"verdict": "keep"}'}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
        }
    )

    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )
    parsed, token_usage = invoke_structured(llm, [])

    assert parsed == _Schema(verdict="keep")
    assert token_usage.input_tokens == 100
    assert token_usage.output_tokens == 20
    assert token_usage.total_tokens == 120
    assert token_usage.reasoning_tokens == 5

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer key123"}
    assert kwargs["json"]["use_case"] == "example_a"
    assert kwargs["json"]["schema"] == _Schema.model_json_schema()
    assert "model" not in kwargs["json"]


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_model_is_included_only_when_explicitly_passed(mock_post):
    mock_post.return_value = _mock_response(
        {
            "blocked": False,
            "result": {
                "choices": [{"message": {"content": '{"verdict": "keep"}'}}],
                "usage": None,
            },
        }
    )

    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        model="gpt-5.6",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "gpt-5.6"


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_guardrail_blocked_response_raises_through_invoke_structured(mock_post):
    mock_post.return_value = _mock_response({"blocked": True, "reason": "guardrail_intervened"})

    llm = build_structured_llm(
        _Schema,
        use_case="example_b",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )

    with pytest.raises(GuardrailBlockedError):
        invoke_structured(llm, [])


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_malformed_content_raises_validation_error_through_invoke_structured(mock_post):
    mock_post.return_value = _mock_response(
        {
            "blocked": False,
            "result": {
                "choices": [{"message": {"content": '{"not_verdict": "oops"}'}}],
                "usage": None,
            },
        }
    )

    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )

    with pytest.raises(Exception):
        invoke_structured(llm, [])


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_non_2xx_response_raises_http_status_error(mock_post):
    mock_post.return_value = _mock_response({}, status_ok=False)

    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )

    with pytest.raises(httpx.HTTPStatusError):
        llm.invoke([])


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_unknown_response_fields_are_ignored(mock_post):
    mock_post.return_value = _mock_response(
        {
            "blocked": False,
            "latency_ms": 1234,
            "result": {"choices": [{"message": {"content": '{"verdict": "keep"}'}}]},
        }
    )

    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )
    parsed, _ = invoke_structured(llm, [])

    assert parsed == _Schema(verdict="keep")


def _llm():
    return build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_error_envelope_is_parsed_into_cloud_chat_error(mock_post):
    mock_post.return_value = _mock_response(
        {"error": "quota_exceeded", "message": "Allowance used.", "request_id": "r-1"},
        status_code=402,
    )
    with pytest.raises(CloudChatError) as excinfo:
        _llm().invoke([])
    error = excinfo.value
    assert isinstance(error, httpx.HTTPStatusError)
    assert (error.status_code, error.error_code) == (402, "quota_exceeded")
    assert (error.server_message, error.request_id) == ("Allowance used.", "r-1")


@pytest.mark.parametrize("body", [None, ["not", "an", "object"]])
@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_error_without_an_envelope_still_raises_cloud_chat_error(mock_post, body):
    mock_post.return_value = httpx.Response(
        403,
        content=b"Forbidden" if body is None else None,
        json=body if body is not None else None,
        request=httpx.Request("POST", "https://example.com/v1/chat"),
    )
    with pytest.raises(CloudChatError) as excinfo:
        _llm().invoke([])
    assert excinfo.value.status_code == 403
    assert excinfo.value.error_code is None
