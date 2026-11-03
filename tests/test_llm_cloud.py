import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from noemeo_agent_kit import llm_cloud
from noemeo_agent_kit.llm_cloud import (
    CONTRACT_VERSION,
    CloudChatError,
    GuardrailBlockedError,
    Notice,
    _deprecation,
    build_structured_llm,
    parse_notice,
)
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


def _ok_body():
    return {
        "blocked": False,
        "result": {"choices": [{"message": {"content": '{"verdict": "keep"}'}}], "usage": None},
    }


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_session_id_is_sent_as_the_x_session_id_header(mock_post):
    mock_post.return_value = _mock_response(_ok_body())
    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        session_id="0b9e8f2c-3d41-4a7e-9c55-6f1d2a8b7e03",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {
        "Authorization": "Bearer key123",
        "X-Session-Id": "0b9e8f2c-3d41-4a7e-9c55-6f1d2a8b7e03",
    }


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_no_x_session_id_header_is_sent_without_a_session_id(mock_post):
    mock_post.return_value = _mock_response(_ok_body())
    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"Authorization": "Bearer key123"}


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_client_version_is_sent_as_the_x_client_version_header(mock_post):
    mock_post.return_value = _mock_response(_ok_body())
    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        client_version="example-app/1.4.2",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {
        "Authorization": "Bearer key123",
        "X-Client-Version": "example-app/1.4.2",
    }


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_client_version_and_session_id_headers_are_sent_together(mock_post):
    mock_post.return_value = _mock_response(_ok_body())
    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        session_id="0b9e8f2c-3d41-4a7e-9c55-6f1d2a8b7e03",
        client_version="example-app/1.4.2",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {
        "Authorization": "Bearer key123",
        "X-Session-Id": "0b9e8f2c-3d41-4a7e-9c55-6f1d2a8b7e03",
        "X-Client-Version": "example-app/1.4.2",
    }


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_no_x_client_version_header_is_sent_without_a_client_version(mock_post):
    mock_post.return_value = _mock_response(_ok_body())
    llm = build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        session_id="0b9e8f2c-3d41-4a7e-9c55-6f1d2a8b7e03",
    )
    invoke_structured(llm, [])

    _, kwargs = mock_post.call_args
    assert "X-Client-Version" not in kwargs["headers"]


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


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_client_outdated_response_raises_cloud_chat_error(mock_post):
    mock_post.return_value = _mock_response(
        {
            "error": "client_outdated",
            "message": "Please update to continue.",
            "request_id": "r-426",
        },
        status_code=426,
    )
    with pytest.raises(CloudChatError) as excinfo:
        _llm().invoke([])
    error = excinfo.value
    assert (error.status_code, error.error_code) == (426, "client_outdated")
    assert (error.server_message, error.request_id) == ("Please update to continue.", "r-426")


# --- contract lifecycle: version, notices -------------------------------

_DEPRECATED_AT = datetime(2026, 1, 1, tzinfo=UTC)  # unix 1767225600
_SUNSET_AT = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
_SUNSET_HEADER = "Thu, 31 Dec 2026 23:59:59 GMT"
_LINK = "https://example.com/migrate"


def _parse(headers):
    return parse_notice(httpx.Headers(headers))


def test_contract_version_is_major_dot_minor():
    assert re.fullmatch(r"\d+\.\d+", CONTRACT_VERSION)


def test_readme_states_the_contract_version_the_module_exports():
    readme = (Path(__file__).parent.parent / "README.md").read_text()
    assert f"currently **{CONTRACT_VERSION}**" in readme


def test_notice_with_both_headers():
    notice = _parse({"Deprecation": "@1767225600", "Sunset": _SUNSET_HEADER})
    assert notice == Notice(deprecated=_DEPRECATED_AT, sunset=_SUNSET_AT, link=None)


def test_notice_with_only_deprecation():
    notice = _parse({"Deprecation": "@1767225600"})
    assert notice == Notice(deprecated=_DEPRECATED_AT, sunset=None, link=None)


def test_notice_with_only_sunset():
    notice = _parse({"Sunset": _SUNSET_HEADER})
    assert notice == Notice(deprecated=None, sunset=_SUNSET_AT, link=None)
    assert notice.sunset.utcoffset().total_seconds() == 0


def test_sunset_date_without_a_zone_is_read_as_utc():
    notice = _parse({"Sunset": "Thu, 31 Dec 2026 23:59:59 -0000"})
    assert notice.sunset == _SUNSET_AT
    assert notice.sunset.tzinfo is not None


def test_no_notice_without_deprecation_or_sunset():
    assert _parse({}) is None
    assert _parse({"Link": f'<{_LINK}>; rel="sunset"'}) is None


@pytest.mark.parametrize("bad", ["1767225600", "@soon", "@", "", "@99999999999999999999999"])
def test_malformed_deprecation_is_ignored(bad):
    assert _parse({"Deprecation": bad}) is None
    notice = _parse({"Deprecation": bad, "Sunset": _SUNSET_HEADER})
    assert notice == Notice(deprecated=None, sunset=_SUNSET_AT, link=None)


@pytest.mark.parametrize("bad", ["tomorrow", "", "32 Foo 2026 99:99:99 GMT"])
def test_malformed_sunset_is_ignored(bad):
    assert _parse({"Sunset": bad}) is None
    notice = _parse({"Deprecation": "@1767225600", "Sunset": bad})
    assert notice == Notice(deprecated=_DEPRECATED_AT, sunset=None, link=None)


def _link_of(link_header):
    return _parse({"Deprecation": "@1767225600", "Link": link_header}).link


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (f'<{_LINK}>; rel="sunset"', _LINK),
        (f'<{_LINK}>; rel="deprecation"', _LINK),
        (f'<{_LINK}>; rel="sunset deprecation"', _LINK),
        (f'<{_LINK}>; rel="deprecation sunset"', _LINK),
        (f"<{_LINK}>; rel=sunset", _LINK),
        (f'<{_LINK}>; REL="Sunset"', _LINK),
        (f'<https://example.com/other>; rel="next", <{_LINK}>; rel="sunset"', _LINK),
        (
            '<https://example.com/a>; rel="deprecation", <https://example.com/b>; rel="sunset"',
            "https://example.com/a",
        ),
        ('<https://example.com/other>; rel="next"', None),
        ('<https://example.com/other>; rel="sunsets"', None),
        ("garbage", None),
        ("<unterminated; rel=sunset", None),
        ('; rel="sunset"', None),
        ("", None),
    ],
)
def test_link_is_the_first_sunset_or_deprecation_target(header, expected):
    assert _link_of(header) == expected


def test_notice_headers_are_case_insensitive_through_httpx_headers():
    headers = httpx.Headers(
        {"DEPRECATION": "@1767225600", "sunset": _SUNSET_HEADER, "LiNk": f'<{_LINK}>; rel="sunset"'}
    )
    assert parse_notice(headers) == Notice(deprecated=_DEPRECATED_AT, sunset=_SUNSET_AT, link=_LINK)


@pytest.mark.parametrize(
    "names",
    [
        ("deprecation", "sunset", "link"),
        ("Deprecation", "Sunset", "Link"),
        ("DEPRECATION", "SUNSET", "LINK"),
    ],
)
def test_parse_notice_matches_header_names_in_any_case_from_a_plain_dict(names):
    plain = dict(
        zip(
            names,
            ["@1767225600", _SUNSET_HEADER, f'<{_LINK}>; rel="sunset"'],
            strict=True,
        )
    )
    assert parse_notice(plain) == Notice(deprecated=_DEPRECATED_AT, sunset=_SUNSET_AT, link=_LINK)


@pytest.mark.parametrize(
    "headers",
    [
        {"Deprecation": "@\u0663"},
        {"Sunset": "caf\u00e9"},
        {"Link": "<\u00e9>", "Deprecation": "@\u0663"},
    ],
)
def test_parse_notice_returns_none_for_a_mapping_httpx_rejects(headers):
    assert parse_notice(headers) is None


@pytest.mark.parametrize(
    "value", ["@\u0663", "@\u0661\u0662", "@1_000", "@ 5", "@5 ", "@-5", "@+5"]
)
def test_deprecation_rejects_non_ascii_digits_and_other_non_digit_forms(value):
    assert _deprecation(value) is None


def test_deprecation_accepts_plain_ascii_digits():
    assert _deprecation("@1767225600") == _DEPRECATED_AT


def _notice_response(body, status_code=200, headers=None):
    return httpx.Response(
        status_code,
        json=body,
        headers=headers,
        request=httpx.Request("POST", "https://example.com/v1/chat"),
    )


_NOTICE_HEADERS = {
    "Deprecation": "@1767225600",
    "Sunset": _SUNSET_HEADER,
    "Link": f"<{_LINK}>; rel=sunset",
}
_EXPECTED_NOTICE = Notice(deprecated=_DEPRECATED_AT, sunset=_SUNSET_AT, link=_LINK)


def _llm_with(on_notice):
    return build_structured_llm(
        _Schema,
        use_case="example_a",
        activation_key="key123",
        endpoint="https://example.com/v1/chat",
        on_notice=on_notice,
    )


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_on_notice_is_called_with_the_notice_on_a_successful_response(mock_post):
    mock_post.return_value = _notice_response(_ok_body(), headers=_NOTICE_HEADERS)
    seen = []
    parsed, _ = invoke_structured(_llm_with(seen.append), [])
    assert parsed == _Schema(verdict="keep")
    assert seen == [_EXPECTED_NOTICE]


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
@pytest.mark.parametrize("status", [402, 426])
def test_on_notice_is_called_on_an_error_response_before_the_error_propagates(mock_post, status):
    mock_post.return_value = _notice_response(
        {"error": "client_outdated", "message": "Update.", "request_id": "r-1"},
        status_code=status,
        headers=_NOTICE_HEADERS,
    )
    seen = []
    with pytest.raises(CloudChatError) as excinfo:
        _llm_with(seen.append).invoke([])
    assert excinfo.value.status_code == status
    assert seen == [_EXPECTED_NOTICE]


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_on_notice_is_not_called_without_a_notice(mock_post):
    mock_post.return_value = _notice_response(
        _ok_body(), headers={"Link": f"<{_LINK}>; rel=sunset"}
    )
    seen = []
    invoke_structured(_llm_with(seen.append), [])
    assert seen == []


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_on_notice_is_optional(mock_post):
    mock_post.return_value = _notice_response(_ok_body(), headers=_NOTICE_HEADERS)
    parsed, _ = invoke_structured(_llm_with(None), [])
    assert parsed == _Schema(verdict="keep")


def _raising(_notice):
    raise RuntimeError("callback bug")


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_a_raising_on_notice_does_not_fail_a_successful_call(mock_post, caplog):
    mock_post.return_value = _notice_response(_ok_body(), headers=_NOTICE_HEADERS)
    with caplog.at_level(logging.ERROR, logger=llm_cloud.__name__):
        parsed, _ = invoke_structured(_llm_with(_raising), [])
    assert parsed == _Schema(verdict="keep")
    assert "on_notice failed" in caplog.text


@patch("noemeo_agent_kit.llm_cloud.httpx.post")
def test_a_raising_on_notice_does_not_change_the_cloud_chat_error(mock_post):
    mock_post.return_value = _notice_response(
        {"error": "client_outdated", "message": "Update.", "request_id": "r-1"},
        status_code=426,
        headers=_NOTICE_HEADERS,
    )
    with pytest.raises(CloudChatError) as excinfo:
        _llm_with(_raising).invoke([])
    assert (excinfo.value.status_code, excinfo.value.error_code) == (426, "client_outdated")
