import base64
import json

import pytest
from pydantic import BaseModel

from noemeo_agent_kit.driver import log_invocation, parse_image_event, run_capability
from noemeo_agent_kit.token_usage import TokenUsage


def test_parse_image_event_decodes_base64():
    event = {"image_base64": base64.b64encode(b"fake-jpeg-bytes").decode()}
    assert parse_image_event(event) == b"fake-jpeg-bytes"


def test_parse_image_event_rejects_missing_field():
    with pytest.raises(ValueError, match="image_base64"):
        parse_image_event({})


def test_parse_image_event_rejects_empty_field():
    with pytest.raises(ValueError, match="image_base64"):
        parse_image_event({"image_base64": ""})


class _Result(BaseModel):
    value: str


def test_run_capability_returns_json_safe_body_and_logs(capsys):
    def fn():
        return _Result(value="ok"), TokenUsage(input_tokens=5)

    body = run_capability(fn, capability="test-capability", model="gpt-5.6", prompt_hash="abc123")

    assert body == {"value": "ok"}

    logged = json.loads(capsys.readouterr().out)
    assert logged["capability"] == "test-capability"
    assert logged["model"] == "gpt-5.6"
    assert logged["prompt_hash"] == "abc123"
    assert logged["token_usage"]["input_tokens"] == 5
    assert logged["latency_ms"] >= 0


def test_run_capability_never_puts_token_usage_in_the_body():
    def fn():
        return _Result(value="ok"), TokenUsage(input_tokens=999)

    body = run_capability(fn, capability="test-capability", model="gpt-5.6")

    assert "token_usage" not in body


def test_log_invocation_omits_token_usage_when_none(capsys):
    log_invocation(capability="c", model="m")
    logged = json.loads(capsys.readouterr().out)
    assert "token_usage" not in logged
