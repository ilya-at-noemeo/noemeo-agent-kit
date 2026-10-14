from unittest.mock import MagicMock

import pytest

from noemeo_agent_kit.structured_output import invoke_structured
from noemeo_agent_kit.token_usage import TokenUsage


def test_invoke_structured_returns_parsed_and_token_usage():
    raw = MagicMock()
    raw.usage_metadata = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    llm = MagicMock()
    llm.invoke.return_value = {"parsing_error": None, "parsed": "the-result", "raw": raw}

    parsed, usage = invoke_structured(llm, ["msg"])

    assert parsed == "the-result"
    assert usage == TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)


def test_invoke_structured_reraises_parsing_error():
    llm = MagicMock()
    llm.invoke.return_value = {
        "parsing_error": ValueError("bad json"),
        "parsed": None,
        "raw": MagicMock(),
    }

    with pytest.raises(ValueError, match="bad json"):
        invoke_structured(llm, ["msg"])
