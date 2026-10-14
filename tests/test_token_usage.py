from types import SimpleNamespace

from noemeo_agent_kit.token_usage import TokenUsage, extract_token_usage


def test_extract_token_usage_reads_real_counts():
    raw = SimpleNamespace(
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 40,
            "total_tokens": 160,
            "output_token_details": {"reasoning": 15},
        }
    )
    usage = extract_token_usage(raw)
    assert usage == TokenUsage(
        input_tokens=120, output_tokens=40, total_tokens=160, reasoning_tokens=15
    )


def test_extract_token_usage_missing_reasoning_details_is_none():
    raw = SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    usage = extract_token_usage(raw)
    assert usage.reasoning_tokens is None


def test_extract_token_usage_no_metadata_returns_empty():
    raw = SimpleNamespace(usage_metadata=None)
    assert extract_token_usage(raw) == TokenUsage()
