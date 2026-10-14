import pytest

from noemeo_agent_kit.config import DEFAULT_REASONING_EFFORT, CapabilityConfig, env_choice, env_str


def test_env_str_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("NOEMEO_TEST_VAR", raising=False)
    assert env_str("NOEMEO_TEST_VAR", "fallback") == "fallback"


def test_env_str_uses_env_value_when_set(monkeypatch):
    monkeypatch.setenv("NOEMEO_TEST_VAR", "override")
    assert env_str("NOEMEO_TEST_VAR", "fallback") == "override"


def test_env_choice_accepts_valid_value(monkeypatch):
    monkeypatch.setenv("NOEMEO_TEST_DETAIL", "high")
    assert env_choice("NOEMEO_TEST_DETAIL", "low", ("low", "high", "auto")) == "high"


def test_env_choice_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("NOEMEO_TEST_DETAIL", "bogus")
    with pytest.raises(ValueError, match="NOEMEO_TEST_DETAIL"):
        env_choice("NOEMEO_TEST_DETAIL", "low", ("low", "high", "auto"))


def test_capability_config_defaults():
    config = CapabilityConfig(model="gpt-5.6")
    assert config.reasoning_effort == DEFAULT_REASONING_EFFORT
    assert config.detail is None
