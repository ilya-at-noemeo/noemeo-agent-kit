from noemeo_agent_kit.prompt_hash import hash_prompts


def test_same_input_same_hash():
    assert hash_prompts("a", "b") == hash_prompts("a", "b")


def test_different_input_different_hash():
    assert hash_prompts("a", "b") != hash_prompts("a", "c")


def test_hash_length_is_12():
    assert len(hash_prompts("anything")) == 12


def test_order_matters():
    assert hash_prompts("a", "b") != hash_prompts("b", "a")
