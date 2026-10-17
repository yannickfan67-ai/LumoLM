"""A byte-level 128-token context must never silently swallow the latest question."""
from types import MethodType
import pytest
from lumolm.chat import LumoChat
from lumolm.model import GPT, GPTConfig


def fake_model(block_size=96):
    return GPT(GPTConfig(block_size=block_size, n_layer=1, n_head=2, n_embd=32))


def attach_recorder(model):
    inputs = []
    def generate(self, prompt, n_tokens, temperature, top_k, stop_sequences):
        inputs.append(prompt)
        return prompt + b" ok"
    model.generate = MethodType(generate, model)
    return inputs


def test_latest_question_never_truncated_and_history_bounded():
    model = fake_model(96)
    prompts = attach_recorder(model)
    chat = LumoChat(model)
    for i in range(35):
        assert chat.ask(f"question {i}?") == "ok"
        assert len(prompts[-1]) <= model.config.block_size
        assert prompts[-1].endswith(f"Question: question {i}?\nAnswer:".encode())
    assert chat.last_history_turns >= 1
    assert b"question 0?" not in prompts[-1]
    assert b"question 34?" in prompts[-1]


def test_overly_long_question_rejected_not_cut_off():
    model = fake_model(64)
    calls = attach_recorder(model)
    chat = LumoChat(model)
    with pytest.raises(ValueError, match="context bytes"):
        chat.ask("a" * 100)
    assert calls == []
    assert chat.turns == []


def test_multibyte_characters_consume_actual_utf8_bytes():
    model = fake_model(48)
    prompts = attach_recorder(model)
    chat = LumoChat(model)
    with pytest.raises(ValueError, match="context bytes"):
        chat.ask("你" * 20)
    assert chat.ask("你好?") == "ok"
    assert prompts[0].endswith("Question: 你好?\nAnswer:".encode("utf-8"))
