"""Regression checks for stop sequences and the experimental chat adapter."""
from types import MethodType

import pytest
import torch

from lumolm.chat import LumoChat
from lumolm.model import GPT, GPTConfig


def test_stop_sequence_prevents_next_turn_generation():
    model = GPT(GPTConfig(block_size=16, n_layer=1, n_head=2, n_embd=32))
    def forward_as_newline(self, idx, targets=None, **kwargs):
        logits = torch.full((*idx.shape, 256), -1000.0)
        logits[..., 10] = 1.0
        return logits, None
    model.forward = MethodType(forward_as_newline, model)
    prompt = b"Question: test\nAnswer:"
    output = model.generate(prompt, n_tokens=20, temperature=0,
                            use_kv_cache=False, stop_sequences=(b"\n",))
    assert output == prompt


def test_no_stop_can_span_prompt_boundary():
    model = GPT(GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=32))
    def forward_as_y(self, idx, targets=None, **kwargs):
        logits = torch.full((*idx.shape, 256), -1000.0)
        logits[..., ord("Y")] = 1.0
        return logits, None
    model.forward = MethodType(forward_as_y, model)
    # Only newly generated tokens may form a stop sequence.
    assert model.generate(b"X", 1, temperature=0, use_kv_cache=False,
                          stop_sequences=(b"XY",)) == b"XY"


def test_invalid_stops_fail_fast():
    model = GPT(GPTConfig(block_size=8, n_layer=1, n_head=2, n_embd=32))
    with pytest.raises(ValueError, match="stop sequences"):
        model.generate(b"x", 1, stop_sequences=(b"",))


def test_chat_extracts_only_own_response_and_remembers_turn():
    model = GPT(GPTConfig(block_size=128, n_layer=1, n_head=2, n_embd=32))
    def fake_generate(prompt, n_tokens, temperature, top_k, stop_sequences):
        assert prompt.endswith(b"Answer:")
        assert stop_sequences == (b"\n",)
        return prompt + b" Hello there."
    model.generate = fake_generate
    chat = LumoChat(model, max_new_tokens=20)
    assert chat.ask("Hi\nthere") == "Hello there."
    assert chat.turns == ["Question: Hi there\nAnswer: Hello there."]
    with pytest.raises(ValueError):
        chat.ask("  ")
