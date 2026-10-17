"""No arithmetic lookups are allowed in learned LumoLM evaluation."""
import pytest
import torch

from lumolm.addition import (
    TASK, create_splits, encoded_prompt, example_for, held_out, predict, prompt_for,
    training_batch,
)
from lumolm.model import GPT, GPTConfig


def test_reversed_number_representation():
    assert prompt_for(17, 28) == b"71+82="
    assert bytes(example_for(17, 28)) == b"71+82=540\n"  # 17+28=45
    assert bytes(example_for(1, 1)) == b"10+10=200\n"
    with pytest.raises(ValueError):
        prompt_for(100, 3)
    with pytest.raises(ValueError):
        prompt_for(-1, 1)


def test_splits_are_canonical_and_disjoint():
    train, test = create_splits()
    assert len(train) + len(test) == 10000
    assert len(train) > 8000 and len(test) > 500
    assert held_out(3, 8) == held_out(8, 3)
    assert set(map(tuple, train.tolist())).isdisjoint(set(map(tuple, test.tolist())))
    seen = set()
    withheld = set()
    for records, dest in ((train, seen), (test, withheld)):
        for row in records:
            row = bytes(row.tolist())
            a = int(row[:2].decode()[::-1])
            b = int(row[3:5].decode()[::-1])
            dest.add(tuple(sorted((a, b))))
    assert seen.isdisjoint(withheld)


def test_masking_and_backward_pass():
    training, _ = create_splits()
    x, targets = training_batch(training, 8, "cpu")
    assert x.shape == targets.shape == (8, 9)
    assert torch.all(targets[:, :5] == -100)
    assert torch.all(targets[:, 5:] != -100)
    assert torch.all(targets[:, -1] == 10)
    model = GPT(GPTConfig(block_size=12, n_layer=1, n_head=2, n_embd=32))
    _, loss = model(x, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_prediction_decodes_model_bytes_without_math_solver():
    model = GPT(GPTConfig(block_size=12, n_layer=1, n_head=2, n_embd=32))
    def mock_generate(prompt, n_tokens, temperature, stop_sequences):
        assert prompt == b"71+82="
        assert n_tokens == 4
        return prompt + b"540"  # model supposedly predicts reversed 045
    model.generate = mock_generate
    # If the implementation instead computed a+b this test would not
    # detect it, so intentionally return an INCORRECT model prediction too.
    value, raw = predict(model, 17, 28)
    assert value == 45 and raw == "540"
    model.generate = lambda *args, **kw: b"71+82=990"
    value, raw = predict(model, 17, 28)
    assert value == 99 and raw == "990"
