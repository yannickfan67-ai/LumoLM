"""Regression tests for answer-only QA supervision and checkpoint compatibility."""
from pathlib import Path

import pytest
import torch

from lumolm.qa import parse_qa_records, qa_batch
from lumolm.model import GPT, GPTConfig
from lumolm.train import train


def test_qa_batch_only_scores_answer_and_never_next_question():
    raw = (b"Question: What is 2+2?\nAnswer: 4\n"
           b"Question: Who?\nAnswer: LumoLM\n")
    examples, skipped = parse_qa_records(raw, block_size=64)
    assert len(examples) == 2 and skipped == 0
    torch.manual_seed(1)
    x, y = qa_batch(examples, 8, 64, "cpu")
    assert x.shape == y.shape
    assert bool(torch.any(y == -100))
    assert bool(torch.any(y != -100))
    for row in range(len(x)):
        scores = y[row][y[row] != -100].tolist()
        assert scores
        assert scores[-1] == 10  # each example ends with a newline
        assert 10 not in scores[:-1]  # no subsequent questions
    model = GPT(GPTConfig(block_size=64, n_layer=1, n_head=2, n_embd=32))
    _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.token_emb.weight.grad is not None


def test_qa_rejects_missing_or_oversized_pairs():
    with pytest.raises(ValueError, match="Question:/Answer:"):
        parse_qa_records(b"plain text", 32)
    with pytest.raises(ValueError, match="all QA records"):
        parse_qa_records(b"Question: " + b"x" * 50 + b"\nAnswer: yes\n", 20)


def test_qa_checkpoint_task_and_resume(tmp_path: Path):
    records = b"Question: Who are you?\nAnswer: LumoLM\n" * 20
    training = tmp_path / "train.txt"
    validation = tmp_path / "val.txt"
    checkpoint = tmp_path / "qa.pt"
    training.write_bytes(records)
    validation.write_bytes(records)
    opts = ["--data", str(training), "--val-data", str(validation),
            "--checkpoint", str(checkpoint), "--task", "qa",
            "--block-size", "64", "--n-layer", "1", "--n-head", "2",
            "--n-embd", "32", "--batch-size", "2", "--eval-interval", "1",
            "--eval-iters", "1", "--device", "cpu"]
    train(opts + ["--steps", "1"])
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["task"] == "qa"
    train(opts + ["--resume", "--steps", "2"])
    with pytest.raises(ValueError, match="resume task differs"):
        train([x for x in opts if x not in ("--task", "qa")] + ["--resume", "--steps", "3"])
    with pytest.raises(ValueError, match="requires a separate"):
        train(["--data", str(training), "--task", "qa", "--device", "cpu"])
