"""Check reproducible code corpora and long-run checkpoint selection."""
import ast
from pathlib import Path

import pytest
import torch

from scripts.build_code_corpus import build, format_example, FUNCTIONS
from lumolm.train import train


def test_code_corpus_disjoint_reproducible_and_valid_python():
    a, b = build(150_000, 30_000)
    aa, bb = build(150_000, 30_000)
    assert (a, b) == (aa, bb)
    assert len(a) > 145_000 and len(b) > 25_000
    training = a.decode("utf-8")
    validation = b.decode("utf-8")
    assert "# Task:" in training and "def " in training
    assert "# Task:" in validation
    # Each sample has two-line preamble followed by a valid Python function.
    for data in (training, validation):
        chunks = [text for text in data.split("\n\n") if text.strip()]
        assert len(chunks) > 100
        for example in chunks:
            ast.parse(example + "\n")
    # Entire examples, not only the final bytes, are independent.
    train_samples = set(training.strip().split("\n\n"))
    val_samples = set(validation.strip().split("\n\n"))
    assert train_samples.isdisjoint(val_samples)


def test_code_sampler_rejects_unsafe_size():
    for tr, va in [(1, 100_000), (100_000_000, 50_000), (100_000, 1)]:
        with pytest.raises(ValueError, match="safety bounds"):
            build(tr, va)


def test_cosine_lr_and_best_validation_checkpoint(tmp_path: Path):
    a, b = build(150_000, 30_000)
    tr = tmp_path / "train.txt"
    va = tmp_path / "val.txt"
    last = tmp_path / "last.pt"
    best = tmp_path / "best.pt"
    tr.write_bytes(a)
    va.write_bytes(b)
    options = [
        "--data", str(tr), "--val-data", str(va),
        "--checkpoint", str(last), "--best-checkpoint", str(best),
        "--steps", "3", "--batch-size", "2", "--block-size", "48",
        "--n-layer", "1", "--n-head", "2", "--n-embd", "32",
        "--lr", "0.001", "--min-lr", "0.0001",
        "--warmup-steps", "1", "--threads", "2",
        "--eval-interval", "1", "--eval-iters", "1", "--device", "cpu"]
    train(options)
    last_state = torch.load(last, weights_only=True)
    best_state = torch.load(best, weights_only=True)
    assert last_state["step"] == 3
    assert best_state["step"] >= 1
    assert best_state["best_val_loss"] is not None
    assert last_state["model"]
    assert best_state["model"]
    with pytest.raises(ValueError, match="best-checkpoint"):
        train(options + ["--best-checkpoint", str(last)])
    with pytest.raises(ValueError, match="min-lr"):
        train(options + ["--min-lr", "100"])
