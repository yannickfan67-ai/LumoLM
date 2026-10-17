"""Data split quality checks for LumoLM synthetic training."""
from pathlib import Path

import pytest
import torch

from scripts.build_corpus import build, held_out
from lumolm.train import train


def get_questions(raw):
    return {
        line[len("Question: "):]
        for line in raw.decode("utf-8").splitlines()
        if line.startswith("Question: ")
    }


def test_expanded_corpus_nontrivial_deterministic_and_no_duplicate_questions():
    train1, val1 = build()
    train2, val2 = build()
    assert (train1, val1) == (train2, val2)
    assert len(train1) > 100_000
    assert len(val1) > 10_000
    assert not (get_questions(train1) & get_questions(val1))
    assert held_out("add", 7, 5) == held_out("add", 5, 7)


def test_train_and_resume_checks_independent_validation(tmp_path: Path):
    tr, va = build()
    training = tmp_path / "train.txt"
    validation = tmp_path / "val.txt"
    ckpt = tmp_path / "tiny.pt"
    training.write_bytes(tr)
    validation.write_bytes(va)
    args = ["--data", str(training), "--val-data", str(validation),
            "--checkpoint", str(ckpt), "--device", "cpu",
            "--n-layer", "1", "--n-head", "2", "--n-embd", "32",
            "--block-size", "16", "--steps", "1", "--batch-size", "2",
            "--eval-interval", "1", "--eval-iters", "1"]
    train(args)
    ck = torch.load(ckpt, weights_only=True)
    assert ck["validation_sha256"] and ck["corpus_sha256"]
    train(args + ["--resume", "--steps", "2"])
    validation.write_bytes(va + b"\nchanged\n")
    with pytest.raises(ValueError, match="validation corpus differs"):
        train(args + ["--resume", "--steps", "3"])
