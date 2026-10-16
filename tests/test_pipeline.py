from pathlib import Path

import pytest
import torch

from lumolm.model import GPT, GPTConfig
from lumolm.train import train
from lumolm.generate import generate


def test_model_shape_and_causality():
    torch.manual_seed(1)
    model = GPT(GPTConfig(block_size=16, n_layer=1, n_head=2, n_embd=32)).eval()
    a = torch.randint(0, 256, (1, 16))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 256
    with torch.no_grad():
        logits_a, loss = model(a, a)
        logits_b, _ = model(b)
    assert logits_a.shape == (1, 16, 256)
    assert torch.isfinite(loss)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_train_resume_and_generate(tmp_path: Path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Hello LumoLM! Next byte prediction. 你好！\n" * 25), encoding="utf-8")
    ckpt = tmp_path / "model.pt"
    base = ["--data", str(corpus), "--checkpoint", str(ckpt),
            "--device", "cpu", "--batch-size", "2", "--block-size", "16",
            "--n-layer", "1", "--n-head", "2", "--n-embd", "32",
            "--eval-iters", "1", "--eval-interval", "1"]
    train(base + ["--steps", "2"])
    assert ckpt.exists()
    train(base + ["--resume", "--steps", "3"])
    saved = torch.load(ckpt, weights_only=True)
    assert saved["step"] == 3
    out = generate(["--checkpoint", str(ckpt), "--prompt", "Hi", "--device", "cpu",
                    "--max-new-tokens", "3", "--temperature", "0"])
    assert out.startswith("Hi") and len(out) >= 2
    corpus.write_text("changed corpus " * 50, encoding="utf-8")
    with pytest.raises(ValueError, match="corpus differs"):
        train(base + ["--resume", "--steps", "4"])


def test_invalid_config_and_short_data():
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(n_head=3, n_embd=32)
    with pytest.raises(ValueError, match="corpus"):
        from lumolm.train import split_corpus
        split_corpus(b"short", 32)
