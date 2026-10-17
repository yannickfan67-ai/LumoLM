"""Regression tests for actual neural code-edit training and safe application."""
import ast
from pathlib import Path

import pytest
import torch

from lumolm.model import GPT, GPTConfig
from lumolm.repair import (
    PATTERNS, build_records, masked_batch, record,
    predict_edit,
)
from lumolm.repair_file import propose


def test_pair_dataset_disjoint_and_training_masks():
    training, heldout=build_records()
    assert len(training)+len(heldout)==len(PATTERNS)*16*48
    assert len(heldout)>500
    assert set((x["prompt"],x["answer"]) for x in training).isdisjoint(
           set((x["prompt"],x["answer"]) for x in heldout))
    for row in training[:25]+heldout[:25]:
        assert len(row["prompt"].encode("utf8")+row["answer"].encode("utf8"))<=193
        ast.parse(row["broken"])
        ast.parse(row["correct"])
        assert row["answer"].strip().startswith("return ")
    x,y=masked_batch(training,7,192,"cpu")
    assert x.shape==y.shape==(7,192)
    assert torch.all((y==-100).any(dim=1))
    cfg=GPTConfig(block_size=192,n_layer=1,n_head=2,n_embd=32)
    model=GPT(cfg)
    _,loss=model(x,y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None


def test_predict_never_applies_reference_fixes():
    model=GPT(GPTConfig(block_size=192,n_layer=1,n_head=2,n_embd=32))
    prompt="FIX\ndef fix_even_0(x):\n    return x % 2 == 1\nEDIT:\n    "
    model.generate=lambda prefix,n_tokens,temperature,stop_sequences:prefix+b"return x + 900\n"
    assert predict_edit(model,prompt)=="return x + 900"


def test_propose_writes_neural_output_not_reference():
    model=GPT(GPTConfig(block_size=192,n_layer=1,n_head=2,n_embd=32))
    model.generate=lambda prefix,n_tokens,temperature,stop_sequences:prefix+b"return n + 11\n"
    proposed,edit=propose(model,"def is_even(n):\n    return n % 2 == 1\n","fix")
    assert edit=="return n + 11"
    assert "return n + 11" in proposed
    assert "return n % 2 == 0" not in proposed
    ast.parse(proposed)


def test_propose_rejects_full_spaghetti_and_unsafe_ast():
    model=GPT(GPTConfig(block_size=192,n_layer=1,n_head=2,n_embd=32))
    with pytest.raises(ValueError,match="one function"):
        propose(model,"def a(x):\n    return x\n\ndef b(y):\n    return y\n","fix")
    with pytest.raises(ValueError,match="context"):
        propose(model,"def "+("broken_"*32)+"(x):\n    return x\n","fix")
    model.generate=lambda prefix,n_tokens,temperature,stop_sequences:prefix+b"return __import__('os').system('echo hi')\n"
    with pytest.raises(ValueError,match="unsupported|only abs"):
        propose(model,"def f(n):\n    return n\n","fix")
