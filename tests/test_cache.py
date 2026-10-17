"""KV-cache correctness and context-overflow regression tests."""
import pytest
import torch

from lumolm.model import GPT, GPTConfig


def small_model(block_size=12):
    torch.manual_seed(7)
    return GPT(GPTConfig(block_size=block_size, n_layer=2, n_head=2, n_embd=32)).eval()


def test_cached_prefill_and_incremental_logits_match_full_attention():
    model = small_model()
    tokens = torch.randint(0, 256, (1, 11))
    with torch.no_grad():
        expected, _ = model(tokens)
        first, loss, cache = model(tokens[:, :5], use_cache=True)
        assert loss is None
        outputs = [first]
        assert len(cache) == model.config.n_layer
        for i in range(5, tokens.size(1)):
            step, _, cache = model(tokens[:, i:i+1], past_key_values=cache, use_cache=True)
            outputs.append(step)
            assert cache[0][0].shape[-2] == i + 1
        actual = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("prompt", [b"A", b"123456789", b"abcdefghijkl", b"abcdefghijklmno"])
def test_cached_and_uncached_greedy_generation_match_across_overflow(prompt):
    model = small_model(block_size=12)
    fast = model.generate(prompt, n_tokens=10, temperature=0, use_kv_cache=True)
    reference = model.generate(prompt, n_tokens=10, temperature=0, use_kv_cache=False)
    assert fast == reference


def test_rejects_invalid_cached_decode_lengths():
    model = small_model()
    _, _, cache = model(torch.ones(1, 2, dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError, match="one token"):
        model(torch.ones(1, 2, dtype=torch.long),
              past_key_values=cache, use_cache=True)
    with pytest.raises(ValueError, match="requires use_cache"):
        model(torch.ones(1, 1, dtype=torch.long), past_key_values=cache)
    _, _, full_cache = model(torch.ones(1, 12, dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.ones(1, 1, dtype=torch.long),
              past_key_values=full_cache, use_cache=True)


def test_non_cached_training_loss_still_has_gradients():
    model = small_model()
    tokens = torch.randint(0, 256, (2, 8))
    _, loss = model(tokens, targets=tokens)
    assert loss is not None
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None
