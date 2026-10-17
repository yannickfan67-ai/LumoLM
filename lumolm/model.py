"""Decoder-only Transformer with byte-level (0..255) tokens."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    vocab_size: int = 256

    def __post_init__(self):
        if min(self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size != 256:
            raise ValueError("LumoLM currently uses exactly 256 UTF-8 byte tokens")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        if past_key_value is not None:
            if t != 1:
                raise ValueError("cached attention accepts exactly one new token")
            old_k, old_v = past_key_value
            k = torch.cat((old_k, k), dim=-2)
            v = torch.cat((old_v, v), dim=-2)
        # A single query with preceding cached keys must attend to every key.
        # is_causal=True for a (1, T) attention matrix would mask all but key 0.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=past_key_value is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().reshape(b, t, c)
        y = self.resid_dropout(self.proj(y))
        return (y, (k, v)) if use_cache else y

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        attn_result = self.attn(self.ln1(x), past_key_value, use_cache)
        if use_cache:
            attn_output, present = attn_result
        else:
            attn_output = attn_result
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return (x, present) if use_cache else x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight  # weight tying

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                *, past_key_values=None, use_cache: bool = False):
        b, t = idx.shape
        if t < 1:
            raise ValueError("input must contain at least one token")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError("cached layer count does not match model")
            if t != 1:
                raise ValueError("cached decoding accepts one token at a time")
            past_length = past_key_values[0][0].size(-2)
            if any(k.size(-2) != past_length or v.size(-2) != past_length
                   for k, v in past_key_values):
                raise ValueError("cached sequence lengths differ across layers")
        else:
            past_length = 0
        if t + past_length > self.config.block_size:
            raise ValueError(
                f"sequence length {t + past_length} exceeds block_size {self.config.block_size}"
            )
        positions = torch.arange(past_length, past_length + t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        presents = []
        for layer, block in enumerate(self.blocks):
            if use_cache:
                prior = None if past_key_values is None else past_key_values[layer]
                x, present = block(x, prior, use_cache=True)
                presents.append(present)
            else:
                x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: bytes, n_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, use_kv_cache: bool = True,
                 stop_sequences: tuple[bytes, ...] = ()) -> bytes:
        if n_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError("n_tokens, temperature, and top_k must be nonnegative")
        if any(not isinstance(stop, bytes) or not stop for stop in stop_sequences):
            raise ValueError('stop sequences must be nonempty byte strings')
        self.eval()
        device = next(self.parameters()).device
        sequence = bytearray(prompt if prompt else b"\n")
        initial_length = len(sequence)
        past = None
        for _ in range(n_tokens):
            if not use_kv_cache:
                idx = torch.tensor(list(sequence[-self.config.block_size:]),
                                   device=device, dtype=torch.long)[None]
                logits, _ = self(idx)
            elif past is None or past[0][0].size(-2) >= self.config.block_size:
                # Rebuild cache after context overflow. Learned absolute
                # positions restart at zero when the context window slides.
                idx = torch.tensor(list(sequence[-self.config.block_size:]),
                                   device=device, dtype=torch.long)[None]
                logits, _, past = self(idx, use_cache=True)
            else:
                idx = torch.tensor([[sequence[-1]]], device=device, dtype=torch.long)
                logits, _, past = self(idx, past_key_values=past, use_cache=True)

            scores = logits[0, -1]
            if temperature == 0:
                token = int(scores.argmax().item())
            else:
                scores = scores / temperature
                if top_k:
                    kth = torch.topk(scores, min(top_k, scores.numel())).values[-1]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                token = int(torch.multinomial(F.softmax(scores, dim=-1), 1).item())
            sequence.append(token)
            for stop in stop_sequences:
                if len(sequence) - initial_length >= len(stop) and sequence.endswith(stop):
                    return bytes(sequence[:-len(stop)])
        return bytes(sequence)
