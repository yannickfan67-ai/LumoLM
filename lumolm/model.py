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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.reshape(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().reshape(b, t, c)
        return self.resid_dropout(self.proj(y))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        positions = torch.arange(t, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: bytes, n_tokens: int, temperature: float = 0.8,
                 top_k: int = 40) -> bytes:
        if n_tokens < 0 or temperature < 0 or top_k < 0:
            raise ValueError("n_tokens, temperature, and top_k must be nonnegative")
        self.eval()
        device = next(self.parameters()).device
        sequence = bytearray(prompt if prompt else b"\n")
        for _ in range(n_tokens):
            idx = torch.tensor(list(sequence[-self.config.block_size:]), device=device).long()[None]
            logits, _ = self(idx)
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
        return bytes(sequence)
