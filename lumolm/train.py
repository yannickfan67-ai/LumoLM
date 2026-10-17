"""Train LumoLM from a UTF-8 text file; supports resumable checkpoints."""
import argparse
import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import GPT, GPTConfig
from .qa import parse_qa_records, qa_batch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="UTF-8 corpus (not bundled into checkpoint)")
    p.add_argument("--val-data", type=Path, help="optional separate, held-out UTF-8 validation corpus")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lumolm.pt"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--task", choices=("language_model", "qa"), default="language_model",
                   help="standard next-byte LM or answer-only Question/Answer supervision")
    p.add_argument("--steps", type=int, default=1000, help="total optimizer steps, including resumed steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def choose_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_corpus(raw: bytes, block_size: int):
    minimum = block_size + 2
    if len(raw) < 2 * minimum:
        raise ValueError(f"corpus must contain at least {2 * minimum} UTF-8 bytes")
    valid_count = max(minimum, int(len(raw) * 0.1))
    valid_count = min(valid_count, len(raw) - minimum)
    return torch.tensor(list(raw[:-valid_count]), dtype=torch.long), torch.tensor(list(raw[-valid_count:]), dtype=torch.long)


def batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, iters,
                  batch_fn=batch):
    model.eval()
    losses = []
    for data in (train_data, val_data):
        values = []
        for _ in range(iters):
            x, y = batch_fn(data, batch_size, block_size, device)
            _, loss = model(x, y)
            values.append(float(loss.item()))
        losses.append(sum(values) / len(values))
    model.train()
    return losses


def save_checkpoint(path: Path, model: GPT, optimizer, step: int, digest: str,
                    validation_digest: str | None = None,
                    task: str = "language_model"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "config": asdict(model.config),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "corpus_sha256": digest,
        "validation_sha256": validation_digest,
        "task": task,
        "rng_state": torch.get_rng_state(),
    }
    fd, name = tempfile.mkstemp(prefix=".lumolm-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def train(argv=None):
    args = parse_args(argv)
    for key in ("steps", "batch_size", "grad_accum", "eval_interval", "eval_iters"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    torch.manual_seed(args.seed)
    if args.task == "qa" and args.val_data is None:
        raise ValueError("--task qa requires a separate --val-data corpus")
    device = choose_device(args.device)
    raw = args.data.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    validation_raw = args.val_data.read_bytes() if args.val_data is not None else None
    validation_digest = (hashlib.sha256(validation_raw).hexdigest()
                         if validation_raw is not None else None)
    saved = None
    if args.resume:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["corpus_sha256"] != digest:
            raise ValueError("resume corpus differs from the original training corpus")
        if saved.get("validation_sha256") != validation_digest:
            raise ValueError("resume validation corpus differs from the original")
        if saved.get("task", "language_model") != args.task:
            raise ValueError("resume task differs from the checkpoint")
        if saved["format_version"] != 1:
            raise ValueError("unsupported checkpoint format")
        cfg = GPTConfig(**saved["config"])
    else:
        cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer,
                        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    if args.task == "qa":
        train_data, train_skipped = parse_qa_records(raw, cfg.block_size)
        val_data, val_skipped = parse_qa_records(validation_raw, cfg.block_size)
        batch_fn = qa_batch
        print(f"qa train_examples={len(train_data)} train_skipped={train_skipped} "
              f"val_examples={len(val_data)} val_skipped={val_skipped}", flush=True)
    elif validation_raw is None:
        train_data, val_data = split_corpus(raw, cfg.block_size)
        batch_fn = batch
    else:
        minimum = cfg.block_size + 2
        if len(raw) < minimum or len(validation_raw) < minimum:
            raise ValueError(f"both corpora must contain at least {minimum} bytes")
        train_data = torch.tensor(list(raw), dtype=torch.long)
        val_data = torch.tensor(list(validation_raw), dtype=torch.long)
        batch_fn = batch
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng_state"])
        start = int(saved["step"])
    params = sum(p.numel() for p in model.parameters())
    print(f"LumoLM: {params:,} parameters | device={device} | bytes={len(raw)} | step={start}", flush=True)
    model.train()
    for step in range(start + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = batch_fn(train_data, args.batch_size, cfg.block_size, device)
            _, loss = model(x, y)
            train_loss += loss.detach().item() / args.grad_accum
            (loss / args.grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            train_eval, val_eval = estimate_loss(
                model, train_data, val_data, args.batch_size,
                cfg.block_size, device, args.eval_iters, batch_fn,
            )
            print(f"step={step} batch_loss={train_loss:.4f} train={train_eval:.4f} val={val_eval:.4f}", flush=True)
            save_checkpoint(args.checkpoint, model, optimizer, step, digest,
                            validation_digest, args.task)
    if start >= args.steps:
        print("checkpoint already reached requested --steps; no additional training", flush=True)
    return args.checkpoint


if __name__ == "__main__":
    train()
