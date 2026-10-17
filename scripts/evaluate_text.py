"""Inspect LumoLM text completions and byte-level heldout loss; not a chat benchmark."""
import argparse
import json
from pathlib import Path

import torch

from lumolm.model import GPT, GPTConfig
from lumolm.train import choose_device, batch


@torch.no_grad()
def evaluate(checkpoint: Path, validation: Path, prompts: list[str], device: str,
             eval_batches: int = 16, batch_size: int = 8, seed: int = 9):
    if eval_batches < 1 or batch_size < 1:
        raise ValueError("evaluation counts must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model = model.to(device).eval()
    corpus = validation.read_bytes()
    if len(corpus) < model.config.block_size + 2:
        raise ValueError("validation corpus is too short")
    val_data = torch.tensor(list(corpus), dtype=torch.long)
    torch.manual_seed(seed)
    losses = []
    for _ in range(eval_batches):
        x, y = batch(val_data, batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    # This is token-level cross entropy averaged over a fixed random sample,
    # not exact functional correctness or evidence of instruction following.
    report = {"validation_loss": round(sum(losses) / len(losses), 5),
              "validation_bytes": len(corpus), "training_steps": saved.get("step"),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "seed": seed}
    print("EVAL_TEXT=" + json.dumps(report), flush=True)
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        completed = model.generate(encoded, n_tokens=150, temperature=0,
                                   top_k=40, use_kv_cache=True)
        continuation = completed[len(encoded):].decode("utf-8", "replace")
        response = {"prompt": prompt, "continuation": continuation,
                    "unique_words": len(set(continuation.lower().split()))}
        print("GEN_TEXT=" + json.dumps(response, ensure_ascii=True), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--validation", required=True, type=Path)
    p.add_argument("--prompt", action="append")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batches", type=int, default=16)
    args = p.parse_args()
    if args.threads < 1:
        p.error("threads must be >= 1")
    torch.set_num_threads(args.threads)
    evaluate(args.checkpoint, args.validation,
             args.prompt or ["Once upon a time"], choose_device(args.device),
             eval_batches=args.batches)


if __name__ == "__main__":
    main()
