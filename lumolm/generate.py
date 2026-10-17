"""Generate text from a trained LumoLM checkpoint."""
import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .train import choose_device


def generate(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lumolm.pt"))
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8, help="0 for greedy decoding")
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    p.add_argument("--no-kv-cache", action="store_true", help="disable incremental KV caching")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.to(device)
    output = model.generate(args.prompt.encode("utf-8"), args.max_new_tokens,
                            args.temperature, args.top_k,
                            use_kv_cache=not args.no_kv_cache)
    text = output.decode("utf-8", errors="replace")
    print(text)
    return text


if __name__ == "__main__":
    generate()
