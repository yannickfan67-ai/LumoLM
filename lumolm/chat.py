"""Experimental terminal dialogue with a *trained* LumoLM checkpoint.

This is a byte-level next-token model, not an instruction-following assistant.
Responses can be nonsense even when checkpoints load and tests pass.
"""
import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .train import choose_device


class LumoChat:
    def __init__(self, model: GPT, max_new_tokens: int = 120, temperature: float = 0.0,
                 top_k: int = 40):
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.turns: list[str] = []

    def ask(self, question: str) -> str:
        question = question.strip().replace("\r", " ").replace("\n", " ")
        if not question:
            raise ValueError("question cannot be empty")
        prompt = "\n".join(self.turns + [f"Question: {question}\nAnswer:"]).encode("utf-8")
        raw = self.model.generate(
            prompt, n_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=self.top_k, stop_sequences=(b"\n",)
        )
        completion = raw[len(prompt):]
        answer = completion.decode("utf-8", errors="replace").strip()
        self.turns.append(f"Question: {question}\nAnswer: {answer}")
        # Preserve only dialogue text that fits roughly within the context window.
        # Byte-based generation already truncates the complete prompt exactly.
        while len(self.turns) > 1 and len("\n".join(self.turns).encode("utf-8")) > self.model.config.block_size:
            self.turns.pop(0)
        return answer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--question", help="single question; omit to use interactive mode")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    args = p.parse_args(argv)
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.to(choose_device(args.device))
    chat = LumoChat(model, args.max_new_tokens, args.temperature, args.top_k)
    print("LumoLM experimental chat: not a trained assistant; expect poor responses.")
    if args.question is not None:
        print("LumoLM:", chat.ask(args.question))
    else:
        print("Type 'exit' to finish.")
        while True:
            try:
                message = input("You: ")
            except (KeyboardInterrupt, EOFError):
                break
            if message.strip().lower() in {"exit", "quit"}:
                break
            if not message.strip():
                continue
            print("LumoLM:", chat.ask(message))


if __name__ == "__main__":
    main()
