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

    def _make_prompt(self, question: str) -> bytes:
        # Byte-level LumoLM uses a *byte* context window, not words/tokens.
        # The previous implementation joined all old turns first; generate()
        # silently discarded the beginning of the prompt, sometimes including
        # the current question. Select fitting history BEFORE inference.
        current = f"Question: {question}\nAnswer:".encode("utf-8")
        limit = self.model.config.block_size
        if len(current) > limit:
            raise ValueError(
                f"question requires {len(current)} bytes with the Q/A template, "
                f"but this checkpoint only supports {limit} context bytes; "
                "shorten the question or train a model with a larger block_size"
            )
        parts = [current]
        self.last_history_turns = 0
        for previous in reversed(self.turns):
            encoded = previous.encode("utf-8")
            used = sum(len(part) for part in parts) + len(parts)
            if used + len(encoded) > limit:
                break
            parts.insert(0, encoded)
            self.last_history_turns += 1
        return b"\n".join(parts)

    def ask(self, question: str) -> str:
        question = question.strip().replace("\r", " ").replace("\n", " ")
        if not question:
            raise ValueError("question cannot be empty")
        prompt = self._make_prompt(question)
        raw = self.model.generate(
            prompt, n_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=self.top_k, stop_sequences=(b"\n",)
        )
        completion = raw[len(prompt):]
        answer = completion.decode("utf-8", errors="replace").strip()
        self.turns.append(f"Question: {question}\nAnswer: {answer}")
        # The model cannot retain a long conversation in its current window.
        # Bound Python-side history too, and keep only useful recent context.
        if len(self.turns) > 64:
            self.turns = self.turns[-64:]
        return answer


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--question", action="append", help="ask once, or repeat for multi-turn chat; omit for interactive mode")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--show-context", action="store_true",
                   help="report how many previous dialogue turns fit in the byte context window")
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
        for question in args.question:
            print("You:", question)
            print("LumoLM:", chat.ask(question))
            if args.show_context:
                print(f"[diagnostic] retained prior turns: {chat.last_history_turns}")
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
            try:
                answer = chat.ask(message)
            except ValueError as exc:
                print(f"[input error] {exc}")
                continue
            print("LumoLM:", answer)
            if args.show_context:
                print(f"[diagnostic] retained prior turns: {chat.last_history_turns}")


if __name__ == "__main__":
    main()
