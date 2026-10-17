"""Evaluate trained LumoLM on seen vs held-out synthetic arithmetic examples.

This is NOT a general language benchmark: test records use the same small
synthetic templates, with distinct operands. Report outcomes without gaming
the score or silently replacing model answers.
"""
import argparse
import json
from pathlib import Path

import torch

from lumolm.model import GPT, GPTConfig
from lumolm.train import choose_device


def pairs(path: Path):
    lines = path.read_text("utf-8").splitlines()
    for i in range(len(lines) - 1):
        if lines[i].startswith("Question: ") and lines[i + 1].startswith("Answer: "):
            yield lines[i][10:], lines[i + 1][8:]


def run_probe(model, label, cases, max_new_tokens=24):
    correct = 0
    outputs = []
    with torch.no_grad():
        for q, expected in cases:
            prefix = f"Question: {q}\nAnswer:".encode("utf-8")
            generated = model.generate(prefix, max_new_tokens, temperature=0,
                                       stop_sequences=(b"\n",))
            actual = generated[len(prefix):].decode("utf-8", "replace").strip()
            correct += actual == expected
            outputs.append({"question": q, "expected": expected, "got": actual})
    report = {"split": label, "exact": correct, "count": len(cases),
              "rate": round(correct / len(cases), 4) if cases else None}
    print("SCORE=" + json.dumps(report, ensure_ascii=True), flush=True)
    for item in outputs[:8]:
        print("SAMPLE=" + json.dumps({"split": label, **item}, ensure_ascii=True),
              flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--samples", type=int, default=24)
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be positive")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.to(choose_device(args.device)).eval()
    # Keep deterministic, limited-size samples and print all score denominators.
    seen = [(q, a) for q, a in pairs(args.train) if a.lstrip("-").isdigit()]
    held_out = [(q, a) for q, a in pairs(args.val) if a.lstrip("-").isdigit()]
    # Sample evenly across file after deterministic training dataset shuffle.
    run_probe(model, "seen_math", seen[:args.samples])
    run_probe(model, "heldout_math", held_out[:args.samples])
    run_probe(model, "out_of_domain", [
        ("What is 67+82?", "149"),
        ("What is 3+92?", "95"),
        ("What is 5*17?", "85"),
        ("What is 47-88?", "-41"),
        ("What year was LumoLM created?", "unknown"),
    ])


if __name__ == "__main__":
    main()
