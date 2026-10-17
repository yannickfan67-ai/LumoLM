"""Exercise actual checkpoint completions; this is not an assistant-quality benchmark."""
import argparse
import json

import torch

from lumolm.model import GPT, GPTConfig
from lumolm.train import choose_device


PROMPTS = (
    "Hello",
    "Question: What is LumoLM?\nAnswer:",
    "Question: Can you talk?\nAnswer:",
    "Question: Do you know Minecraft?\nAnswer:",
    "Question: What is 2+2?\nAnswer:",
    "Question: 你好嗎？\nAnswer:",
    "Question: What is 7+5?\nAnswer:",  # unseen arithmetic, no claimed correctness
    "Question: 你知道現在的日期嗎？\nAnswer:",  # unseen, no clock
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(choose_device(args.device)).eval()
    print("MODEL_CONFIG=" + json.dumps(ckpt["config"], sort_keys=True))
    print("TRAINING_STEP=" + str(ckpt["step"]))
    for p in PROMPTS:
        raw = model.generate(p.encode("utf-8"), args.max_new_tokens, temperature=0)
        continuation = raw[len(p.encode("utf-8")):]
        answer = continuation.decode("utf-8", "replace")
        first_line = answer.split("\n", 1)[0]
        print("PROBE=" + json.dumps({
            "prompt": p, "first_line": first_line,
            "raw_continuation": answer,
            "invalid_utf8_replacements": answer.count("\ufffd"),
        }, ensure_ascii=True))


if __name__ == "__main__":
    main()
