"""Train/test genuine *learned* two-digit addition using the LumoLM GPT.

This is a narrow arithmetic-only experiment, not a general chat assistant.
The digit representation reverses each operand and the result so the model
predicts units before tens (standard column order). Decoding reverses digits
only; NO symbolic addition is executed by the inference path.

Train/test pairs are separated by unordered operand identity, so (3, 8)
and (8, 3) can never cross the split. Evaluation reports exact matches on
fresh operand pairs from the model itself, not a Python calculator.
"""
import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .train import choose_device

MAX_OPERAND = 99
SEED = 20261017
TASK = "reversed_two_digit_addition_v1"
QUESTION = re.compile(r"^\s*(\d{1,2})\s*\+\s*(\d{1,2})\s*$")


def held_out(a: int, b: int) -> bool:
    a, b = sorted((a, b))
    return (a * 37 + b * 61 + a * b * 7) % 11 == 0


def prompt_for(a: int, b: int) -> bytes:
    if not (0 <= a <= MAX_OPERAND and 0 <= b <= MAX_OPERAND):
        raise ValueError("addition experiment supports operands from 0 through 99 only")
    return (f"{a:02d}"[::-1] + "+" + f"{b:02d}"[::-1] + "=").encode("ascii")


def example_for(a: int, b: int):
    # Training target generation is allowed to compute the ground truth.
    # Model inference does not use this function.
    data = prompt_for(a, b) + f"{a+b:03d}"[::-1].encode("ascii") + b"\n"
    assert len(data) == 10
    return list(data)


def create_splits():
    train, validation = [], []
    for a in range(MAX_OPERAND + 1):
        for b in range(MAX_OPERAND + 1):
            (validation if held_out(a, b) else train).append(example_for(a, b))
    return torch.tensor(train, dtype=torch.long), torch.tensor(validation, dtype=torch.long)


def training_batch(records, count: int, device: str):
    pick = torch.randint(records.size(0), (count,))
    tokens = records[pick].to(device)
    x, y = tokens[:, :-1], tokens[:, 1:].clone()
    y[:, :5] = -100
    return x, y


@torch.no_grad()
def predict(model: GPT, a: int, b: int):
    model.eval()
    prefix = prompt_for(a, b)
    generated = model.generate(prefix, 4, temperature=0, stop_sequences=(b"\n",))
    suffix = generated[len(prefix):]
    try:
        text = suffix.decode("ascii")
    except UnicodeDecodeError:
        return None, repr(suffix)
    if len(text) != 3 or not text.isdigit():
        return None, text
    # Only undo the deliberately reversed output representation.
    # DO NOT replace this with a computed a+b: that would falsify evaluation.
    return int(text[::-1]), text


@torch.no_grad()
def report(model: GPT, data, label: str, device: str, sample_limit: int = 0):
    model.eval()
    num = min(len(data), sample_limit) if sample_limit else len(data)
    examples = []
    accurate = 0
    for row in data[:num]:
        a = int(bytes(row[:2].tolist()).decode("ascii")[::-1])
        b = int(bytes(row[3:5].tolist()).decode("ascii")[::-1])
        predicted, generated = predict(model, a, b)
        accurate += predicted == a + b
        if len(examples) < 10:
            examples.append({"question": f"{a}+{b}", "predicted": predicted,
                             "expected": a+b, "generated_reversed": generated})
    outcome = {"split": label, "correct": accurate, "total": num,
               "accuracy": round(accurate / num, 6) if num else None,
               "examples": examples}
    print("EVAL=" + json.dumps(outcome, ensure_ascii=True), flush=True)
    return outcome


def new_model(args):
    return GPT(GPTConfig(block_size=12, n_layer=args.n_layer,
                         n_head=args.n_head, n_embd=args.n_embd))


def train(args):
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    training, validation = create_splits()
    model = new_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # This run is from scratch, not a continuation of a generic text checkpoint.
    print(f"ADDITION training_pairs={len(training)} heldout_pairs={len(validation)} "
          f"parameters={sum(p.numel() for p in model.parameters())} device={device}",
          flush=True)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = training_batch(training, args.batch_size, device)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = training_batch(validation, 256, device)
                _, val_loss = model(vx, vy)
            print(f"ADDITION_STEP={step} train_loss={loss.item():.4f} "
                  f"heldout_token_loss={val_loss.item():.4f}", flush=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "config": asdict(model.config),
                "model": model.state_dict(), "step": args.steps, "task": TASK,
                "seed": SEED, "train_pairs": len(training),
                "heldout_pairs": len(validation)}, args.checkpoint)
    report(model, training, "training_pairs", device, args.eval_sample)
    report(model, validation, "heldout_pairs", device, args.eval_sample)
    for a, b in [(1, 1), (3, 3), (4, 4), (5, 5), (2, 1), (9, 9),
                 (17, 28), (56, 87)]:
        answer, digits = predict(model, a, b)
        print("PROBE=" + json.dumps({"a":a,"b":b,"model_answer":answer,
                                      "target":a+b,"raw_digits":digits}), flush=True)


def load(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("task") != TASK:
        raise ValueError("not an arithmetic checkpoint; generic chat/story weights cannot be used")
    model = GPT(GPTConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(required=True)
    actions.add_argument("--train", action="store_true")
    actions.add_argument("--evaluate", action="store_true")
    actions.add_argument("--question", action="append", help="e.g. --question 3+3")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/addition.pt"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--eval-sample", type=int, default=0,
                   help="max examples per split; 0 evaluates ALL pairs")
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    if min(args.steps, args.batch_size, args.eval_interval, args.threads) < 1:
        p.error("steps, batch size, evaluation interval and threads must be positive")
    if args.eval_sample < 0:
        p.error("eval-sample must not be negative")
    if args.train:
        train(args)
    else:
        torch.set_num_threads(args.threads)
        device = choose_device(args.device)
        model = load(args.checkpoint, device)
        if args.evaluate:
            training, validation = create_splits()
            report(model, training, "training_pairs", device, args.eval_sample)
            report(model, validation, "heldout_pairs", device, args.eval_sample)
        else:
            for question in args.question:
                m = QUESTION.fullmatch(question)
                if not m:
                    raise ValueError("only two-digit nonnegative addition like 3+3 is supported")
                a, b = map(int, m.groups())
                answer, raw = predict(model, a, b)
                print(f"QUESTION={question!r} MODEL_PREDICTION={answer!r} "
                      f"REVERSED_DIGITS={raw!r}")


if __name__ == "__main__":
    main()
