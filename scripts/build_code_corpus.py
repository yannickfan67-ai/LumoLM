"""Generate original, diverse *small* Python completion examples.

Synthetic training text is original. This builds short code samples with
independent train/validation seeds and held-out parameter combinations.
It is NOT a substitute for a licensed code repository corpus and does
NOT establish the ability to write arbitrary software.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

TRAIN_SEED = 20261017
VAL_SEED = 20261231

# Code patterns are intentionally readable and checked with ast.parse below.
# Short samples fit the small (byte-level) context window used in experiments.
NAMES = ["x", "n", "value", "number", "item", "amount", "count", "score",
         "delta", "steps", "index", "total"]
FUNCTIONS = [
    ("offset", "return {x} + {k}", "increase {x} by {k}"),
    ("difference", "return {x} - {k}", "subtract {k} from {x}"),
    ("scale", "return {x} * {k}", "multiply {x} by {k}"),
    ("square", "return {x} * {x}", "square {x}"),
    ("absolute", "return abs({x})", "absolute value of {x}"),
    ("is_even", "return {x} % 2 == 0", "check whether {x} is even"),
    ("is_odd", "return {x} % 2 != 0", "check whether {x} is odd"),
    ("clamp", "return max({lo}, min({hi}, {x}))", "clamp {x} between {lo} and {hi}"),
    ("max_pair", "return max({x}, {y})", "maximum of two values"),
    ("min_pair", "return min({x}, {y})", "minimum of two values"),
    ("rounded", "return round({x}, {k})", "round {x} to {k} decimals"),
    ("multiply_pair", "return {x} * {y}", "product of two values"),
    ("is_positive", "return {x} > 0", "check if {x} is positive"),
    ("floor_div", "return {x} // {k}", "floor divide {x} by {k}"),
    ("remainder", "return {x} % {k}", "remainder of {x} / {k}"),
    ("repeat", "return str({x}) * {k}", "repeat text {k} times"),
    ("as_list", "return [{x}] * {k}", "make a list containing {k} copies"),
    ("as_text", "return str({x})", "convert value to a string"),
    ("increment", "return {x} + 1", "increment a number"),
    ("decrement", "return {x} - 1", "decrement a number"),
]
MAX_SAMPLE = 230


def format_example(rng: random.Random, i: int) -> str:
    key, expression, explanation = FUNCTIONS[i % len(FUNCTIONS)]
    x = rng.choice(NAMES)
    y = rng.choice([name for name in NAMES if name != x])
    k = rng.randint(2, 24)
    lo = rng.randint(-15, 5)
    hi = lo + rng.randint(10, 60)
    function_name = f"{key}_{rng.choice(('basic','simple','safe','fast','local'))}_{i}"
    # The random numeric constant and function name create parameter diversity,
    # but structures are shared and held-out scores reflect that limited scope.
    desc = explanation.format(x=x, y=y, k=k, lo=lo, hi=hi)
    body = expression.format(x=x, y=y, k=k, lo=lo, hi=hi)
    params = f"{x}, {y}" if "{y}" in expression else x
    sample = (f"# Task: Write Python to {desc}.\n"
              f"def {function_name}({params}):\n"
              f"    {body}\n\n")
    if len(sample.encode("utf-8")) > MAX_SAMPLE:
        raise ValueError("generated sample exceeds context window")
    return sample


def build(max_train_bytes: int, max_val_bytes: int):
    if not (100_000 <= max_train_bytes <= 30_000_000 and
            20_000 <= max_val_bytes <= 3_000_000):
        raise ValueError("code corpus size is outside supported safety bounds")

    def make(limit, seed, start):
        rng = random.Random(seed)
        chunks = []
        size = 0
        i = start
        seen = set()
        while size < limit - MAX_SAMPLE:
            data = format_example(rng, i)
            i += 1
            # Preserve exact train/validation record disjointness; position
            # and random seed separate most records. All items are original.
            if data in seen:
                continue
            seen.add(data)
            buf = data.encode("utf-8")
            chunks.append(buf)
            size += len(buf)
        return b"".join(chunks), seen

    train, tr_seen = make(max_train_bytes, TRAIN_SEED, 0)
    validation, va_seen = make(max_val_bytes, VAL_SEED, 10_000_000)
    if tr_seen & va_seen:
        raise ValueError("training and validation contain identical samples")
    return train, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/code_generated"))
    parser.add_argument("--train-bytes", type=int, default=3_000_000)
    parser.add_argument("--val-bytes", type=int, default=300_000)
    args = parser.parse_args(argv)
    train, val = build(args.train_bytes, args.val_bytes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.txt").write_bytes(train)
    (args.output_dir / "validation.txt").write_bytes(val)
    manifest = {
        "source": "original procedural Python code examples",
        "code_license": "generated in the LumoLM project; check repository LICENSE for redistribution",
        "generation_seed_train": TRAIN_SEED,
        "generation_seed_validation": VAL_SEED,
        "warning": "synthetic small functions; held-out variations are NOT unseen algorithms",
        "train_bytes": len(train), "val_bytes": len(val),
        "train_sha256": hashlib.sha256(train).hexdigest(),
        "val_sha256": hashlib.sha256(val).hexdigest(),
        "templates": len(FUNCTIONS),
    }
    (args.output_dir / "SOURCE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("CODE_CORPUS=" + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
