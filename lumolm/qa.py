"""Question/Answer supervision for the experimental byte-level LumoLM.

The standard language-model objective predicts question text and boilerplate
as well as answers. This optional objective masks the prompt in the loss so
the model spends its small capacity learning the answer conditional on a
question. It does not confer mathematical reasoning or factual grounding.
"""
import re

import torch


_RECORD = re.compile(rb"(?m)^Question: ([^\r\n]+)\r?\nAnswer: ([^\r\n]*)\r?$")


def parse_qa_records(raw: bytes, block_size: int):
    """Return [(input token IDs, first answer target offset), ...].

    Each example is self-contained, preventing next-record leakage.
    Oversized examples are omitted, never silently truncated into a target
    that has lost its question. Returns a count of skipped examples.
    """
    if block_size < 4:
        raise ValueError("QA block size must be at least 4")
    examples = []
    skipped = 0
    matched = 0
    for m in _RECORD.finditer(raw):
        matched += 1
        question, answer = m.group(1), m.group(2)
        prefix = b"Question: " + question + b"\nAnswer:"
        sequence = prefix + answer + b"\n"
        if len(sequence) < 3 or len(sequence) > block_size + 1:
            skipped += 1
            continue
        # Store full sequence so targets are shifted by exactly one.
        examples.append((torch.tensor(list(sequence), dtype=torch.long),
                         len(prefix) - 1))
    if not matched:
        raise ValueError("QA corpus must contain Question:/Answer: pairs")
    if not examples:
        raise ValueError("all QA records exceed the context window")
    return examples, skipped


def qa_batch(examples, batch_size: int, block_size: int, device: str):
    """Pad within a minibatch and mask targets at prompt/padded positions."""
    choices = torch.randint(len(examples), (batch_size,))
    selected = [examples[int(i)] for i in choices]
    longest = max(len(tokens) - 1 for tokens, _ in selected)
    if longest > block_size:
        raise ValueError("a QA record exceeds block_size")
    x = torch.zeros((batch_size, longest), dtype=torch.long)
    y = torch.full((batch_size, longest), -100, dtype=torch.long)
    for row, (tokens, first_target) in enumerate(selected):
        n = len(tokens) - 1
        x[row, :n] = tokens[:-1]
        y[row, first_target:n] = tokens[first_target + 1:]
    return x.to(device), y.to(device)
