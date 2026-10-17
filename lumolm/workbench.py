"""LumoLM Workbench: one interface for separate tiny models and local tools.

Model weights are NEVER averaged/merged. Roles are explicit, lazy-loaded,
task-tagged when possible, and traced to their own checkpoint.
File tools are restricted to an explicit workspace. No shell or web execution.
"""
import argparse
import ast
import json
import math
import operator
import re
import tempfile
from collections import Counter
from pathlib import Path

import torch

from .addition import QUESTION, TASK as ADDITION_TASK, load as load_addition, predict
from .chat import LumoChat
from .model import GPT, GPTConfig
from .train import choose_device

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
       ast.Mod: operator.mod, ast.Pow: operator.pow}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MATH = re.compile(r"^[0-9\s.+\-*/()%]+$")
ALLOWED = {".txt", ".md", ".log", ".csv", ".json", ".py"}
HELP = """Commands:
  /help                     commands and scope
  /models                   checkpoint status
  /add 17+28                use trained neural addition (0..99 only)
  /calc (12+7)*3            exact Python arithmetic TOOL, not learned reasoning
  /story Once upon a time   TinyStories neural continuation
  /chat What is LumoLM?     toy bilingual FAQ neural reply (may be wrong)
  /notes add your text      save a local note
  /notes list               list saved notes
  /notes clear              clear notes (explicit command)
  /files                    list accessible workspace files
  /stats notes.txt          lines/words/characters/byte counts
  /find notes.txt | hello   matching lines, case-insensitive
  /summary notes.txt        extractive preview, NOT an AI summary
  /quit                     leave the terminal
Unprefixed 0..99 addition -> neural /add if loaded; other arithmetic ->
calculator. Unprefixed prose -> /chat. Nothing accesses the network or shell.
"""


def safe_calc(expr: str):
    if len(expr) > 160 or not MATH.fullmatch(expr):
        raise ValueError("only short arithmetic expressions using numbers and +-*/()% are supported")
    root = ast.parse(expr, mode="eval")
    count = 0

    def compute(node, depth=0):
        nonlocal count
        count += 1
        if depth > 18 or count > 50:
            raise ValueError("arithmetic expression too complex")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            result = UNARY[type(node.op)](compute(node.operand, depth + 1))
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a = compute(node.left, depth + 1)
            b = compute(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(b) > 10 or abs(a) > 10000):
                raise ValueError("exponent too large")
            result = OPS[type(node.op)](a, b)
        else:
            raise ValueError("unsupported operation")
        if isinstance(result, complex) or not math.isfinite(result) or abs(result) > 1_000_000_000_000:
            raise ValueError("result exceeds safe numeric range")
        return result

    result = compute(root.body)
    return str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)


class Workbench:
    def __init__(self, workspace: Path, chat_checkpoint: Path | None = None,
                 story_checkpoint: Path | None = None,
                 addition_checkpoint: Path | None = None, device: str = "cpu",
                 max_new_tokens: int = 100):
        self.root = workspace.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "chat": Path(chat_checkpoint) if chat_checkpoint else Path("checkpoints/probe.pt"),
            "story": Path(story_checkpoint) if story_checkpoint else Path("checkpoints/story.pt"),
            "addition": Path(addition_checkpoint) if addition_checkpoint else Path("checkpoints/addition.pt"),
        }
        self.paths = {k: p.expanduser().resolve() for k, p in self.paths.items()}
        self.device = choose_device(device)
        self.max_new_tokens = max_new_tokens
        self.models = {}
        self.notes_path = self.root / ".lumolm" / "notes.json"

    @staticmethod
    def response(text: str, source: str):
        return {"text": text, "source": source}

    def model(self, role: str):
        if role in self.models:
            return self.models[role]
        path = self.paths[role]
        if not path.is_file():
            raise ValueError(f"{role} checkpoint missing: {path}; model not loaded")
        if role == "addition":
            model = load_addition(path, self.device)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved.get("format_version") != 1 or saved.get("task") == ADDITION_TASK:
                raise ValueError(f"invalid {role} text-generation checkpoint: {path}")
            model = GPT(GPTConfig(**saved["config"]))
            model.load_state_dict(saved["model"])
            model.to(self.device).eval()
            if role == "chat":
                model = LumoChat(model, max_new_tokens=self.max_new_tokens)
        self.models[role] = model
        return model

    def _file(self, relative: str) -> Path:
        p = Path(relative.strip())
        if not p.parts or p.is_absolute():
            raise ValueError("use a relative workspace filename")
        target = (self.root / p).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError("file path escapes workspace")
        if target.suffix.lower() not in ALLOWED:
            raise ValueError("only text-based files (.txt/.md/.log/.csv/.json/.py) are supported")
        if not target.is_file():
            raise ValueError(f"file not found: {relative}")
        if target.stat().st_size > 1_000_000:
            raise ValueError("file exceeds the 1 MB safe read limit")
        return target

    def _text(self, name: str):
        return self._file(name).read_text(encoding="utf-8")

    def _notes(self):
        if not self.notes_path.exists():
            return []
        result = json.loads(self.notes_path.read_text(encoding="utf-8"))
        if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
            raise ValueError("notes file is not a string list")
        return result

    def _save_notes(self, values):
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=self.notes_path.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(values, f, ensure_ascii=False, indent=2)
        tmp.replace(self.notes_path)

    def handle(self, message: str):
        message = message.strip()
        if not message:
            return self.response("Empty message.", "system")
        if message == "/help":
            return self.response(HELP, "system")
        if message == "/models":
            status = []
            for role, path in self.paths.items():
                status.append(f"{role}: {'available' if path.is_file() else 'missing'} ({path})")
            return self.response("\n".join(status), "system")
        if message == "/files":
            files = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                           if p.is_file() and p.suffix.lower() in ALLOWED
                           and ".lumolm" not in p.parts)
            return self.response("\n".join(files[:100]) or "(no text files)", "tool:file-list")

        cmd, _, value = message.partition(" ")
        if cmd == "/calc":
            return self.response(safe_calc(value.strip()), "tool:calculator")
        if cmd == "/add" or (not message.startswith("/") and QUESTION.fullmatch(message)):
            query = value if cmd == "/add" else message
            match = QUESTION.fullmatch(query)
            if not match:
                raise ValueError("learned addition requires A+B with both operands 0..99")
            a, b = (int(x) for x in match.groups())
            result, digits = predict(self.model("addition"), a, b)
            if result is None:
                return self.response(f"model emitted invalid digits: {digits!r}",
                                     "neural:addition:unverified")
            # Report neural prediction unchanged; don't secretly overwrite
            # with the reference tool's result.
            reference = a + b
            note = "" if result == reference else f" [WARNING: tool verification disagrees, {reference}]"
            return self.response(f"{result}{note}", "neural:addition")
        if cmd == "/story":
            if not value.strip():
                raise ValueError("supply a story beginning")
            model = self.model("story")
            raw = value.strip().encode("utf-8")
            completed = model.generate(raw, self.max_new_tokens, temperature=0.8)
            return self.response(completed.decode("utf-8", "replace"), "neural:story")
        if not message.startswith("/") and MATH.fullmatch(message) and any(ch in message for ch in "+-*/%"):
            return self.response(safe_calc(message), "tool:calculator")
        if cmd == "/chat" or not message.startswith("/"):
            query = value if cmd == "/chat" else message
            return self.response(self.model("chat").ask(query), "neural:toy-chat")
        if cmd == "/notes":
            action, _, data = value.partition(" ")
            notes = self._notes()
            if action == "add":
                if not data.strip() or len(data) > 2000:
                    raise ValueError("note must be 1-2000 characters")
                notes.append(data.strip())
                self._save_notes(notes[-500:])
                return self.response(f"saved note #{len(notes)}", "tool:notes")
            if action == "list":
                return self.response("\n".join(f"{i+1}. {t}" for i, t in enumerate(notes))
                                     or "(no notes)", "tool:notes")
            if action == "clear":
                self._save_notes([])
                return self.response("notes cleared on explicit request", "tool:notes")
            raise ValueError("use /notes add TEXT, /notes list, or /notes clear")
        if cmd == "/stats":
            text = self._text(value)
            stats = {"lines": len(text.splitlines()), "words": len(text.split()),
                     "characters": len(text), "utf8_bytes": len(text.encode("utf-8"))}
            return self.response(json.dumps(stats, ensure_ascii=False), "tool:text-stats")
        if cmd == "/find":
            name, sep, phrase = value.partition("|")
            if not sep or not phrase.strip():
                raise ValueError("use /find filename.txt | search phrase")
            lines = self._text(name).splitlines()
            matches = [f"{i}: {line[:180]}" for i, line in enumerate(lines, 1)
                       if phrase.strip().casefold() in line.casefold()]
            return self.response("\n".join(matches[:30]) or "(no matches)", "tool:literal-search")
        if cmd == "/summary":
            text = self._text(value)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Deliberately named a preview, not a semantic LLM summary.
            preview = "\n".join(lines[:5])[:750]
            return self.response("Extractive first-lines preview:\n" + (preview or "(empty)"),
                                 "tool:extractive-preview")
        raise ValueError("unknown command; enter /help")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("."))
    p.add_argument("--chat-checkpoint", type=Path)
    p.add_argument("--story-checkpoint", type=Path)
    p.add_argument("--addition-checkpoint", type=Path)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--message", action="append", help="run one task, repeatable")
    p.add_argument("--json", action="store_true", help="JSON lines with source tags")
    args = p.parse_args(argv)
    if args.threads < 1 or args.max_new_tokens < 1:
        p.error("threads and max-new-tokens must be positive")
    torch.set_num_threads(args.threads)
    hub = Workbench(args.workspace, args.chat_checkpoint, args.story_checkpoint,
                    args.addition_checkpoint, args.device, args.max_new_tokens)

    def run(message):
        try:
            result = hub.handle(message)
        except (ValueError, OSError, ZeroDivisionError, SyntaxError, OverflowError) as exc:
            result = {"text": str(exc), "source": "error"}
        if args.json:
            print(json.dumps({"request": message, **result}, ensure_ascii=False))
        else:
            print(f"[{result['source']}] {result['text']}")

    if args.message:
        for task in args.message:
            run(task)
        return
    print("LumoLM Workbench | separate neural models + safe local tools | /help | /quit")
    while True:
        try:
            command = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if command.strip() in ("/quit", "exit", "quit"):
            break
        run(command)


if __name__ == "__main__":
    main()
