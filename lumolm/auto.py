"""Visible, bounded natural-language task routing for LumoLM Workbench.

This is a deterministic controller. Its decision record is inspectable;
it must not be represented as hidden LLM planning or autonomous reasoning.
Never execute generated scripts or arbitrary shell commands.
"""
import json
import re

from .codegen import generate_code, is_code_request, check_python

ARITH = re.compile(r"^[\d\s.+\-*/()%]+$")
SAVE_NOTE = re.compile(r"^\s*(?:記下|记下|記住|记住|記錄|记录|幫我記|帮我记|remember|note:)\s*[:：]?\s*(.+)$", re.I | re.S)
SHOW_NOTES = re.compile(r"^\s*(?:查看|列出|列舉|显示|顯示|show|list)\s*(?:所有)?\s*(?:筆記|笔记|notes?)\s*$", re.I)
CALC = re.compile(r"^\s*(?:幫我)?(?:計算|计算|算一下|算|calculate|what is)\s*[:：]?\s*([\d\s.+\-*/()%]+)\??\s*$", re.I)
STORY = re.compile(r"^\s*(?:寫|写|創作|创作|write|make)\s*(?:一個|一个|a|an)?\s*(?:短篇故事|故事|story)\s*[:：]?\s*(.*)$", re.I | re.S)
CHECK = re.compile(r"^\s*(?:檢查|检查|check|review)\s*(?:一下)?\s*(?:這個|这个)?\s*(?:Python\s*)?(?:程式|代码|代碼|code|檔案|文件)?\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]{0,60}\.py)\s*$", re.I)
FILE_COUNT = re.compile(r"^\s*(?:統計|统计|分析|analy[sz]e|stats?)\s*[:：]?\s*([\w./-]+\.(?:txt|md|log|csv|json|py))\s*$", re.I)
FILE_FIND = re.compile(r"^\s*(?:查找|尋找|查找|搜索|搜尋|find|search)\s+(.+\.(?:txt|md|log|csv|json|py))\s*(?:裡|中|for|包含|的)?\s*[|:：]\s*(.+)$", re.I)


def decide_request(message: str):
    """Return intent + payload, or fallback. Rules are intentionally limited."""
    text = message.strip()
    if is_code_request(text):
        return "code", text
    match = SAVE_NOTE.fullmatch(text)
    if match:
        return "save_note", match.group(1)
    if SHOW_NOTES.fullmatch(text):
        return "list_notes", ""
    match = CHECK.fullmatch(text)
    if match:
        return "check", match.group(1)
    match = FILE_FIND.fullmatch(text)
    if match:
        return "find", f"{match.group(1)} | {match.group(2)}"
    match = FILE_COUNT.fullmatch(text)
    if match:
        return "stats", match.group(1)
    match = CALC.fullmatch(text)
    if match:
        return "calculate", match.group(1).strip()
    if ARITH.fullmatch(text) and any(op in text for op in "+-*/%"):
        return "calculate", text
    if any(phrase in text.casefold() for phrase in ("編故事", "编故事", "寫故事", "写故事", "write a story", "once upon a time")):
        match = STORY.match(text)
        # TinyStories is an English continuation model; don't claim it can
        # follow arbitrary instructions or generate code.
        return "story", match.group(1) if match and match.group(1).strip() else "Once upon a time"
    return "chat", text


class AutoAgent:
    def __init__(self, hub):
        self.hub = hub
        self.steps = []

    def note(self, action: str, detail: str):
        self.steps.append({"action": action, "detail": detail})

    def execute(self, message: str):
        intent, payload = decide_request(message)
        self.note("route", intent)
        if intent == "code":
            draft = generate_code(payload)
            self.note("generate", f"audited template {draft.recipe}, not a neural code model")
            validation = check_python(draft.code, draft.filename)
            self.note("static-check", "Python AST valid" if validation["valid"] else "syntax error")
            if not validation["valid"]:
                raise ValueError("generated Python source failed syntax check")
            path = self.hub.root / draft.filename
            # No overwrites, even on repeat prompts. Never use symlink paths.
            if path.exists() or path.is_symlink():
                raise ValueError(f"{draft.filename} already exists; refusing to overwrite")
            if not path.resolve().is_relative_to(self.hub.root):
                raise ValueError("generated path escapes workspace")
            with path.open("x", encoding="utf-8") as stream:
                stream.write(draft.code)
            self.note("save", f"new workspace file: {draft.filename} (not executed)")
            self.hub.last_trace = self.steps
            return {
                "text": f"Created {draft.filename} ({draft.explanation}).\n"
                        f"Python syntax check: PASS. NOT EXECUTED.\n\n{draft.code}",
                "source": "template:python",
                "trace": self.steps,
                "file": draft.filename,
            }
        if intent == "calculate":
            self.note("calculate", "bounded deterministic calculator")
            result = self.hub.handle("/calc " + payload)
        elif intent == "save_note":
            self.note("write-note", "persistent note store inside workspace")
            result = self.hub.handle("/notes add " + payload)
        elif intent == "list_notes":
            self.note("read-notes", "persistent note store inside workspace")
            result = self.hub.handle("/notes list")
        elif intent == "find":
            self.note("search", "literal text search within workspace")
            result = self.hub.handle("/find " + payload)
        elif intent == "stats":
            self.note("analyze", "text statistics within workspace")
            result = self.hub.handle("/stats " + payload)
        elif intent == "check":
            self.note("static-check", f"parse Python syntax without executing {payload}")
            result = self.hub.handle("/check " + payload)
        elif intent == "story":
            self.note("neural", "TinyStories next-byte continuation; not instruction tuned")
            result = self.hub.handle("/story " + payload)
        else:
            self.note("neural", "toy QA language model; not a reliable general assistant")
            result = self.hub.handle("/chat " + payload)
        self.hub.last_trace = self.steps
        return {**result, "trace": self.steps}
