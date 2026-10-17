"""Auditable, limited Python code synthesis for LumoLM Workbench.

IMPORTANT: this module uses validated recipes and parameters, NOT an
LLM-based code model. It never executes generated scripts. A code-generation
request outside known recipes receives an explicit limitation instead of
fabricated code. Independent code-specialist model support could be added
later without misrepresenting these outputs.
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeDraft:
    recipe: str
    filename: str
    code: str
    explanation: str
    source: str = "template:python"


RECIPES = {
    "fibonacci": {
        "patterns": (r"fibonacci", r"fib\b", r"費波那契", r"斐波那契", r"斐波納契"),
        "function": "fibonacci",
        "body": '''def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers; n must be nonnegative."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    result = []
    a, b = 0, 1
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result
''',
        "example": "print(fibonacci(10))",
        "desc": "生成指定長度的 Fibonacci 數列",
    },
    "factorial": {
        "patterns": (r"factorial", r"階乘", r"阶乘"),
        "function": "factorial",
        "body": '''def factorial(n: int) -> int:
    """Return n! with input validation."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
''',
        "example": "print(factorial(6))",
        "desc": "計算非負整數階乘",
    },
    "prime": {
        "patterns": (r"prime", r"質數", r"素数", r"素數"),
        "function": "is_prime",
        "body": '''def is_prime(n: int) -> bool:
    """Test primality of an integer using trial division."""
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= n:
        if n % divisor == 0:
            return False
        divisor += 2
    return True
''',
        "example": "print(is_prime(97))",
        "desc": "判斷整數是否為質數",
    },
    "palindrome": {
        "patterns": (r"palindrome", r"回文"),
        "function": "is_palindrome",
        "body": '''def is_palindrome(text: str) -> bool:
    """Ignore punctuation and case when checking a palindrome."""
    cleaned = "".join(char.casefold() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]
''',
        "example": 'print(is_palindrome("Never odd or even"))',
        "desc": "檢查字串是否為回文",
    },
    "dedupe": {
        "patterns": (r"deduplicat", r"remove duplicates?", r"去重", r"刪除重複", r"删除重复"),
        "function": "unique_items",
        "body": '''def unique_items(items: list) -> list:
    """Remove duplicates while preserving order (works for unhashable values)."""
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
''',
        "example": "print(unique_items([2, 1, 2, 3, 1]))",
        "desc": "保持順序的清單去重",
    },
    "sort": {
        "patterns": (r"\bsort\b", r"排序"),
        "function": "sort_numbers",
        "body": '''def sort_numbers(values: list[int], reverse: bool = False) -> list[int]:
    """Return a sorted copy of the supplied numbers."""
    return sorted(values, reverse=reverse)
''',
        "example": "print(sort_numbers([9, 1, 4, 2]))",
        "desc": "數字排序，保留原清單",
    },
    "word_count": {
        "patterns": (r"word.count", r"count words?", r"詞頻", r"词频", r"單詞統計", r"单词统计"),
        "function": "word_counts",
        "body": '''import re
from collections import Counter


def word_counts(text: str) -> dict[str, int]:
    """Count case-insensitive Unicode words."""
    tokens = re.findall(r"\w+", text.casefold())
    return dict(Counter(tokens))
''',
        "example": 'print(word_counts("Hello hello world"))',
        "desc": "文字單詞頻率統計",
    },
    "text_stats": {
        "patterns": (r"line.count", r"count lines?", r"行數統計", r"行数统计", r"統計行數", r"统计行数"),
        "function": "text_stats",
        "body": '''def text_stats(text: str) -> dict[str, int]:
    """Count lines, whitespace-separated words and characters."""
    return {
        "lines": len(text.splitlines()),
        "words": len(text.split()),
        "characters": len(text),
    }
''',
        "example": 'print(text_stats("one two\\nthree"))',
        "desc": "統計文字行數、詞數和字元數",
    },
    "json_file": {
        "patterns": (r"\bjson\b.*(?:read|load|讀取|读取)", r"(?:read|load|讀取|读取).*\bjson\b"),
        "function": "read_json",
        "body": '''import json
from pathlib import Path


def read_json(filename: str) -> object:
    """Parse JSON from a UTF-8 text file."""
    with Path(filename).open(encoding="utf-8") as stream:
        return json.load(stream)
''',
        "example": '# print(read_json("example.json"))  # supply an existing JSON file',
        "desc": "讀取 JSON 檔案",
    },
    "csv_to_json": {
        "patterns": (r"csv.*json", r"csv.*(?:轉|转).*json"),
        "function": "csv_to_json",
        "body": '''import csv
import json
from pathlib import Path


def csv_to_json(csv_path: str, json_path: str) -> int:
    """Convert a CSV file to a JSON array. Returns the row count."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with Path(json_path).open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, indent=2)
    return len(rows)
''',
        "example": '# print(csv_to_json("input.csv", "output.json"))  # read/write actual files',
        "desc": "將 CSV 資料轉換為 JSON",
    },
}

# Most specific recipe first; otherwise "CSV to JSON" might be misread as JSON
# file reading. No recipe guesses for arbitrary requests.
ORDER = ("csv_to_json", "fibonacci", "factorial", "prime", "palindrome",
         "dedupe", "word_count", "text_stats", "sort", "json_file")

FILE_NAME = re.compile(r"(?<![A-Za-z0-9_/.-])([A-Za-z][A-Za-z0-9_-]{0,60}\.py)\b")
WRITE_CUES = (
    "write", "create", "generate", "build", "implement", "寫", "写",
    "編寫", "编写", "生成", "製作", "制作", "做個", "做个",
)
CODE_TARGETS = (
    "python", "script", "function", "program", "程式", "代码",
    "代碼", "函数", "函數", "腳本", "脚本",
)


def is_code_request(text: str) -> bool:
    lower = text.casefold().strip()
    if lower.startswith("/code "):
        return True
    if not any(cue in lower for cue in WRITE_CUES):
        return False
    return any(target in lower for target in CODE_TARGETS)


def choose_recipe(request: str) -> str:
    text = request.casefold()
    matches = [
        key for key in ORDER
        if any(re.search(pattern, text, re.IGNORECASE)
               for pattern in RECIPES[key]["patterns"])
    ]
    if not matches:
        raise ValueError(
            "Coding request not yet supported by the built-in templates. "
            "Supported: Fibonacci, factorial, prime check, palindrome, "
            "deduplicate, sort, word-count, line stats, read JSON, CSV to JSON. "
            "No general-purpose code LLM is connected."
        )
    if len(matches) > 1:
        # CSV/JSON conversion intentionally mentions JSON as a format.
        if matches == ["csv_to_json", "json_file"]:
            return "csv_to_json"
        raise ValueError("Request mixes multiple coding jobs; ask for one function at a time")
    return matches[0]


def generate_code(request: str) -> CodeDraft:
    recipe = choose_recipe(request)
    spec = RECIPES[recipe]
    name_match = FILE_NAME.search(request)
    # Do not silently ignore a requested directory or traversal filename.
    py_paths = re.findall(r"[^\s,;：]+\.py\b", request, re.I)
    if py_paths and (not name_match or any("/" in p or "\\\\" in p or ".." in p for p in py_paths)):
        raise ValueError("output filename must be a simple .py name, not a path")
    filename = name_match.group(1) if name_match else recipe + ".py"
    # Do not allow overrides to create a file outside the workspace.
    if len(filename) > 68 or Path(filename).name != filename:
        raise ValueError("unsafe Python filename")
    source = (
        '"""Generated by LumoLM limited, audited Python recipe engine.\n'
        'Source: template, NOT neural model. Review before running.\n'
        '"""\n\n'
        + spec["body"].rstrip() + "\n\n"
        + 'if __name__ == "__main__":\n'
        + "    " + spec["example"] + "\n"
    )
    # Validate syntax without executing user-facing generated code.
    ast.parse(source, filename=filename)
    return CodeDraft(recipe=recipe, filename=filename, code=source,
                     explanation=spec["desc"])


def check_python(source: str, filename: str = "<generated>") -> dict:
    """Pure static syntax analysis. Never import or execute submitted code."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {"valid": False, "line": exc.lineno, "error": exc.msg,
                "source": "tool:syntax-check"}
    functions = [node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    return {"valid": True, "functions": functions, "classes": classes,
            "source": "tool:syntax-check"}
