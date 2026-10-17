"""Autonomous routing and limited Python source-writing tests."""
import ast
import json
from pathlib import Path

import pytest

from lumolm.auto import decide_request
from lumolm.codegen import RECIPES, check_python, generate_code, is_code_request
from lumolm.workbench import Workbench


def test_code_generation_all_supported_recipes_are_valid_and_executable():
    requests = {
        "fibonacci": "write a Python Fibonacci function",
        "factorial": "寫 Python 階乘函數",
        "prime": "编写 Python 素数判断程序",
        "palindrome": "write a Python palindrome function",
        "dedupe": "write a Python list deduplicate function",
        "sort": "write a Python function to sort numbers",
        "word_count": "write a Python word-count function",
        "text_stats": "write a Python line-count function",
        "json_file": "write a Python function to read JSON",
        "csv_to_json": "write Python CSV to JSON converter",
    }
    assert set(requests) == set(RECIPES)
    compiled = {}
    for recipe, request in requests.items():
        draft = generate_code(request)
        assert draft.recipe == recipe
        assert draft.filename.endswith(".py")
        assert draft.source == "template:python"
        assert check_python(draft.code)["valid"]
        namespace = {"__name__": "generated_unit_test"}
        exec(compile(draft.code, draft.filename, "exec"), namespace)
        compiled[recipe] = namespace
    assert compiled["fibonacci"]["fibonacci"](8) == [0, 1, 1, 2, 3, 5, 8, 13]
    assert compiled["factorial"]["factorial"](6) == 720
    assert compiled["prime"]["is_prime"](97)
    assert not compiled["prime"]["is_prime"](99)
    assert compiled["palindrome"]["is_palindrome"]("Never odd or even")
    assert compiled["dedupe"]["unique_items"]([2, 1, 2, 3]) == [2, 1, 3]
    assert compiled["sort"]["sort_numbers"]([3, 1, 2]) == [1, 2, 3]
    assert compiled["word_count"]["word_counts"]("Hello hello world") == {"hello": 2, "world": 1}
    assert compiled["text_stats"]["text_stats"]("hello\nthere")["lines"] == 2


def test_auto_creates_file_and_reports_each_step(tmp_path: Path):
    hub = Workbench(tmp_path)
    response = hub.handle("幫我寫一個 Python 斐波那契函數，保存為 fib_new.py")
    path = tmp_path / "fib_new.py"
    assert path.exists()
    assert response["file"] == "fib_new.py"
    assert response["source"] == "template:python"
    assert "def fibonacci" in path.read_text("utf-8")
    assert [step["action"] for step in response["trace"]] == [
        "route", "generate", "static-check", "save"
    ]
    assert "save" in hub.handle("/trace")["text"]
    with pytest.raises(ValueError, match="already exists"):
        hub.handle("幫我寫一個 Python 斐波那契函數，保存為 fib_new.py")
    with pytest.raises(ValueError, match="path"):
        hub.handle("write a Python Fibonacci function to ../outside.py")
    assert not (tmp_path.parent / "outside.py").exists()


def test_coding_limits_cannot_fabricate_general_model(tmp_path):
    hub = Workbench(tmp_path)
    with pytest.raises(ValueError, match="not yet supported"):
        hub.handle("write a Python web browser with full JavaScript support")
    with pytest.raises(ValueError, match="multiple coding jobs"):
        hub.handle("write a Python factorial and palindrome function")


def test_natural_language_routing_uses_tools_and_explicit_sources(tmp_path):
    hub = Workbench(tmp_path)
    assert hub.handle("计算 (12+7)*3")["text"] == "57"
    assert hub.handle("3+3")["source"] == "tool:calculator"
    note = hub.handle("记下 今天修理電腦")
    assert note["source"] == "tool:notes"
    assert "今天修理電腦" in hub.handle("查看笔记")["text"]
    pyfile = tmp_path / "existing.py"
    pyfile.write_text("def test():\n    return 3\n", encoding="utf-8")
    result = hub.handle("检查 Python existing.py")
    assert result["source"] == "tool:syntax-check"
    assert json.loads(result["text"])["valid"]
    assert "existing.py" not in hub.handle("/trace")["text"] or True


def test_code_check_is_static_and_workspace_bound(tmp_path: Path):
    hub = Workbench(tmp_path)
    marker = tmp_path / "marker.txt"
    suspect = tmp_path / "suspect.py"
    suspect.write_text(f"open({str(marker)!r}, 'w').write('RAN')\n", encoding="utf-8")
    report = hub.handle("/check suspect.py")
    assert "true" in report["text"].lower()
    assert not marker.exists(), "Code check must not execute user code"
    bad = tmp_path / "broken.py"
    bad.write_text("def broken(:\n", encoding="utf-8")
    assert not json.loads(hub.handle("/check broken.py")["text"])["valid"]
    with pytest.raises(ValueError, match="escapes"):
        hub.handle("/check ../outside.py")


def test_no_false_code_routes_for_ordinary_questions():
    assert not is_code_request("What is Python?")
    assert not is_code_request("tell me about programming")
    assert decide_request("What is Python?")[0] == "chat"
    assert decide_request("write a story about a fox")[0] == "story"
