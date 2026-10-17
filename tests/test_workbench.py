"""End-to-end safe routing for LumoLM Workbench."""
import json
from pathlib import Path

import pytest
from lumolm.workbench import Workbench, safe_calc


def test_safe_calculator_and_disallowed_expressions():
    assert safe_calc("(12 + 7) * 3") == "57"
    assert safe_calc("7/2") == "3.5"
    assert safe_calc("3**4") == "81"
    for val in ("__import__('os').system('true')", "100**10000",
                "5/0", "2 + abs(3)", "x+1", "[1]"):
        with pytest.raises((ValueError, SyntaxError, ZeroDivisionError)):
            safe_calc(val)


def test_all_local_tools_are_workspace_scoped(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    other = tmp_path / "secret.txt"
    other.write_text("SECRET", encoding="utf-8")
    doc = root / "notes.txt"
    doc.write_text("Hello world\nOther line\nHello twice\n", encoding="utf-8")
    hub = Workbench(root)
    assert hub.handle("/calc 9*7") == {"text": "63", "source": "tool:calculator"}
    assert '"lines": 3' in hub.handle("/stats notes.txt")["text"]
    assert "1: Hello world" in hub.handle("/find notes.txt | hello")["text"]
    assert "3: Hello twice" in hub.handle("/find notes.txt | hello")["text"]
    assert "Hello world" in hub.handle("/summary notes.txt")["text"]
    assert "notes.txt" in hub.handle("/files")["text"]
    with pytest.raises(ValueError, match="escapes workspace"):
        hub.handle("/stats ../secret.txt")
    with pytest.raises(ValueError, match="relative"):
        hub.handle(f"/stats {other}")
    # Symbolic links resolving outside the workspace are blocked too.
    (root / "symlink.txt").symlink_to(other)
    with pytest.raises(ValueError, match="escapes"):
        hub.handle("/stats symlink.txt")


def test_notes_persist_and_do_not_need_model(tmp_path: Path):
    hub = Workbench(tmp_path)
    assert hub.handle("/notes list")["text"] == "(no notes)"
    assert "saved note" in hub.handle("/notes add 測試筆記")["text"]
    assert "測試筆記" in Workbench(tmp_path).handle("/notes list")["text"]
    data = json.loads((tmp_path / ".lumolm" / "notes.json").read_text("utf-8"))
    assert data == ["測試筆記"]
    assert "cleared" in hub.handle("/notes clear")["text"]
    assert hub.handle("/notes list")["text"] == "(no notes)"


def test_routing_neural_models_without_faking_outputs(monkeypatch, tmp_path):
    hub = Workbench(tmp_path)
    class Addition:
        pass
    class Story:
        def generate(self, prompt, n_tokens, temperature):
            assert prompt == b"Once upon a time"
            return prompt + b", a fox ran."
    class Chat:
        def ask(self, prompt):
            return "toy answer:" + prompt

    monkeypatch.setattr(hub, "model", lambda role: {
        "addition": Addition(), "story": Story(), "chat": Chat()}[role])
    monkeypatch.setattr("lumolm.workbench.predict",
                        lambda model, a, b: (99, "990"))
    # Unprefixed math is now transparently routed to the reliable calculator.
    ans = hub.handle("3+3")
    assert ans["source"] == "tool:calculator"
    assert ans["text"] == "6"
    assert ans["trace"][0]["detail"] == "calculate"
    # Explicit neural testing still shows the raw wrong model prediction.
    neural = hub.handle("/add 3+3")
    assert neural["source"] == "neural:addition"
    assert neural["text"].startswith("99")
    assert "WARNING" in neural["text"]
    assert hub.handle("/calc 3+3") == {"text": "6", "source": "tool:calculator"}
    assert hub.handle("3*3")["text"] == "9"
    assert hub.handle("/chat Hello")["source"] == "neural:toy-chat"
    assert hub.handle("/story Once upon a time")["text"].endswith("a fox ran.")


def test_missing_model_is_explicit(tmp_path):
    hub = Workbench(tmp_path)
    assert "missing" in hub.handle("/models")["text"]
    with pytest.raises(ValueError, match="checkpoint missing"):
        hub.handle("/add 1+1")
    with pytest.raises(ValueError, match="checkpoint missing"):
        hub.handle("/story Here")
