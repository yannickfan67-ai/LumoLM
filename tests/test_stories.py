"""TinyStories import must stay size-bounded and preserve attribution."""
import hashlib
import json
from io import BytesIO

import pytest

from scripts import sample_tinystories as stories


class DummyResponse(BytesIO):
    pass


def test_bounded_prefix_range_and_utf8(monkeypatch):
    sample = ("Once there was a fox.\n" * 1000).encode() + b"\xe2"
    seen = []
    def fake_open(request, timeout):
        seen.append((request.get_header("Range"), timeout, request.full_url))
        return DummyResponse(sample)
    monkeypatch.setattr(stories, "urlopen", fake_open)
    result = stories.download_sample(stories.SOURCES["train"], 4096)
    assert 512 < len(result) <= 4096
    assert result.endswith(b"\n")
    assert result.decode("utf-8")
    assert seen == [("bytes=0-4095", 60, stories.SOURCES["train"])]


def test_records_manifest_and_separate_splits(tmp_path):
    a = b"Story A.\n" * 120
    b = b"Story B.\n" * 90
    info = stories.materialize_samples(a, b, tmp_path / "corpus")
    out = tmp_path / "corpus"
    assert (out / "train.txt").read_bytes() == a
    assert (out / "validation.txt").read_bytes() == b
    data = json.loads((out / "SOURCE.json").read_text("utf-8"))
    assert info == data
    assert data["dataset_license"] == "cdla-sharing-1.0"
    assert data["splits"]["train"]["sha256"] == hashlib.sha256(a).hexdigest()
    with pytest.raises(ValueError, match="identical"):
        stories.materialize_samples(a, a, tmp_path / "same")


@pytest.mark.parametrize("payload", [b"", b"<html>oops</html>\n" * 100])
def test_bad_response_rejected(payload):
    with pytest.raises(ValueError):
        stories.usable_prefix(payload)


@pytest.mark.parametrize("length", [1, 200_000_000])
def test_download_size_bounds(length):
    with pytest.raises(ValueError, match="sample limit"):
        stories.download_sample(stories.SOURCES["train"], length)
