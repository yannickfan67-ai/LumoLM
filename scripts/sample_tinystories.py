"""Download bounded, independent TinyStories text samples for LM experiments.

TinyStories (roneneldan/TinyStories) is published with the CDLA-Sharing-1.0
dataset license. Read that license and preserve the source/attribution notice
if you redistribute data. The downloaded story text is NOT added to Git.
This tool trains no model by itself and deliberately limits transferred bytes.

Source: https://huggingface.co/datasets/roneneldan/TinyStories
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REPO = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
SOURCES = {
    "train": REPO + "TinyStoriesV2-GPT4-train.txt",
    "validation": REPO + "TinyStoriesV2-GPT4-valid.txt",
}
LICENSE = "cdla-sharing-1.0"
LICENSE_INFO = "https://huggingface.co/datasets/roneneldan/TinyStories"


def usable_prefix(content: bytes) -> bytes:
    """Remove any truncated UTF-8 character and partial final text line."""
    if not content:
        raise ValueError("remote source returned no data")
    sample = content.decode("utf-8", errors="ignore")
    pos = sample.rfind("\n")
    if pos < 512:
        raise ValueError("sample contains too little complete text")
    sample = sample[:pos + 1]
    if "<!doctype html" in sample[:100].lower() or "<html" in sample[:100].lower():
        raise ValueError("received HTML instead of source text")
    return sample.encode("utf-8")


def download_sample(url: str, max_bytes: int) -> bytes:
    if max_bytes < 4096 or max_bytes > 100 * 1024 * 1024:
        raise ValueError("sample limit must be between 4 KiB and 100 MiB")
    req = Request(url, headers={
        "User-Agent": "LumoLM-licensed-corpus-sampler/1.0",
        "Range": f"bytes=0-{max_bytes - 1}",
    })
    # Servers may ignore Range. Read only the capped prefix regardless.
    with urlopen(req, timeout=60) as response:
        content = response.read(max_bytes)
    return usable_prefix(content)


def materialize_samples(train_bytes: bytes, val_bytes: bytes, output: Path):
    """Persist bounded datasets + source provenance. Never merge the splits."""
    output.mkdir(parents=True, exist_ok=True)
    if train_bytes == val_bytes:
        raise ValueError("train and validation samples unexpectedly identical")
    info = {
        "source": LICENSE_INFO,
        "dataset_license": LICENSE,
        "train_url": SOURCES["train"],
        "validation_url": SOURCES["validation"],
        "redistribution_note": "Check CDLA-Sharing-1.0 conditions before redistributing.",
        "splits": {},
    }
    for split, data in (("train", train_bytes), ("validation", val_bytes)):
        file = output / f"{split}.txt"
        file.write_bytes(data)
        info["splits"][split] = {
            "filename": file.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = output / "SOURCE.json"
    manifest.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/tinystories_sample"))
    parser.add_argument("--max-train-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-val-bytes", type=int, default=256 * 1024)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == Path(".").resolve():
        parser.error("output-dir cannot be the repository root")
    train = download_sample(SOURCES["train"], args.max_train_bytes)
    valid = download_sample(SOURCES["validation"], args.max_val_bytes)
    info = materialize_samples(train, valid, args.output_dir)
    for split, details in info["splits"].items():
        print(f"{split}: bytes={details['bytes']} sha256={details['sha256']}")
    print(f"dataset license: {LICENSE}; source: {LICENSE_INFO}")


if __name__ == "__main__":
    main()
