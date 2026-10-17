"""Legacy command name for LumoLM's maintained package-based chat CLI.

Run from the cloned repository: python3 lumolm_chat_cli.py
For the earlier standalone ZIP, copy/upgrade the *whole package* rather than
replacing just probe.pt. A TinyStories completion checkpoint is not chat-tuned.
"""
import sys
from pathlib import Path

from lumolm.chat import main


if __name__ == "__main__":
    if "--checkpoint" not in sys.argv:
        options = (
            Path("probe.pt"),  # earlier standalone package layout
            Path("checkpoints/probe.pt"),
            Path("checkpoints/expanded.pt"),
        )
        checkpoint = next((p for p in options if p.is_file()), None)
        if checkpoint is None:
            sys.exit(
                "LumoLM checkpoint not found. Provide --checkpoint /path/to/probe.pt "
                "or extract the synthetic-QA checkpoint to this directory. "
                "TinyStories story.pt is a text-completion model, NOT a chat model."
            )
        sys.argv.extend(("--checkpoint", str(checkpoint)))
    main()
