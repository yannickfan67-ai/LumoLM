# LumoLM Workbench — unified local interface

This brings three separately trained PyTorch GPT checkpoints under ONE interface. Their tensors are not averaged or merged, which would corrupt each model's ability.

Models:
- checkpoints/probe.pt — toy bilingual FAQ; may give incorrect, nonsensical answers.
- checkpoints/story.pt — English story continuation; not a chat model.
- checkpoints/addition.pt — neural integer addition for two operands 0..99, previously scored 828/829 on held-out unordered pairs. The 0+0 case was wrong in that evaluation.

Deterministic local tools, always marked "tool:", provide safe arithmetic, notes, file listing, file statistics, case-insensitive text search, and first-lines extractive preview. Tool answers do NOT demonstrate neural reasoning.

## Automatic task routing and Python code generation (new)

Natural language is now routed to a small, auditable task controller. Examples:

    You: 幫我寫一個 Python 斐波那契函數，保存為 fib.py
    [action:route] code
    [action:generate] audited template fibonacci, not a neural code model
    [action:static-check] Python AST valid
    [action:save] new workspace file fib.py (not executed)

    You: 計算 (12+7)*3
    [tool:calculator] 57

    You: 記下 我需要測試 LumoLM
    [tool:notes] saved note #1

    You: 檢查 Python fib.py
    [tool:syntax-check] {"valid":true,...}

    You: /trace

The controller supports English, Traditional and Simplified Chinese cues for
known tasks and exposes its routing decisions through visible `action:`
messages (JSON API responses include a `trace` array). `/trace` shows
the most recent routing steps. Use `--no-trace` to suppress progress lines
without affecting execution.

**Code generation is limited and deterministic**: the supported Python
recipes are Fibonacci, factorial, prime check, palindrome, ordered
deduplication, numeric sort, word count, line statistics, JSON reader, and
CSV-to-JSON converter. It is not a trained neural coding assistant and cannot
understand arbitrary requirements. The source is always labeled
`template:python`. New generated files are saved only inside the
`--workspace`, checked via Python AST without execution, and never
overwrite existing files. `/check script.py` performs syntax inspection
without importing or executing the script. Unknown coding tasks are explicitly
refused; no invented code or external shell calls.

Other auto-routed intents include calculation, story continuation, notes and
file statistics/search. All tool results remain clearly distinguished from
experimental neural outputs. A neural addition result can be explicitly
requested via `/add 3+3` for comparison; reliable normal arithmetic goes
to `tool:calculator`.

## Start

From this repository or integrated ZIP:

    python3 -m pip install -r requirements.txt
    python3 lumolm_workbench.py --workspace ./my-work --device cpu --threads 2

The integrated ZIP includes all 3 checkpoints. If running a source clone, supply checkpoint paths explicitly if they are not in ./checkpoints.

## Commands

    /help
    /models
    3+3
    /add 0+0
    /calc (12+7)*3
    /story Once upon a time
    /chat 你好，你是誰？
    /notes add remember this
    /notes list
    /files
    /stats notes.txt
    /find notes.txt | hello
    /summary notes.txt
    /quit

Script-friendly one shot:

    python3 lumolm_workbench.py --workspace ./my-work --message "3+3" --message "/calc 4*5" --message "/models" --json

In automatic mode, plain A+B is calculated with the reliable calculator; if the addition weight exists, its separate neural prediction is displayed as a cross-check. The program warns when the learned answer disagrees with tool verification instead of secretly correcting neural output. Other plain numerical expressions use the calculator. General prose uses the low-quality toy FAQ model. The story model is used ONLY with /story.

The /stats, /summary and /find tools access only relative text files inside --workspace, maximum 1 MiB per file; symlinks to outside are blocked. Notes are saved at --workspace/.lumolm/notes.json, and can only be cleared on explicit request. This program does not browse the web, execute shell commands, write arbitrary files, or access private directories.

## Limitations

There is NO Lumo-270M checkpoint in this bundle. These are three small experimental networks, not a single jointly-trained multitask language model. Free chat and story outputs may be incorrect. /summary is a deterministic first-lines preview, NOT genuine neural summarization. No capability for subtraction or multiplication has been taught to the addition model; the calculator handles those operations separately.

### More training data and research checkpoints

A separate experimental workflow trains a new, bigger English
story-continuation model on 16 MiB of TinyStories for 12,000 steps,
and a small Python syntax-completion model on 3 MB of procedurally
generated original examples for 8,000 steps. The models are separately
tagged and evaluated; neither can be silently merged into the
existing general Workbench weight files. Run them explicitly through
`python -m lumolm.generate --checkpoint <file> --prompt "def "`.
See README for data provenance, validation and limitations.
