# LumoLM Workbench — unified local interface

This brings three separately trained PyTorch GPT checkpoints under ONE interface. Their tensors are not averaged or merged, which would corrupt each model's ability.

Models:
- checkpoints/probe.pt — toy bilingual FAQ; may give incorrect, nonsensical answers.
- checkpoints/story.pt — English story continuation; not a chat model.
- checkpoints/addition.pt — neural integer addition for two operands 0..99, previously scored 828/829 on held-out unordered pairs. The 0+0 case was wrong in that evaluation.

Deterministic local tools, always marked "tool:", provide safe arithmetic, notes, file listing, file statistics, case-insensitive text search, and first-lines extractive preview. Tool answers do NOT demonstrate neural reasoning.

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

A plain A+B with two operands in 0..99 goes to the learned model (if the addition weight file is installed), not the reliable calculator. The program warns when the learned answer disagrees with tool verification instead of secretly correcting neural output. Other plain numerical expressions use the calculator. General prose uses the low-quality toy FAQ model. The story model is used ONLY with /story.

The /stats, /summary and /find tools access only relative text files inside --workspace, maximum 1 MiB per file; symlinks to outside are blocked. Notes are saved at --workspace/.lumolm/notes.json, and can only be cleared on explicit request. This program does not browse the web, execute shell commands, write arbitrary files, or access private directories.

## Limitations

There is NO Lumo-270M checkpoint in this bundle. These are three small experimental networks, not a single jointly-trained multitask language model. Free chat and story outputs may be incorrect. /summary is a deterministic first-lines preview, NOT genuine neural summarization. No capability for subtraction or multiplication has been taught to the addition model; the calculator handles those operations separately.
