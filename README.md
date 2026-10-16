# LumoLM

An **experimental, from-scratch byte-level GPT** implemented in PyTorch. It includes a causal Transformer, UTF-8 byte tokenizer (vocabulary of 256), local training, resumable checkpoints, text generation, unit tests, and a CPU CI smoke test.

**Current status:** Code and a very small example corpus are provided. The example can verify the full training/inference path, but it **cannot** train an assistant or useful large language model. Training a capable LLM needs much more carefully licensed data, compute and evaluation. No pretrained weights are bundled.

## Quick start

Python 3.10+ is recommended. Install PyTorch appropriate for your machine (CUDA, MPS, or CPU), then:

```bash
python -m pip install -r requirements.txt
python -m lumolm.train --data data/demo.txt --checkpoint checkpoints/demo.pt --device cpu --steps 20 --batch-size 4 --block-size 32 --n-layer 2 --n-head 2 --n-embd 64 --eval-interval 10 --eval-iters 2
python -m lumolm.generate --checkpoint checkpoints/demo.pt --prompt 'Question:' --max-new-tokens 80 --temperature 0.8
```

For a fast end-to-end check:

```bash
python -m lumolm.train --data data/demo.txt --checkpoint checkpoints/smoke.pt --device cpu --steps 2 --batch-size 2 --block-size 16 --n-layer 1 --n-head 2 --n-embd 32 --eval-iters 1 --eval-interval 1
python -m lumolm.generate --checkpoint checkpoints/smoke.pt --prompt 'Hello' --max-new-tokens 12 --temperature 0
python -m pytest -q
```

## Train your own model

Use a sufficiently large, UTF-8 **plain text** corpus you have permission to use. Train/validation are split contiguously (last 10% used for validation; at least `block_size + 2` bytes on either side). Do not add secrets or private data to your corpus. Example:

```bash
python -m lumolm.train --data /path/to/your-corpus.txt --checkpoint checkpoints/experiment.pt --device auto --steps 10000 --batch-size 16 --block-size 128 --n-layer 4 --n-head 4 --n-embd 128 --eval-interval 500 --eval-iters 20
```

To resume (the **same corpus** and existing checkpoint are required; `--steps` is the total target number of optimizer updates):

```bash
python -m lumolm.train --data /path/to/your-corpus.txt --checkpoint checkpoints/experiment.pt --resume --steps 15000
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. For CPU training, start with a small model. Use `--grad-accum N` to accumulate gradients over N mini-batches. Output checkpoints are saved atomically at evaluation intervals and at the end; they include model, optimizer, training step, RNG state, model configuration, and corpus SHA-256. Checkpoints and private datasets are git-ignored. Load **only trusted** checkpoints.

## Limitations / next steps

- Byte-level GPT is easy to inspect but less efficient than a learned subword tokenizer for large-scale training. It may produce malformed UTF-8; inference decodes malformed sequences with replacement characters.
- No instruction tuning, chat template, large-scale dataset streaming, multi-GPU training, KV-cache or distributed checkpointing yet.
- GitHub Actions runs a tiny **CPU smoke test**, not large-scale GPU training; there is no claim that training is running continuously in the cloud.
- Larger models require a proper tokenizer, evaluation benchmark, dataset pipeline and hardware plan before claiming useful LLM quality.

## License and data

No external datasets or pretrained weights are included. Check dataset licenses and privacy before training or publication. This repository's source license has not been specified by the owner yet.
