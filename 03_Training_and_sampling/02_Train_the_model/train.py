"""Validate FineWeb token IDs and train a tiny model on CPU."""

import csv
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
import torch
from torch.nn import functional as F

from tiny_transformer import TinyTransformer, stream_sample


ROOT = Path(__file__).resolve().parent
TOKENIZER = "o200k_base"
SEED = 42
SAMPLE_SEED = 123
TRAIN_SECONDS = 60
CONTEXT_LENGTH = 48
BATCH_SIZE = 8


def load_tokens(corpus, encoding):
    expected = encoding.encode_ordinary(corpus)
    path = ROOT.parent / "01_Prepare_the_input" / "fineweb_tokens.txt"
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not re.fullmatch(r"[0-9]+(?:(?:\s*,\s*|\s+)[0-9]+)*", text):
            raise ValueError("Expected a list of integer token IDs")
        saved = [int(value) for value in re.findall(r"[0-9]+", text)]
    except (FileNotFoundError, ValueError):
        saved = None

    if saved != expected:
        # Preserve the original until the entire replacement has been written.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected) + "\n", encoding="utf-8")
        temporary.replace(path)
        print(f"Regenerated {path.name} ({len(expected):,} tokens).", flush=True)
    else:
        print(f"Validated {path.name} ({len(expected):,} tokens).", flush=True)
    return expected


def choose_prompt(corpus, encoding, tokens, to_compact):
    # Keep the space because it can be part of the tokenizer's representation.
    candidates = [" Don't care about"]
    candidates.extend(encoding.decode(tokens[:length]) for length in range(6, 0, -1))
    for prompt in candidates:
        ids = encoding.encode_ordinary(prompt)
        if prompt and corpus.find(prompt) >= 0 and ids and all(i in to_compact for i in ids):
            return prompt, [to_compact[i] for i in ids]
    raise ValueError("Could not find a short prompt in the training vocabulary.")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    batch_rng = torch.Generator().manual_seed(SEED)
    corpus = (ROOT.parent / "01_Prepare_the_input" / "fineweb_corpus.txt").read_text(encoding="utf-8")
    encoding = tiktoken.get_encoding(TOKENIZER)
    tokens = load_tokens(corpus, encoding)
    if len(tokens) <= CONTEXT_LENGTH:
        raise ValueError(f"The corpus must contain at least {CONTEXT_LENGTH + 1} tokens.")

    original_ids = sorted(set(tokens))
    to_compact = {token: index for index, token in enumerate(original_ids)}
    data = torch.tensor([to_compact[token] for token in tokens], dtype=torch.long)
    prompt, prompt_ids = choose_prompt(corpus, encoding, tokens, to_compact)
    config = dict(vocab_size=len(original_ids), context_length=CONTEXT_LENGTH,
                  width=64, heads=4, layers=2)
    model = TinyTransformer(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    offsets = torch.arange(CONTEXT_LENGTH + 1)
    losses = []
    elapsed = 0.0

    print("step | loss | elapsed time (s) | tokens/second", flush=True)
    with (ROOT.parent / "03_Check_training" / "training_log.csv").open("w", encoding="utf-8", newline="") as log, \
            (ROOT.parent / "03_Check_training" / "generated_samples.txt").open("w", encoding="utf-8") as samples:
        writer = csv.writer(log)
        writer.writerow(["step", "loss", "elapsed_seconds", "tokens_per_second"])
        log.flush()

        def sample(label):
            header = f"\n[{label}, seed {SAMPLE_SEED}]\n{prompt}"
            print(header, end="", flush=True)
            samples.write(header)
            for fragment in stream_sample(model, prompt_ids, original_ids, encoding, SAMPLE_SEED):
                print(fragment, end="", flush=True)
                samples.write(fragment)
            print(flush=True)
            samples.write("\n")
            samples.flush()
            model.train()

        sample("before training")
        while elapsed < TRAIN_SECONDS:
            started = perf_counter()
            starts = torch.randint(len(data) - CONTEXT_LENGTH, (BATCH_SIZE,), generator=batch_rng)
            batch = data[starts[:, None] + offsets]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, len(original_ids)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            # The budget and throughput count updates, excluding the sample runs.
            elapsed += perf_counter() - started
            losses.append(loss_value)
            step = len(losses)
            rate = step * BATCH_SIZE * CONTEXT_LENGTH / elapsed
            row = (step, f"{loss_value:.6f}", f"{elapsed:.3f}", f"{rate:.1f}")
            print(" | ".join(map(str, row)), flush=True)
            writer.writerow(row)
            log.flush()
            if step % 20 == 0:
                sample(f"step {step}")
        sample(f"end of training, step {step}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "original_token_ids": original_ids,
        "tokenizer": TOKENIZER,
        "step": step,
    }, ROOT.parent / "03_Check_training" / "tiny_transformer.pt")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, step + 1), losses, linewidth=0.9)
    ax.set(xlabel="Training update", ylabel="Next-token cross-entropy", title="CPU training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT.parent / "03_Check_training" / "loss_plot.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
