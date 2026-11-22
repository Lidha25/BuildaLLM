"""Complete 'The sky is' 100 times using the locally trained model."""

import sys
from pathlib import Path

import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_Train_the_model"))
from tiny_transformer import TinyTransformer, stream_sample


ROOT = Path(__file__).resolve().parent


def load_model():
    torch.set_num_threads(2)
    checkpoint = torch.load(
        ROOT.parent / "03_Check_training" / "tiny_transformer.pt", map_location="cpu", weights_only=True
    )
    model = TinyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    encoding = tiktoken.get_encoding(checkpoint["tokenizer"])
    original_ids = checkpoint["original_token_ids"]
    vocabulary = {token: index for index, token in enumerate(original_ids)}
    return model, encoding, original_ids, vocabulary


def save_completions(path, prompt, prompt_ids, model, encoding, original_ids, sentence=None):
    with path.open("w", encoding="utf-8") as output:
        if sentence is not None:
            output.write(f"Chosen sentence: {sentence}\nBeginning: {prompt}\n\n")
        for number in range(1, 101):
            continuation = "".join(
                stream_sample(model, prompt_ids, original_ids, encoding, seed=number)
            )
            output.write(f"{number:03d}. {prompt}{continuation}\n")
    print(f"Saved 100 completions to {path.name}")


def main():
    model, encoding, original_ids, vocabulary = load_model()
    prompt = "The sky is"
    tokens = encoding.encode_ordinary(prompt)
    missing = [token for token in tokens if token not in vocabulary]
    if missing:
        raise SystemExit(
            f"The checkpoint is missing token IDs {missing} for {prompt!r}. "
            "Run sample_corpus.py to test a prompt from the corpus."
        )
    save_completions(
        ROOT / "sky_completions.txt", prompt, [vocabulary[token] for token in tokens],
        model, encoding, original_ids,
    )


if __name__ == "__main__":
    main()
