"""Select a short corpus sentence and sample 100 continuations of its opening."""

import re
from collections import Counter

from sample_sky import ROOT, load_model, save_completions


def choose_sentence(corpus, encoding, vocabulary):
    candidates = []
    for segment in re.split(r"(?<=[.!?])\s+", corpus):
        sentence = segment.strip()
        words = list(re.finditer(r"\S+", sentence))
        if not 4 <= len(words) <= 18 or not sentence.endswith((".", "!", "?")):
            continue
        if not sentence[0].isalpha() or not sentence[0].isupper():
            continue
        beginning = sentence[:words[2].end()]
        opening = words[0].group().casefold()
        candidates.append((sentence, beginning, opening, len(words)))

    frequency = Counter(opening for _, _, opening, _ in candidates)
    candidates.sort(key=lambda entry: (-frequency[entry[2]], entry[3]))
    for sentence, beginning, _, _ in candidates:
        ids = encoding.encode_ordinary(beginning)
        if ids and all(token in vocabulary for token in ids):
            return sentence, beginning, [vocabulary[token] for token in ids]
    raise SystemExit("No short corpus sentence had a three-word opening in the saved vocabulary.")


def main():
    model, encoding, original_ids, vocabulary = load_model()
    with (ROOT.parent / "01_Prepare_the_input" / "fineweb_corpus.txt").open(encoding="utf-8", newline="") as source:
        corpus = source.read()
    sentence, beginning, prompt_ids = choose_sentence(corpus, encoding, vocabulary)
    save_completions(
        ROOT / "corpus_completions.txt", beginning, prompt_ids,
        model, encoding, original_ids, sentence=sentence,
    )


if __name__ == "__main__":
    main()
