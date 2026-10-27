"""
Timing harness for building a `Corpus` from raw text, stage by stage.

    python bench/bench_corpus.py [n_sentences] [repeats]

Times the three serial stages of `Corpus.from_sents` separately -- tokenizing
(`tokenize_texts_v2`), building the vocabulary (`Vocab`) and mapping tokens to
ids -- on synthetic text, reproducible from `SEED`. The text is not
`bench_pair_counts.py`'s three-letter words: tokenizer cost depends on what it
has to rewrite, so the words carry capitals, hyphens, clitics, digits and
non-ASCII letters, and sentences end in punctuation.

Each row prints `best` and `mean` of `repeats` runs, plus a `checksum` of the
output; a revision that moves the checksum changed the answer, and
`tests/test_preprocessing.py` / `tests/test_vocab.py` are the gates. Compare two
revisions the way `bench_pair_counts.py` describes, alternating the order.
"""

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperhyper.corpus import Corpus, TransformToIndicesClosure, Vocab
from hyperhyper.preprocessing import tokenize_texts_v2

SEED = 20261027
DEFAULT_SENTENCES = 120000
DEFAULT_REPEATS = 3
SENTENCE_LENGTH = 14
VOCAB_SIZE = 30000
ZIPF_EXPONENT = 1.07
SYLLABLES = ["ka", "ri", "to", "men", "sch", "ü", "ber", "an", "de", "st", "ion", "é"]
ENDINGS = [".", ",", "!", ";", " (x)", ""]


def build_texts(n_sentences):
    rng = np.random.default_rng(SEED)
    words = []
    for i in range(VOCAB_SIZE):
        picks = rng.integers(len(SYLLABLES), size=1 + i % 4)
        w = "".join(SYLLABLES[p] for p in picks)
        if i % 31 == 0:
            w = w.capitalize()
        if i % 47 == 0:
            w += "-" + SYLLABLES[i % len(SYLLABLES)]
        if i % 59 == 0:
            w += "'s"
        if i % 50 == 0:
            w += str(i % 97)
        words.append(w)
    words = np.array(words)
    weights = 1.0 / np.arange(1, VOCAB_SIZE + 1) ** ZIPF_EXPONENT
    drawn = rng.choice(
        VOCAB_SIZE, size=(n_sentences, SENTENCE_LENGTH), p=weights / weights.sum()
    )
    return [
        " ".join(row) + ENDINGS[i % len(ENDINGS)] for i, row in enumerate(words[drawn])
    ]


def timed(fn, repeats):
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        runs.append(time.perf_counter() - start)
    return result, min(runs), sum(runs) / len(runs)


def checksum(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:12]


def main(argv):
    n_sentences = int(argv[1]) if len(argv) > 1 else DEFAULT_SENTENCES
    repeats = int(argv[2]) if len(argv) > 2 else DEFAULT_REPEATS
    texts = build_texts(n_sentences)
    print(f"{n_sentences} sentences, {sum(map(len, texts))} chars, repeats={repeats}")

    tokens, *t_tok = timed(lambda: tokenize_texts_v2(texts), repeats)
    vocab, *t_voc = timed(lambda: Vocab(tokens), repeats)
    corpus = Corpus.__new__(Corpus)
    corpus.vocab = vocab
    to_indices = TransformToIndicesClosure(corpus)
    ids, *t_ids = timed(lambda: [to_indices(t) for t in tokens], repeats)

    print(f"{'stage':12s} {'best':>8s} {'mean':>8s}  checksum")
    for label, (best, mean), out in [
        ("tokenize", t_tok, tokens),
        (
            "vocab",
            t_voc,
            [list(d.items()) for d in (vocab.token2id, vocab.cfs, vocab.dfs)],
        ),
        ("to ids", t_ids, [a.tolist() for a in ids]),
    ]:
        print(f"{label:12s} {best:7.3f}s {mean:7.3f}s  {checksum(out)}")


if __name__ == "__main__":
    main(sys.argv)
