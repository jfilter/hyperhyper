"""
Tests for the tokenizers and for how `preprocessing` handles optional spaCy.

Two things are pinned here, both of which used to cost seconds per call:

* `tokenize_texts_parallel` must not spawn a process pool for small inputs --
  spawning one costs ~3s, and tokenizing an entire evaluation dataset costs
  ~0.08s -- while still using the pool for genuinely large ones.
* importing this module must not import spaCy, which costs ~1.3s warm (~2.2s
  cold) and is only needed by `texts_to_sents`. Every process-pool child
  re-imports the module and used to pay for it.
"""

import subprocess
import sys
from concurrent import futures

import pytest

from hyperhyper import preprocessing
from hyperhyper.preprocessing import (
    simple_preproc,
    texts_to_sents,
    tokenize_string,
    tokenize_texts,
    tokenize_texts_parallel,
)

TEXTS = [
    "The English Wikipedia, founded in 2001.",
    "<b>Simple</b> English uses basic vocabulary!",
    "",
    "ice-cream and 42 apples",
]


@pytest.fixture()
def pool_calls(monkeypatch):
    """
    Count `ProcessPoolExecutor` constructions and run the work serially.

    `hyperhyper.utils` does `from concurrent import futures` and then reaches
    for `futures.ProcessPoolExecutor`, so the attribute on `concurrent.futures`
    is what the code under test actually looks up.
    """
    calls = []

    class Counting:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fun, iterable, **kwargs):
            return map(fun, iterable)

    monkeypatch.setattr(futures, "ProcessPoolExecutor", Counting)
    return calls


# tokenization


def test_simple_preproc():
    assert simple_preproc("Wikipedia 2001") == "wikipedia 0000"


def test_tokenize_string_strips_tags_and_punctuation():
    assert tokenize_string("<b>Ice-cream</b>, 42!") == ["ice", "cream", "00"]


def test_tokenize_texts_parallel_matches_the_serial_tokenizer(pool_calls):
    assert tokenize_texts_parallel(TEXTS) == tokenize_texts(TEXTS)


def test_small_input_does_not_spawn_a_pool(pool_calls):
    """
    The regression this whole change is about: a pool for a handful of short
    strings is 4 orders of magnitude more expensive than the work itself.
    """
    tokenize_texts_parallel(TEXTS)

    assert pool_calls == []


def test_large_input_still_uses_the_pool(monkeypatch, pool_calls):
    """
    The threshold must not quietly delete the parallel path that large corpora
    rely on. Lowering it is equivalent to handing in a huge input, without
    building one.
    """
    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 1)

    result = tokenize_texts_parallel(TEXTS)

    assert len(pool_calls) == 1
    # ... and the pooled path still produces exactly the same tokens
    assert result == tokenize_texts(TEXTS)


def test_threshold_counts_characters_not_items(monkeypatch, pool_calls):
    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 100)

    tokenize_texts_parallel(["a b c"] * 5)  # 25 chars, way under
    assert pool_calls == []

    tokenize_texts_parallel(["x" * 60] * 2)  # 120 chars, over
    assert len(pool_calls) == 1


def test_accepts_an_iterator(pool_calls):
    """
    `eval_similarity` hands in the columns of a dataset, which are tuples, and
    callers may hand in a generator. Measuring the input must not consume it.
    """
    assert tokenize_texts_parallel(iter(TEXTS)) == tokenize_texts(TEXTS)


# optional spaCy


def test_importing_preprocessing_does_not_import_spacy():
    """
    spaCy is only needed by `texts_to_sents`. Importing it eagerly cost every
    process-pool child ~1.3s (warm) for nothing.
    """
    code = "import hyperhyper.preprocessing, sys; print('spacy' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "False"


def test_texts_to_sents_without_spacy(monkeypatch):
    """
    The optional-dependency guard, and its message, are part of the API.
    """
    monkeypatch.setattr(preprocessing, "spacy", None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        texts_to_sents(["some text"])

    assert str(excinfo.value) == (
        "spaCy is required for texts_to_sents; "
        'install with: pip install "hyperhyper[full]"'
    )


def test_a_disabled_spacy_is_not_re_imported(monkeypatch):
    """
    The lazy import must not paper over a caller (or a broken install) that has
    already settled the question by setting `spacy` to `None`.
    """
    monkeypatch.setattr(preprocessing, "spacy", None)

    assert preprocessing._import_spacy() is None
    assert preprocessing.spacy is None
