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
import unicodedata
from concurrent import futures

import pytest

from hyperhyper import preprocessing
from hyperhyper.preprocessing import (
    simple_preproc,
    texts_to_sents,
    tokenize_string,
    tokenize_string_v2,
    tokenize_texts,
    tokenize_texts_parallel,
    tokenize_texts_parallel_v2,
    tokenize_texts_v2,
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


# `tokenize_string` is pickled by reference into every bunch, so its output must
# never move. ADR 0002 item 2 inlined the three `gensim.parsing` regex helpers it
# used to import; this pins that the inlining was byte-for-byte -- v1 still
# lowercases, maps digits to "0", strips `<...>` tags, and shatters intra-word
# punctuation exactly as before. These expected outputs are the captured v1
# behaviour, independent of gensim now that the helpers live in-repo.
V1_BATTERY = [
    (
        "The English Wikipedia, founded in 2001.",
        ["the", "english", "wikipedia", "founded", "in", "0000"],
    ),
    (
        "<b>Simple</b> English uses basic vocabulary!",
        ["simple", "english", "uses", "basic", "vocabulary"],
    ),
    ("", []),
    ("ice-cream and 42 apples", ["ice", "cream", "and", "00", "apples"]),
    ("city's president", ["city", "s", "president"]),
    ("U.S.A. today", ["u", "s", "a", "today"]),
    ("3.50 dollars", ["0", "00", "dollars"]),
    ("café résumé", ["café", "résumé"]),
    (unicodedata.normalize("NFD", "café"), ["cafe"]),  # the NFD bug v2 fixes
    ("co_op under_score", ["co_op", "under_score"]),  # underscore is a word char
    ("foo#bar%baz@qux", ["foo", "bar", "baz", "qux"]),
]


@pytest.mark.parametrize("text,expected", V1_BATTERY)
def test_tokenize_string_v1_is_byte_unchanged(text, expected):
    assert tokenize_string(text) == expected


# ADR 0002 item 5: `tokenize_string_v2` -- the verified semantics table.
V2_TABLE = [
    ("City's", {}, ["city's"]),
    ("city\u2019s", {}, ["city's"]),  # U+2019 curly apostrophe -> ASCII '
    ("ice-cream", {}, ["ice-cream"]),
    ("ice\u2011cream", {}, ["ice-cream"]),  # U+2011 non-breaking hyphen -> ASCII -
    (unicodedata.normalize("NFD", "café"), {}, ["café"]),  # NFC fix
    ("2001", {}, ["2001"]),  # digits kept by default
    ("2001", {"normalize_digits": True}, ["0000"]),  # legacy hyperwords convention
    ("U.S.A.", {}, ["u", "s", "a"]),  # `.` deliberately not a joiner
]


@pytest.mark.parametrize("text,kwargs,expected", V2_TABLE)
def test_tokenize_string_v2_semantics(text, kwargs, expected):
    assert tokenize_string_v2(text, **kwargs) == expected


def test_tokenize_string_v2_lower_flag():
    assert tokenize_string_v2("Hello WORLD", lower=False) == ["Hello", "WORLD"]
    assert tokenize_string_v2("Hello WORLD") == ["hello", "world"]


def test_tokenize_string_v2_does_not_glue_across_dot():
    # `.` is not in the joiner set, so scraped `word.Next` stays two tokens
    assert tokenize_string_v2("word.Next sentence") == ["word", "next", "sentence"]


def test_tokenize_texts_v2_maps_over_the_list():
    assert tokenize_texts_v2(["ice-cream", "city's"]) == [["ice-cream"], ["city's"]]


def test_tokenize_texts_parallel_v2_matches_serial(pool_calls):
    texts = ["ice-cream 2001", "the city's café", "don't stop"]
    assert tokenize_texts_parallel_v2(texts) == tokenize_texts_v2(texts)


def test_tokenize_texts_parallel_v2_small_input_does_not_spawn_a_pool(pool_calls):
    tokenize_texts_parallel_v2(["ice-cream", "city's"])
    assert pool_calls == []


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
    The decision must not quietly delete the parallel path that large corpora
    rely on. Forcing the verdict is equivalent to handing in an input big enough
    for the pool to pay, without building one.
    """
    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 1)
    monkeypatch.setattr(preprocessing, "_pool_is_worth_starting", lambda *_: True)

    result = tokenize_texts_parallel(TEXTS)

    assert len(pool_calls) == 1
    # ... and the pooled path still produces exactly the same tokens
    assert result == tokenize_texts(TEXTS)


def test_character_threshold_short_circuits_before_measuring(monkeypatch, pool_calls):
    """
    Below `PARALLEL_MIN_CHARS` the decision is made without measuring at all --
    the probe would otherwise cost more than tokenizing the whole input, which
    is the small-input case this gate exists for. The threshold counts
    characters, not items.
    """
    measured = []
    monkeypatch.setattr(
        preprocessing,
        "_pool_is_worth_starting",
        lambda *_: measured.append(1) or True,
    )
    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 100)

    tokenize_texts_parallel(["a b c"] * 5)  # 25 chars, way under
    assert pool_calls == []
    assert measured == []  # not even probed

    tokenize_texts_parallel(["x" * 60] * 2)  # 120 chars, over -> now measured
    assert measured == [1]
    assert len(pool_calls) == 1


def test_pool_is_refused_when_it_would_not_pay(monkeypatch, pool_calls):
    """
    The bug this replaced: `PARALLEL_MIN_CHARS` alone let every corpus over a
    couple of megabytes into a pool that was measured to be a NET LOSS at every
    size up to 163M characters -- paying ~3s of spawn startup to make
    tokenization slower. Clearing the character gate must not be enough; the
    work has to actually be worth distributing.
    """
    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 1)

    result = tokenize_texts_parallel(TEXTS)

    assert pool_calls == []  # a handful of short strings: nowhere near worth it
    assert result == tokenize_texts(TEXTS)


def _verdict_for(monkeypatch, *, probe_seconds, workers=10, n_texts=5000):
    """
    Run the pool decision against a *fake clock* and a fixed worker count.

    Both have to be pinned or the test encodes the machine it was written on: an
    earlier version of this test slept for a millisecond per text and asserted
    the verdict, which passed on a 10-core laptop and failed on every 2-core CI
    runner. The arithmetic is what is under test, not the hardware.
    """
    monkeypatch.setattr(preprocessing, "_default_workers", lambda: workers)
    ticks = iter([0.0, probe_seconds])
    monkeypatch.setattr(preprocessing.time, "perf_counter", lambda: next(ticks))
    return preprocessing._pool_is_worth_starting(["a b c"] * n_texts, str.split)


def test_pool_is_started_when_the_work_dwarfs_the_startup(monkeypatch):
    # probe of 2000 of 5000 texts took 4s -> ~10s serial; 10 workers puts the
    # pool at 3 + 1 = 4s, comfortably past the 1.3x margin
    assert _verdict_for(monkeypatch, probe_seconds=4.0) is True


def test_pool_is_refused_when_the_startup_dominates(monkeypatch):
    # ~0.25s of work: the 3s startup can never be earned back
    assert _verdict_for(monkeypatch, probe_seconds=0.1) is False


def test_pool_is_refused_inside_the_margin(monkeypatch):
    # 4.25s serial against a 3.43s pool: faster on paper, but not by the 1.3x
    # the margin demands -- and the model ignores IPC, so the margin is what
    # stops a marginal win from being taken as a real one. The break-even sits
    # just above, at a 4.48s serial estimate.
    assert _verdict_for(monkeypatch, probe_seconds=1.7) is False
    assert _verdict_for(monkeypatch, probe_seconds=1.9) is True


def test_pool_is_refused_on_a_single_worker(monkeypatch):
    assert _verdict_for(monkeypatch, probe_seconds=4.0, workers=1) is False


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


def test_pool_probe_samples_across_the_input(monkeypatch):
    """
    A corpus is usually ordered -- by document, by date, by source -- so its head
    is not a fair sample of it. Probing only `texts[:N]` of a corpus that opens
    with short lines and ends with long ones would underestimate the work and
    refuse a pool worth starting.
    """
    seen = []

    def record(text):
        seen.append(text)
        return text.split()

    monkeypatch.setattr(preprocessing, "PROBE_TEXTS", 10)
    texts = [f"t{i}" for i in range(1000)]
    preprocessing._pool_is_worth_starting(texts, record)

    assert len(seen) == 10
    # spread across the input rather than taken from its head: a stride of
    # `len // PROBE_TEXTS` reaches the last decile, where `texts[:10]` would
    # never leave the first percent
    assert seen[0] == "t0"
    assert seen[-1] == "t900"
    assert seen == [f"t{i}" for i in range(0, 1000, 100)]
