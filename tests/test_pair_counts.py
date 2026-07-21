import math
import random

import numpy as np
import pytest

import hyperhyper
from hyperhyper import bunch, pair_counts
from hyperhyper.preprocessing import tokenize_texts


@pytest.fixture()
def uniform_corpus(tmp_path):
    """
    150 sentences of 5 distinct tokens each: 150 sentences, 750 tokens, and
    every word occurring exactly 150 times.

    The sentence count and the token count differ by a factor of 5 here, which
    is what makes the subsampling threshold below hand-checkable.
    """
    corpus = hyperhyper.Corpus.from_texts(
        ["a b c d e"] * 150, preproc_func=tokenize_texts
    )
    corpus.texts_to_file(tmp_path / "uniform", 25)
    return corpus


def test_uniform_corpus_size_is_sentences_not_tokens(uniform_corpus):
    """
    The premise of the two tests below: `corpus.size` counts sentences.
    """
    assert uniform_corpus.size == 150
    assert sum(uniform_corpus.counts.values()) == 750
    assert dict(uniform_corpus.counts) == dict.fromkeys(range(5), 150)


def test_subsample_threshold_is_scaled_by_token_count(uniform_corpus):
    """
    Regression test: the threshold is `subsample_factor * total tokens`, the
    `sample * train_words` of word2vec/hyperwords, not `* corpus.size` (which
    counts *sentences*).

    With `subsample_factor=0.5` the correct threshold is 0.5 * 750 = 375. Every
    word occurs 150 times, 150 < 375, so no word is frequent enough to be
    subsampled at all and the counts must come out completely untouched.

    The old sentence-based threshold was 0.5 * 150 = 75 < 150, so every word
    got scaled by sqrt(75/150) and the total came out at half. That is the
    core of the bug: the wrong scale does not just rescale the matrix, it
    changes *which* words cross the threshold.
    """
    untouched = hyperhyper.count_pairs(uniform_corpus, subsample=None)
    subsampled = hyperhyper.count_pairs(
        uniform_corpus, subsample="deter", subsample_factor=0.5
    )
    assert np.array_equal(subsampled.toarray(), untouched.toarray())


def test_subsample_scaling_factor_is_hand_computable(uniform_corpus):
    """
    The other side of the same threshold, where subsampling *does* apply.

    threshold = 0.02 * 750 tokens = 15, every word occurs 150 times, so the
    keep factor is sqrt(15/150) = sqrt(0.1). Pair counts are scaled by the
    factor of both of their words, so the total scales by exactly 0.1.

    Against the sentence count the threshold would have been 0.02 * 150 = 3 and
    the total would have shrunk to 3/150 = 0.02 of the original instead.
    """
    base = hyperhyper.count_pairs(uniform_corpus, subsample=None).sum()
    scaled = hyperhyper.count_pairs(
        uniform_corpus, subsample="deter", subsample_factor=0.02
    ).sum()
    assert scaled == pytest.approx(0.1 * base, rel=1e-6)


def test_count(corpus_on_disk):
    pair_c = hyperhyper.count_pairs(corpus_on_disk)
    assert pair_c.shape == (corpus_on_disk.vocab.size + 1,) * 2
    assert pair_c.nnz > 0
    assert pair_c.sum() > 0


@pytest.mark.parametrize("subsample", ["prob", "deter"])
def test_count_subsample(corpus_on_disk, subsample):
    pair_c = hyperhyper.count_pairs(corpus_on_disk, subsample=subsample)
    assert pair_c.shape == (corpus_on_disk.vocab.size + 1,) * 2
    assert pair_c.nnz > 0


def test_subsample_keep_probability_falls_with_frequency():
    """
    Regression test for an inverted subsampling probability.

    `iterate_tokens` treats this map as the probability of *keeping* a token,
    so it has to shrink as the word gets more frequent. The old code stored the
    word2vec *discard* probability (`1 - sqrt(...)`) instead, which kept the
    most frequent words 99% of the time and threw away the rare ones.
    """
    counts = {0: 2, 1: 10, 2: 100, 3: 10000}
    keep = pair_counts.subsample_keep_probabilities(counts, threshold=1.0)

    assert keep[0] > keep[1] > keep[2] > keep[3]
    assert keep[2] == pytest.approx(0.1)
    assert keep[3] == pytest.approx(0.01)

    # words at or below the threshold are never subsampled
    assert pair_counts.subsample_keep_probabilities({0: 1}, threshold=1.0) == {}


@pytest.mark.parametrize("dynamic_window", ["prob", "deter", "decay"])
def test_count_dynamic_window(corpus_on_disk, dynamic_window):
    pair_c = hyperhyper.count_pairs(corpus_on_disk, dynamic_window=dynamic_window)
    assert pair_c.shape == (corpus_on_disk.vocab.size + 1,) * 2
    assert pair_c.nnz > 0


def test_count_subsample_reduces_counts(corpus_on_disk):
    """
    Subsampling scales down the occurrences of frequent words, so it has to end
    up with fewer counted pairs than no subsampling at all.
    """
    with_sub = hyperhyper.count_pairs(corpus_on_disk, subsample="deter").sum()
    without_sub = hyperhyper.count_pairs(corpus_on_disk, subsample=None).sum()
    assert with_sub < without_sub


def test_count_invalid_mode(corpus_on_disk):
    with pytest.raises(ValueError):
        hyperhyper.count_pairs(corpus_on_disk, subsample="nope")
    with pytest.raises(ValueError):
        hyperhyper.count_pairs(corpus_on_disk, dynamic_window="nope")


# the randomized code paths run inside worker processes, which under `spawn` do
# not inherit the parent's RNG state, so the seed has to be threaded all the
# way down into the workers
SEED_ARGS = {"subsample": "prob", "subsample_factor": 0.1, "dynamic_window": "prob"}


def test_count_seed_is_reproducible(corpus_on_disk):
    first = hyperhyper.count_pairs(corpus_on_disk, seed=1312, **SEED_ARGS)
    second = hyperhyper.count_pairs(corpus_on_disk, seed=1312, **SEED_ARGS)
    assert np.array_equal(first.toarray(), second.toarray())


@pytest.fixture()
def varied_corpus(tmp_path):
    """
    A corpus whose chunks all differ from each other.

    Reproducibility of the parallel merge is only observable when the per-chunk
    matrices are actually different -- with identical chunks every summation
    order gives the same answer no matter what.
    """
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(60)]
    sents = [" ".join(rng.choices(vocab, k=12)) for _ in range(1000)]
    corpus = hyperhyper.Corpus.from_texts(sents, preproc_func=tokenize_texts)
    # many small chunks: the more partial matrices there are, the more room the
    # workers have to finish in a different order from one run to the next
    corpus.texts_to_file(tmp_path / "varied", 10)
    return corpus


def test_parallel_merge_is_bit_reproducible(varied_corpus, monkeypatch):
    """
    Regression test: the per-chunk matrices used to be added into the running
    total in `futures.as_completed` order. float32 addition is not associative,
    so the result -- and the `.npz` the bunch caches on disk -- differed between
    runs, which makes cache comparisons and hashes meaningless.

    `dynamic_window="decay"` gives the counts irrational values, so the
    rounding actually has something to bite on; with integer counts the sums
    are exact and the bug stays invisible.

    Note this is deliberately `array_equal`, not `allclose`: the whole point is
    that the bits match, and `allclose` passed even with the bug.

    The pool is forced on here. A corpus small enough to run as a test is now
    counted in-process, where the completion order this test is about does not
    exist -- so without the override the test would pass without ever reaching
    the code it guards.
    """
    monkeypatch.setattr(pair_counts, "POOL_STARTUP_SECONDS", 0.0)
    runs = [
        hyperhyper.count_pairs(
            varied_corpus, subsample=None, dynamic_window="decay"
        ).toarray()
        for _ in range(3)
    ]
    for i, run in enumerate(runs[1:], start=1):
        assert np.array_equal(runs[0], run), f"run {i} differs from run 0"


def test_count_seed_is_honoured(corpus_on_disk):
    """
    The converse of the test above. Without it, simply ignoring the seed would
    still pass.
    """
    first = hyperhyper.count_pairs(corpus_on_disk, seed=1312, **SEED_ARGS)
    other = hyperhyper.count_pairs(corpus_on_disk, seed=23, **SEED_ARGS)
    assert not np.allclose(first.toarray(), other.toarray())


# --------------------------------------------------------------------------
# how the work is distributed
# --------------------------------------------------------------------------
#
# `count_pairs` now decides at run time whether a process pool is worth its
# startup cost, so most of the tests above take the serial route. These pin the
# things that decision is not allowed to change.


def _force_pool(monkeypatch):
    """Make the pool look free, so any corpus at all goes through it."""
    monkeypatch.setattr(pair_counts, "POOL_STARTUP_SECONDS", 0.0)


def _forbid_pool(monkeypatch):
    """Turn starting a pool into a test failure rather than a slow test."""

    def boom(*args, **kwargs):
        raise AssertionError("a process pool was started")

    monkeypatch.setattr(pair_counts.futures, "ProcessPoolExecutor", boom)


def test_serial_and_pool_paths_are_bit_identical(varied_corpus, monkeypatch):
    """
    The load-bearing property of the whole scheduling change.

    Whether the chunks are counted in this process or fanned out to workers is
    now a performance decision taken from a run-time measurement, so it can flip
    between two runs of the same code on the same corpus. If the two routes did
    not agree bit for bit, that decision would silently change the cached
    matrix. `dynamic_window="decay"` makes the per-chunk counts irrational, so
    a difference in summation order has something to bite on.
    """
    kwargs = {"subsample": None, "dynamic_window": "decay", "window": 5}

    with monkeypatch.context() as m:
        _force_pool(m)
        pooled = hyperhyper.count_pairs(varied_corpus, **kwargs).toarray()

    with monkeypatch.context() as m:
        _forbid_pool(m)
        serial = hyperhyper.count_pairs(varied_corpus, **kwargs).toarray()

    assert serial.sum() > 0
    np.testing.assert_array_equal(serial, pooled)


def test_small_corpus_does_not_start_a_pool(uniform_corpus, monkeypatch):
    """
    150 sentences cannot repay a multi-second pool startup, and used to try
    anyway. Asserted by making the pool unavailable rather than by timing it.
    """
    _forbid_pool(monkeypatch)
    assert hyperhyper.count_pairs(uniform_corpus).nnz > 0


def test_a_costly_corpus_still_uses_the_pool(varied_corpus, monkeypatch):
    """
    The converse: the size check must not have turned the pool off altogether.
    """
    used = []
    real = pair_counts.futures.ProcessPoolExecutor
    monkeypatch.setattr(
        pair_counts.futures,
        "ProcessPoolExecutor",
        lambda *a, **kw: used.append(1) or real(*a, **kw),
    )
    _force_pool(monkeypatch)
    hyperhyper.count_pairs(varied_corpus)
    assert used, "the pool was never started"


def test_single_chunk_never_starts_a_pool(varied_corpus, monkeypatch, tmp_path):
    """
    One chunk means one task, so a pool can only ever add its own startup.
    """
    varied_corpus.texts_to_file(tmp_path / "one", 10_000)
    assert len(varied_corpus.texts) == 1
    _force_pool(monkeypatch)
    _forbid_pool(monkeypatch)
    assert hyperhyper.count_pairs(varied_corpus).nnz > 0


def test_merge_order_is_batched_and_sorted_within_a_batch():
    """
    The summation order is part of the answer (float32 addition is not
    associative), so it is pinned here as well as in the equivalence suite:
    consecutive groups of `2 * workers + 1` paths in corpus order, each group
    sorted lexicographically. Note `texts_10` sorts before `texts_2`.
    """
    paths = [f"texts_{i}.pkl" for i in range(12)]
    assert pair_counts.merge_order(paths, workers=2) == (
        sorted(paths[:5]) + sorted(paths[5:10]) + sorted(paths[10:])
    )
    # a single group, and lexicographic rather than numeric
    assert pair_counts.merge_order(paths, workers=10)[:3] == [
        "texts_0.pkl",
        "texts_1.pkl",
        "texts_10.pkl",
    ]


def test_merge_order_covers_every_chunk_exactly_once():
    paths = [f"texts_{i}.pkl" for i in range(37)]
    for workers in (1, 2, 3, 10, 64):
        assert sorted(pair_counts.merge_order(paths, workers)) == sorted(paths)


# --------------------------------------------------------------------------
# the chunking that feeds the parallelism
# --------------------------------------------------------------------------
#
# `text_chunk_size` lives on `Bunch`, but the only thing it actually governs is
# how many tasks the pair counting has to spread, so it is pinned next to the
# counting it exists to serve.


def test_auto_chunk_size_gives_the_pool_something_to_spread():
    """
    The regression the fixed 100000 default was: at 250k sentences it produced
    three chunks, so a ten-core machine could not use more than three of them.
    """
    n = 250_000
    size = bunch._auto_text_chunk_size(n, workers=10)
    n_chunks = math.ceil(n / size)
    assert n_chunks >= 10 * bunch.CHUNKS_PER_WORKER
    # and the old default really was the bug
    assert math.ceil(n / 100_000) < 10


@pytest.mark.parametrize("workers", [1, 2, 4, 10, 64])
def test_auto_chunk_size_tracks_the_worker_count(workers):
    n = 4_000_000
    size = bunch._auto_text_chunk_size(n, workers=workers)
    assert math.ceil(n / size) >= min(
        workers * bunch.CHUNKS_PER_WORKER, math.ceil(n / bunch.MIN_TEXT_CHUNK_SIZE)
    )


def test_auto_chunk_size_stays_within_its_bounds():
    """
    A tiny corpus must not be cut into per-sentence chunks (the round trip would
    cost more than the counting), and a huge one must not put an unbounded
    number of sentences into a single worker's memory.
    """
    assert bunch._auto_text_chunk_size(10, workers=10) == bunch.MIN_TEXT_CHUNK_SIZE
    assert bunch._auto_text_chunk_size(0, workers=10) == bunch.MIN_TEXT_CHUNK_SIZE
    assert bunch._auto_text_chunk_size(10**9, workers=10) == bunch.MAX_TEXT_CHUNK_SIZE


def test_explicit_chunk_size_is_still_honoured(tmp_path):
    """
    `text_chunk_size` is public and its meaning has not changed; only the
    default has moved from a fixed 100000 to "size it from the corpus".
    """
    corpus = hyperhyper.Corpus.from_texts(
        ["a b c d e"] * 100, preproc_func=tokenize_texts
    )
    b = hyperhyper.Bunch(tmp_path / "explicit", corpus, text_chunk_size=7)
    try:
        assert len(b.corpus.texts) == math.ceil(100 / 7)
    finally:
        b.close()


def test_invalid_chunk_size_is_rejected(tmp_path):
    corpus = hyperhyper.Corpus.from_texts(
        ["a b c d e"] * 10, preproc_func=tokenize_texts
    )
    with pytest.raises(ValueError, match="text_chunk_size"):
        hyperhyper.Bunch(tmp_path / "bad", corpus, text_chunk_size=0)
