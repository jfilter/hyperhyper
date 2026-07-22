import math
import random

import numpy as np
import pytest

import hyperhyper
from hyperhyper import bunch, pair_counts, utils
from hyperhyper.preprocessing import tokenize_texts
from hyperhyper.utils import load_id_chunk


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


def test_empty_corpus_raises_a_clear_error(tmp_path):
    """
    Regression test for BUG 3: an empty corpus produces no text chunks, so
    `count_pairs_parallel` returns None and `count_matrix.nnz` used to raise an
    opaque `AttributeError: 'NoneType' object has no attribute 'nnz'`. It must
    raise a clear, actionable error instead.
    """
    corpus = hyperhyper.Corpus.from_texts([], preproc_func=tokenize_texts)
    with pytest.raises(ValueError, match="empty corpus"):
        hyperhyper.count_pairs(corpus)


def test_count(corpus_on_disk):
    pair_c = hyperhyper.count_pairs(corpus_on_disk)
    assert pair_c.shape == (corpus_on_disk.vocab.size + 1,) * 2
    assert pair_c.nnz > 0
    assert pair_c.sum() > 0


@pytest.mark.parametrize("subsample", ["prob", "dirty", "deter"])
def test_count_subsample(corpus_on_disk, subsample):
    pair_c = hyperhyper.count_pairs(corpus_on_disk, subsample=subsample)
    assert pair_c.shape == (corpus_on_disk.vocab.size + 1,) * 2
    assert pair_c.nnz > 0


@pytest.fixture()
def frequent_word_corpus(tmp_path):
    """
    A corpus built to separate the clean and dirty subsampling variants.

    Every "aa xx bb" sentence puts the frequent word ``xx`` between two rare
    words that are otherwise never adjacent; the "xx xx ..." sentences inflate
    ``xx``'s frequency so it -- and only it -- crosses the subsampling
    threshold. With ``window=1`` the only way ``aa`` and ``bb`` can ever
    co-occur is if the ``xx`` between them is *removed* from the sequence so the
    window closes up, which is exactly what dirty subsampling does and clean
    subsampling does not.
    """
    sents = ["aa xx bb"] * 200 + ["xx " * 10] * 200
    corpus = hyperhyper.Corpus.from_texts(sents, preproc_func=tokenize_texts)
    corpus.texts_to_file(tmp_path / "frequent", 50)
    return corpus


# `aa` and `bb` occur 200 times, `xx` 2200 times, 2600 tokens in total. With
# this factor the threshold is 0.15 * 2600 = 390, so only `xx` (2200 > 390) is
# subsampled while `aa`/`bb` (200 < 390) are left completely alone.
FREQUENT_WORD_FACTOR = 0.15


def test_only_the_frequent_word_is_subsampled(frequent_word_corpus):
    """
    The premise of the clean-vs-dirty test below: the fixture really does put
    exactly one word -- the frequent ``xx`` -- above the subsampling threshold.
    """
    counts = frequent_word_corpus.counts
    token2id = frequent_word_corpus.vocab.token2id
    threshold = FREQUENT_WORD_FACTOR * sum(counts.values())
    keep = pair_counts.subsample_keep_probabilities(counts, threshold)
    assert set(keep) == {token2id["xx"]}
    assert token2id["aa"] not in keep
    assert token2id["bb"] not in keep


def _co_occurrence(matrix, corpus, word_a, word_b):
    """The total counted co-occurrence between two words, both directions."""
    i = corpus.vocab.token2id[word_a]
    j = corpus.vocab.token2id[word_b]
    return matrix[i, j] + matrix[j, i]


def test_dirty_subsampling_reaches_past_the_dropped_token(frequent_word_corpus):
    """
    The concrete difference between the clean and dirty subsampling variants
    (Levy, Goldberg & Dagan 2015, section 3.1).

    With ``window=1``, ``aa`` and ``bb`` are two positions apart in every
    "aa xx bb" sentence, so they can only ever be counted as co-occurring if the
    ``xx`` between them is removed and the window closes up:

      * CLEAN (``subsample="prob"``) blanks ``xx`` but keeps its slot, so the
        window still spans that slot and never reaches from ``aa`` to ``bb``.
        Their co-occurrence is therefore exactly zero -- and it is zero
        regardless of the seed, because even a *kept* ``xx`` sits between them.
      * DIRTY (``subsample="dirty"``) deletes ``xx`` before the window is built,
        so whenever ``xx`` is dropped the sentence becomes "aa bb" and the two
        words co-occur. Their co-occurrence is therefore strictly positive.
    """
    common = {
        "window": 1,
        "dynamic_window": None,
        "subsample_factor": FREQUENT_WORD_FACTOR,
        "seed": 1312,
    }
    clean = hyperhyper.count_pairs(
        frequent_word_corpus, subsample="prob", **common
    ).toarray()
    dirty = hyperhyper.count_pairs(
        frequent_word_corpus, subsample="dirty", **common
    ).toarray()

    clean_ab = _co_occurrence(clean, frequent_word_corpus, "aa", "bb")
    dirty_ab = _co_occurrence(dirty, frequent_word_corpus, "aa", "bb")

    assert clean_ab == 0, (
        "clean subsampling keeps the dropped word's slot, so the window must "
        f"never reach from aa to bb -- got {clean_ab} co-occurrences"
    )
    assert dirty_ab > 0, (
        "dirty subsampling removes the dropped word, so aa and bb must end up "
        "adjacent whenever the xx between them is dropped"
    )


def test_dirty_subsampling_is_reproducible_for_a_fixed_seed(frequent_word_corpus):
    """
    Dirty subsampling draws random numbers, so -- like ``"prob"`` -- two runs at
    the same seed must still give the bit-identical matrix, or the ``.npz`` a
    ``Bunch`` caches stops being meaningful.
    """
    kwargs = {
        "window": 2,
        "subsample": "dirty",
        "subsample_factor": FREQUENT_WORD_FACTOR,
        "seed": 99,
    }
    first = hyperhyper.count_pairs(frequent_word_corpus, **kwargs).toarray()
    second = hyperhyper.count_pairs(frequent_word_corpus, **kwargs).toarray()
    assert np.array_equal(first, second)
    assert first.sum() > 0


def test_dirty_and_clean_subsampling_differ(frequent_word_corpus):
    """
    Beyond the single hand-checked cell above: over the whole matrix, dirty and
    clean subsampling are genuinely different reductions, not two spellings of
    one. Dirty closes windows up and so counts strictly more pairs in total.
    """
    common = {
        "window": 2,
        "subsample_factor": FREQUENT_WORD_FACTOR,
        "seed": 7,
    }
    clean = hyperhyper.count_pairs(frequent_word_corpus, subsample="prob", **common)
    dirty = hyperhyper.count_pairs(frequent_word_corpus, subsample="dirty", **common)
    assert not np.array_equal(clean.toarray(), dirty.toarray())
    # removing a token can only pull further words into a window, never push
    # them out, so dirty counts at least as many pairs as clean
    assert dirty.sum() > clean.sum()


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


def test_merge_order_is_sorted_by_path():
    """
    The summation order is part of the answer (float32 addition is not
    associative), so it is pinned here as well as in the equivalence suite: one
    canonical order, sorted lexicographically, taking no argument that could
    make it differ between machines. Note `texts_10` sorts before `texts_2`.
    """
    paths = [f"texts_{i}.pkl" for i in range(12)]
    assert pair_counts.merge_order(paths) == sorted(paths)
    assert pair_counts.merge_order(paths)[:3] == [
        "texts_0.pkl",
        "texts_1.pkl",
        "texts_10.pkl",
    ]
    # the input order must not survive into the output
    assert pair_counts.merge_order(list(reversed(paths))) == sorted(paths)


def test_merge_order_covers_every_chunk_exactly_once():
    paths = [f"texts_{i}.pkl" for i in range(37)]
    assert sorted(pair_counts.merge_order(paths)) == sorted(paths)


def test_merge_order_takes_no_worker_count():
    """
    The old signature was `merge_order(paths, workers)` and the core count was
    what made the result machine-dependent. Passing one now has to be an error
    rather than being quietly accepted and ignored.
    """
    with pytest.raises(TypeError):
        pair_counts.merge_order(["texts_0.pkl"], 10)


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
    size = bunch._auto_text_chunk_size(n)
    n_chunks = math.ceil(n / size)
    assert n_chunks == bunch.TARGET_TEXT_CHUNKS
    # and the old default really was the bug
    assert math.ceil(n / 100_000) < 10


def test_auto_chunk_size_hits_the_target_chunk_count():
    """
    Between the two clamps, the corpus is cut into exactly
    `TARGET_TEXT_CHUNKS` pieces -- that number, not the core count, is what the
    parallelism is bought with.
    """
    for n in (100_000, 250_000, 500_000, 1_000_000):
        size = bunch._auto_text_chunk_size(n)
        assert math.ceil(n / size) == bunch.TARGET_TEXT_CHUNKS


def test_auto_chunk_size_stays_within_its_bounds():
    """
    A tiny corpus must not be cut into per-sentence chunks (the round trip would
    cost more than the counting), and a huge one must not put an unbounded
    number of sentences into a single worker's memory.
    """
    assert bunch._auto_text_chunk_size(10) == bunch.MIN_TEXT_CHUNK_SIZE
    assert bunch._auto_text_chunk_size(0) == bunch.MIN_TEXT_CHUNK_SIZE
    assert bunch._auto_text_chunk_size(10**9) == bunch.MAX_TEXT_CHUNK_SIZE


# --------------------------------------------------------------------------
# machine independence: the same corpus must give the same matrix everywhere
# --------------------------------------------------------------------------
#
# The tests above pin *what* the order and the chunking are. These pin the
# property those choices exist for, and they are the ones that would have caught
# the bug: up to f68cc74 both the summation order (`workers * 2 + 1` groups) and
# the chunk count (`workers * 4`) were functions of the local core count, so two
# machines with different core counts produced different matrices for the same
# corpus and the same arguments.
#
# The configurations are chosen so the test can actually fail. `window=2
# dynamic_window="deter"` (the shipped default) and `dynamic_window=None` are
# immune to ANY reordering -- their counts are 1.0 and multiples of 0.5, exact
# in float32 -- so they would pass against a still-broken implementation and
# prove nothing. window=5 "deter" and "decay" were measured to move 760-900
# cells of 3721 by up to 4.883e-04 on the fixture below when the order changed,
# which is the sensitivity this test needs.
ORDER_SENSITIVE_CONFIGS = [
    {"window": 5, "dynamic_window": "deter", "subsample": None},
    {"window": 5, "dynamic_window": "decay", "subsample": None},
    {"window": 10, "dynamic_window": "decay", "subsample": "deter"},
]

# 1 takes the serial route (a one-worker pool is never worth starting), the rest
# fan out. Under the old rule these give merge groups of 3, 7, 15 and 33 over the
# 25 chunks of `order_sensitive_corpus`, i.e. four genuinely different orders --
# `test_the_old_merge_order_really_did_depend_on_the_worker_count` below checks
# that they really do disagree, so these tests cannot quietly go vacuous.
SIMULATED_WORKER_COUNTS = (1, 3, 7, 16)


@pytest.fixture(scope="module")
def order_sensitive_corpus(tmp_path_factory):
    """
    A corpus on which the merge order demonstrably matters.

    Two properties are load-bearing and neither is automatic:

      * a Zipfian vocabulary, so the per-chunk matrices have wildly different
        magnitudes and adding them in a different order actually rounds
        differently. A uniform corpus cancels almost all of it -- measured: the
        same sweep on a flat 60-word vocabulary moved a single cell, and would
        have let a still-broken implementation pass.
      * 25 chunks, enough that the old `workers * 2 + 1` grouping produces
        several groups. At 10 chunks every worker count in the sweep gives one
        or two groups and the old bug is invisible (measured: 0 cells).

    Under the old implementation this fixture moves up to 900 of its 3721 cells
    across `SIMULATED_WORKER_COUNTS`, by up to 4.883e-04.
    """
    rng = random.Random(20260721)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = [alphabet[i // 26] + alphabet[i % 26] for i in range(60)]
    weights = [1.0 / (i + 1) for i in range(60)]
    sents = [" ".join(rng.choices(words, weights=weights, k=10)) for _ in range(1000)]
    corpus = hyperhyper.Corpus.from_texts(sents, preproc_func=tokenize_texts)
    corpus.texts_to_file(tmp_path_factory.mktemp("order") / "texts", 40)
    assert len(corpus.texts) == 25
    return corpus


def _old_merge_order(texts_paths, workers):
    """
    The pre-f68cc74 order: consecutive groups of `2 * workers + 1` paths in
    corpus order, each group sorted. Kept here purely so the test below can show
    that the corpus and configuration it uses are ones the old rule got wrong.
    """
    group = workers * 2 + 1
    return [
        p
        for i in range(0, len(texts_paths), group)
        for p in sorted(texts_paths[i : i + group])
    ]


def _count_with_workers(corpus, workers, monkeypatch, merge=None, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(pair_counts, "_default_workers", lambda: workers)
        if merge is not None:
            m.setattr(pair_counts, "merge_order", merge)
        _force_pool(m)
        return hyperhyper.count_pairs(corpus, **kwargs).toarray()


@pytest.mark.parametrize("config", ORDER_SENSITIVE_CONFIGS[:1])
def test_result_is_independent_of_the_worker_count_fast(
    order_sensitive_corpus, monkeypatch, config
):
    """
    One order-sensitive configuration at two worker counts, kept in the fast
    suite so a regression is noticed without waiting for the full sweep.
    """
    _assert_same_across_worker_counts(
        order_sensitive_corpus, monkeypatch, config, (3, 16)
    )


@pytest.mark.slow
@pytest.mark.parametrize("config", ORDER_SENSITIVE_CONFIGS)
def test_result_is_independent_of_the_worker_count(
    order_sensitive_corpus, monkeypatch, config
):
    """
    The guarantee `merge_order` now makes, stated positively.

    Each worker count changes the pool size, the in-flight window and the merge
    grouping -- everything the old implementation let leak into the sum. The
    matrices must come out `array_equal`, not `allclose`: a reordered float32
    sum is exactly the kind of last-bit difference `allclose` waves through, and
    it is the difference this test exists to forbid.
    """
    _assert_same_across_worker_counts(
        order_sensitive_corpus, monkeypatch, config, SIMULATED_WORKER_COUNTS
    )


def test_the_old_merge_order_really_did_depend_on_the_worker_count(
    order_sensitive_corpus, monkeypatch
):
    """
    Keeps the two tests above honest.

    They assert that varying the worker count changes nothing -- an assertion
    that also passes on a corpus where the summation order never mattered in the
    first place, which is how a machine-independence test rots into a no-op. So
    this one runs the SAME corpus and configuration through the old
    core-count-dependent order and demands that it *does* disagree with itself.
    If this test ever starts failing, the fixture has gone insensitive and the
    tests above have stopped proving anything.
    """
    config = ORDER_SENSITIVE_CONFIGS[0]
    matrices = [
        _count_with_workers(
            order_sensitive_corpus,
            workers,
            monkeypatch,
            merge=lambda paths, w=workers: _old_merge_order(paths, w),
            **config,
        )
        for workers in SIMULATED_WORKER_COUNTS
    ]
    differing = max((m != matrices[0]).sum() for m in matrices[1:])
    assert differing > 100, (
        f"the old core-count-dependent merge order moved only {differing} cells "
        f"on this fixture, so it is no longer a demonstration of the bug and the "
        f"machine-independence tests above are not proving anything"
    )


def _assert_same_across_worker_counts(corpus, monkeypatch, config, worker_counts):
    reference = None
    for workers in worker_counts:
        matrix = _count_with_workers(corpus, workers, monkeypatch, **config)
        if reference is None:
            reference = matrix
            assert reference.sum() > 0, "degenerate corpus -- nothing was counted"
            continue
        np.testing.assert_array_equal(
            matrix,
            reference,
            err_msg=(
                f"{config} gave a different matrix at {workers} workers than at "
                f"{worker_counts[0]}. The result must not depend on the core "
                f"count of the machine it was computed on."
            ),
        )


def test_chunking_is_independent_of_the_worker_count(monkeypatch):
    """
    The same guarantee one layer down, and the more important half of it: a
    different chunk *count* means different partial matrices, not merely the
    same ones summed in a different order. `_auto_text_chunk_size` used to call
    `_default_workers()`; it must now ignore it entirely.
    """
    for n_texts in (1_000, 250_000, 4_000_000):
        sizes = set()
        for workers in (1, 2, 4, 10, 64, 256):
            with monkeypatch.context() as m:
                # both the source and the name `bunch` would import it under,
                # so the patch bites whichever way the dependency comes back
                m.setattr(utils, "_default_workers", lambda w=workers: w)
                m.setattr(bunch, "_default_workers", lambda w=workers: w, raising=False)
                sizes.add(bunch._auto_text_chunk_size(n_texts))
        assert len(sizes) == 1, (
            f"{n_texts} sentences were chunked {len(sizes)} different ways "
            f"depending on the core count: {sorted(sizes)}"
        )


def test_bunch_writes_the_same_chunks_whatever_the_core_count(tmp_path, monkeypatch):
    """
    End to end through `Bunch`, which is where the chunk layout is actually
    committed to disk. Two machines building the same bunch from scratch -- the
    "reproduce your result" case -- must lay it out identically.
    """
    corpus = hyperhyper.Corpus.from_texts(
        ["a b c d e f"] * 20_000, preproc_func=tokenize_texts
    )
    layouts = []
    for workers in (2, 10, 64):
        with monkeypatch.context() as m:
            m.setattr(bunch, "_default_workers", lambda w=workers: w, raising=False)
            b = hyperhyper.Bunch(tmp_path / f"w{workers}", corpus)
            try:
                layouts.append([len(t) for t in map(load_id_chunk, b.corpus.texts)])
            finally:
                b.close()
    assert layouts[0] == layouts[1] == layouts[2], (
        f"the corpus was cut differently per core count: "
        f"{[len(x) for x in layouts]} chunks"
    )
    assert len(layouts[0]) == bunch.TARGET_TEXT_CHUNKS


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
