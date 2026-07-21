import random

import numpy as np
import pytest

import hyperhyper
from hyperhyper import pair_counts
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


def test_parallel_merge_is_bit_reproducible(varied_corpus):
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
    """
    runs = [
        hyperhyper.count_pairs(
            varied_corpus, subsample=None, dynamic_window="decay"
        ).toarray()
        for _ in range(4)
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
