"""
Correctness gate for the upcoming vectorization of `hyperhyper/pair_counts.py`.

`bench/reference.py` holds a frozen snapshot of the counting code as of git SHA
f68cc74 plus the merge-order change that made the summation order independent of
the local core count (that file's header records why it was re-taken and how far
the results moved). Every test here runs the *live* `hyperhyper.count_pairs` and the
*frozen* `reference_count_pairs` over the same corpus and the same arguments and
compares the two matrices. Right now the live code and the snapshot are the same
code, so everything passes trivially -- that is the point. Once
`iterate_tokens` is replaced by a numpy implementation, these tests are the only
thing standing between "faster" and "faster but silently different".

THE DISTINCTION THAT MATTERS
============================

Not every configuration can be held to the same standard, and pretending
otherwise would produce either a test that fails spuriously or a test that
proves nothing.

Deterministic configurations -- BIT-IDENTICAL, no excuses
---------------------------------------------------------
    dynamic_window in (None, "deter", "decay")   and   subsample in (None, "deter")

No random number is drawn anywhere on these paths. The output is a pure
function of the corpus and the arguments, so a rewrite that changes even one
float32 low bit has changed the matrix, and there is no legitimate reason for
it to. These are asserted with `np.testing.assert_array_equal`, deliberately
NOT `assert_allclose`: `allclose` would wave through exactly the kind of
quiet numerical drift (a different summation order, float64 intermediates
rounding differently, a reordered accumulation) that this gate exists to catch.
If a vectorized rewrite cannot hit bit-identity here, that is a finding to
discuss, not a tolerance to widen.

Randomized configurations -- STATISTICAL EQUIVALENCE ONLY
----------------------------------------------------------
    dynamic_window == "prob"   or   subsample == "prob"

Bit-identity is *not achievable* here and demanding it would be a bug in the
test, not in the code. The current implementation draws one number per token
from `random.Random`, interleaved with the counting loop; any sensible
vectorization draws a whole array up front from a `numpy` generator. Different
RNG, different draw order, different numbers -- at an identical seed. The
matrices will differ per seed no matter how correct the rewrite is.

What must survive is the *distribution*. So these are checked by:

  1. running both implementations over many seeds,
  2. comparing the mean matrix and the mean total count,
  3. with a tolerance calibrated from the Monte-Carlo noise measured in the
     reference's own output, not from a magic constant (see
     `_assert_means_agree` and `_assert_totals_agree` for the arithmetic).

WHAT THE STATISTICAL TESTS DO NOT PROVE
---------------------------------------
Being honest about this is part of the deliverable:

  * They compare *first moments* (mean matrix, mean total) at a finite number
    of seeds. Two implementations with the same mean but different variance,
    or with the same marginals but a different correlation structure between
    entries, would pass.
  * They cannot detect a bias smaller than the Monte-Carlo resolution at the
    seed count used here. A rewrite whose keep-rate is off by, say, 0.1%
    relative would very likely slip through.
  * They say nothing about any individual seed.

COVERAGE
========
    window            1, 2, 5, 10
    dynamic_window    None, "deter", "prob", "decay"
    subsample         None, "deter", "prob"
    seed              1312, 23

The deterministic half of that product (4 x 3 x 2 x 2 = 48 cells) is swept
exhaustively -- each cell is one `count_pairs` call, so it is cheap. The
randomized half costs `N_STAT_SEEDS` calls per cell, so its window axis is
bracketed rather than swept; see `STAT_WINDOWS` for why that is defensible.
`delete_oov` is left at its default of True throughout: it is a plain filter
applied before any of the knobs here, and varying it would double the runtime
to re-test the same code paths on a slightly shorter token list.

They are backstopped by two things that ARE exact even on the random paths:
`test_random_config_is_reproducible_for_a_fixed_seed` (same seed twice must
give the identical matrix -- a property the rewrite must keep) and
`test_keep_probability_is_sqrt_threshold_over_count` plus
`test_prob_subsampling_matches_the_deter_scaling_in_expectation`, which pin the
keep rate to the closed form `sqrt(threshold / count)` rather than to whatever
the reference happens to do.
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

import hyperhyper
from hyperhyper import pair_counts
from hyperhyper.preprocessing import tokenize_texts

# `bench/` is not a package on the install path; it is a sibling of `tests/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.reference import reference_count_pairs

# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

WINDOWS = (1, 2, 5, 10)
DYNAMIC_WINDOWS = (None, "deter", "prob", "decay")
SUBSAMPLES = (None, "deter", "prob")
SEEDS = (1312, 23)

DETERMINISTIC_DYNAMIC = (None, "deter", "decay")
DETERMINISTIC_SUBSAMPLE = (None, "deter")

# Chosen so that subsampling actually bites without annihilating the corpus.
# The grid corpus below has ~10k tokens, so the threshold is ~60; against the
# Zipfian counts that leaves the tail untouched and pulls the head down to a
# keep probability of roughly 0.2-0.9. With the shipped default of 1e-5 the
# threshold would be 0.1, every single word would sit above it, and every keep
# probability would collapse to ~0.03 -- the matrix would be all but empty and
# the comparison would be vacuous.
SUBSAMPLE_FACTOR = 6e-3

# Seeds per statistical comparison. The tolerance is derived from the observed
# spread (below), so this trades runtime for resolution rather than for
# correctness: too few seeds makes the test weak, never flaky. Every
# `count_pairs` call pays a fixed ~3.5s process-pool startup, which is the only
# reason these numbers are not larger.
N_STAT_SEEDS = 10
N_STAT_SEEDS_FAST = 5

DETERMINISTIC_GRID = [
    (window, dynamic_window, subsample, seed)
    for window in WINDOWS
    for dynamic_window in DETERMINISTIC_DYNAMIC
    for subsample in DETERMINISTIC_SUBSAMPLE
    for seed in SEEDS
]

# The randomized cells cost N_STAT_SEEDS calls each instead of one, so the
# window axis is bracketed (smallest and largest) rather than swept. That is
# defensible because the window barely interacts with the randomness: for
# `subsample="prob"` the keep decision does not involve the window at all, and
# for `dynamic_window="prob"` the window only sets the upper bound of
# `randint(1, window)` -- 2 and 10 exercise a narrow and a wide draw range.
# The window axis is still swept exhaustively where it is cheap, in
# DETERMINISTIC_GRID above.
STAT_WINDOWS = (2, 10)

RANDOM_GRID = [
    (window, dynamic_window, subsample)
    for window in STAT_WINDOWS
    for dynamic_window in DYNAMIC_WINDOWS
    for subsample in SUBSAMPLES
    if dynamic_window == "prob" or subsample == "prob"
]

# A representative slice kept in the fast suite so that a broken harness is
# noticed without waiting for the full grid.
FAST_DETERMINISTIC = [
    (2, "deter", "deter", 1312),  # the shipped defaults
    (5, "decay", None, 1312),  # irrational counts, nothing to hide rounding
    (1, None, "deter", 23),  # degenerate window
]
FAST_RANDOM = [(5, "prob", "prob")]  # both knobs randomized at once


# --------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------


def _zipf_sentences(rng, n_sentences, vocab_size, sentence_length):
    """
    Sentences drawn from a Zipfian vocabulary.

    A uniform corpus would put every word on the same side of the subsampling
    threshold and hide any mistake in *which* words get subsampled; a skewed one
    keeps some words above it and some below.
    """
    # only letters survive gensim's tokenizer, so no digits in the words
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    words = [
        alphabet[i // len(alphabet)] + alphabet[i % len(alphabet)]
        for i in range(vocab_size)
    ]
    weights = [1.0 / (i + 1) for i in range(vocab_size)]
    return [
        " ".join(rng.choices(words, weights=weights, k=sentence_length))
        for _ in range(n_sentences)
    ]


@pytest.fixture(scope="module")
def grid_corpus(tmp_path_factory):
    """
    The corpus every equivalence test runs on.

    Deliberately chunked small (25 chunks): `count_pairs` merges one matrix per
    chunk, and with ten or more chunks the merge order stops being the obvious
    one (`texts_10.pkl` sorts before `texts_2.pkl`), which is exactly the kind
    of detail a rewrite can get wrong.
    """
    rng = random.Random(20260721)
    sents = _zipf_sentences(rng, n_sentences=1000, vocab_size=60, sentence_length=10)
    corpus = hyperhyper.Corpus.from_texts(sents, preproc_func=tokenize_texts)
    corpus.texts_to_file(tmp_path_factory.mktemp("grid") / "texts", 40)
    return corpus


@pytest.fixture(scope="module")
def uniform_corpus(tmp_path_factory):
    """
    Every word occurring equally often, so the keep probability
    `sqrt(threshold / count)` is a single hand-computable number.
    """
    corpus = hyperhyper.Corpus.from_texts(
        ["aa bb cc dd ee ff gg hh"] * 400, preproc_func=tokenize_texts
    )
    corpus.texts_to_file(tmp_path_factory.mktemp("uniform") / "texts", 50)
    return corpus


def _both(corpus, **kwargs):
    live = hyperhyper.count_pairs(corpus, **kwargs).toarray()
    ref = reference_count_pairs(corpus, **kwargs).toarray()
    return live, ref


# --------------------------------------------------------------------------
# deterministic configurations: bit-identical
# --------------------------------------------------------------------------


def _assert_bit_identical(corpus, window, dynamic_window, subsample, seed):
    live, ref = _both(
        corpus,
        window=window,
        dynamic_window=dynamic_window,
        subsample=subsample,
        subsample_factor=SUBSAMPLE_FACTOR,
        seed=seed,
    )
    # not assert_allclose: see the module docstring
    np.testing.assert_array_equal(
        live,
        ref,
        err_msg=(
            f"live count_pairs diverged from the frozen f68cc74 reference for "
            f"window={window} dynamic_window={dynamic_window!r} "
            f"subsample={subsample!r} seed={seed}. This configuration draws no "
            f"random numbers, so the matrices must match bit for bit."
        ),
    )
    # a rewrite that returns an all-zero matrix would pass the comparison above
    assert ref.sum() > 0, "the reference produced nothing -- the corpus is degenerate"


@pytest.mark.parametrize(
    ("window", "dynamic_window", "subsample", "seed"), FAST_DETERMINISTIC
)
def test_deterministic_config_is_bit_identical_fast(
    grid_corpus, window, dynamic_window, subsample, seed
):
    _assert_bit_identical(grid_corpus, window, dynamic_window, subsample, seed)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("window", "dynamic_window", "subsample", "seed"), DETERMINISTIC_GRID
)
def test_deterministic_config_is_bit_identical(
    grid_corpus, window, dynamic_window, subsample, seed
):
    """
    The full grid: 4 windows x 3 deterministic dynamic_window modes x 2
    deterministic subsample modes x 2 seeds.
    """
    _assert_bit_identical(grid_corpus, window, dynamic_window, subsample, seed)


@pytest.mark.slow
@pytest.mark.parametrize("dynamic_window", DETERMINISTIC_DYNAMIC)
@pytest.mark.parametrize("subsample", DETERMINISTIC_SUBSAMPLE)
def test_deterministic_config_ignores_the_seed(grid_corpus, dynamic_window, subsample):
    """
    Makes the two seeds in the grid above mean something.

    On these paths the seed is not consulted at all, so it must not be able to
    move the result. A rewrite that accidentally routes a deterministic mode
    through the RNG (e.g. by always drawing an offset array and only sometimes
    using it) would still be bit-identical to itself but would break here.
    """
    first, second = (
        hyperhyper.count_pairs(
            grid_corpus,
            window=5,
            dynamic_window=dynamic_window,
            subsample=subsample,
            subsample_factor=SUBSAMPLE_FACTOR,
            seed=seed,
        ).toarray()
        for seed in SEEDS
    )
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------
# randomized configurations: statistical equivalence
# --------------------------------------------------------------------------

# Number of standard errors allowed on the mean-total comparison. At 4 sigma a
# correct implementation flakes with probability ~6e-5 per assertion; across the
# handful of statistical assertions in this file that is a flake rate well under
# 1e-3, while still catching any relative bias larger than roughly 4/sqrt(N) of
# a per-seed standard deviation.
SIGMA_TOLERANCE = 4.0


def _sample(corpus, seeds, **kwargs):
    live, ref = [], []
    for seed in seeds:
        a, b = _both(corpus, seed=seed, **kwargs)
        live.append(a)
        ref.append(b)
    return np.array(live, dtype=np.float64), np.array(ref, dtype=np.float64)


def _assert_totals_agree(live, ref, label):
    """
    Compare E[total count] with a tolerance read off the observed spread.

    The difference of the two sample means has standard error
    `sqrt(var_live/N + var_ref/N)` under the null hypothesis that both
    implementations sample the same distribution. Requiring the observed
    difference to sit inside `SIGMA_TOLERANCE` of that is a real hypothesis
    test, and -- unlike a fixed `rtol` -- it automatically tightens as the
    randomness in a configuration gets weaker: if a knob barely randomizes
    anything, the standard error collapses and the test approaches exact
    equality on its own.
    """
    live_totals = live.sum(axis=(1, 2))
    ref_totals = ref.sum(axis=(1, 2))
    n = len(live_totals)
    se = math.sqrt(live_totals.var(ddof=1) / n + ref_totals.var(ddof=1) / n)
    # a floor so float32 accumulation noise cannot fail an otherwise exact match
    tol = max(SIGMA_TOLERANCE * se, 1e-6 * abs(ref_totals.mean()))
    diff = abs(live_totals.mean() - ref_totals.mean())
    assert diff <= tol, (
        f"{label}: mean total count {live_totals.mean():.6g} vs reference "
        f"{ref_totals.mean():.6g}, difference {diff:.6g} exceeds "
        f"{SIGMA_TOLERANCE} standard errors ({tol:.6g}). The two "
        f"implementations do not agree in expectation."
    )


def _assert_means_agree(live, ref, label):
    """
    Compare the mean pair-count *matrix*, calibrated by split-half noise.

    There is no defensible absolute tolerance for a whole matrix, so the null
    distribution is measured instead of guessed. Split the reference's N seeds
    into two halves and take the Frobenius distance between the two half-means:
    that is a direct sample of "how far apart do two independent estimates of
    this matrix land", at N/2 seeds each.

    Two independent N-seed means differ with variance 2*sigma^2/N; two
    independent N/2-seed means with variance 4*sigma^2/N. So under the null the
    live-vs-reference distance should come out about 1/sqrt(2) times the
    split-half distance. Asserting `<=` the split-half distance therefore leaves
    a ~1.41x margin over the expected null value -- loose enough not to flake,
    tight enough that a systematic bias comparable to the per-seed noise is
    caught, and entirely free of magic numbers tied to the corpus scale.
    """
    half = len(ref) // 2
    noise = np.linalg.norm(ref[:half].mean(axis=0) - ref[half:].mean(axis=0))
    signal = np.linalg.norm(live.mean(axis=0) - ref.mean(axis=0))

    assert noise > 0, (
        f"{label}: the reference gave the identical matrix for every seed, so "
        f"this configuration is not actually randomized and the statistical "
        f"comparison is vacuous -- it belongs in the bit-identical grid."
    )
    assert signal <= noise, (
        f"{label}: mean matrices differ by {signal:.6g} (Frobenius), which "
        f"exceeds the {noise:.6g} split-half Monte-Carlo noise of the "
        f"reference itself over {len(ref)} seeds. Under the null this ratio "
        f"should be around 0.71; got {signal / noise:.3f}."
    )


@pytest.mark.parametrize(("window", "dynamic_window", "subsample"), FAST_RANDOM)
def test_random_config_is_statistically_equivalent_fast(
    grid_corpus, window, dynamic_window, subsample
):
    """
    Same comparison as the full grid below, at a seed count the fast suite can
    afford. Fewer seeds only costs resolution -- the tolerance is derived from
    the observed spread, so a smaller sample widens it rather than making the
    test flaky.
    """
    _run_statistical_comparison(
        grid_corpus, window, dynamic_window, subsample, N_STAT_SEEDS_FAST
    )


@pytest.mark.slow
@pytest.mark.parametrize(("window", "dynamic_window", "subsample"), RANDOM_GRID)
def test_random_config_is_statistically_equivalent(
    grid_corpus, window, dynamic_window, subsample
):
    """
    Every grid cell where at least one knob is "prob".

    Bit-identity is impossible here (module docstring); equality of the first
    moment at the resolution of `N_STAT_SEEDS` seeds is what is demanded
    instead.
    """
    _run_statistical_comparison(
        grid_corpus, window, dynamic_window, subsample, N_STAT_SEEDS
    )


def _run_statistical_comparison(corpus, window, dynamic_window, subsample, n_seeds):
    label = f"window={window} dynamic_window={dynamic_window!r} subsample={subsample!r}"
    seeds = range(1000, 1000 + n_seeds)
    live, ref = _sample(
        corpus,
        seeds,
        window=window,
        dynamic_window=dynamic_window,
        subsample=subsample,
        subsample_factor=SUBSAMPLE_FACTOR,
    )
    _assert_totals_agree(live, ref, label)
    _assert_means_agree(live, ref, label)


@pytest.mark.parametrize(
    ("dynamic_window", "subsample"), [("prob", None), (None, "prob")]
)
def test_random_config_is_reproducible_for_a_fixed_seed(
    grid_corpus, dynamic_window, subsample
):
    """
    The exactness the random paths DO owe us.

    A rewrite is free to draw different numbers from the reference, but it is
    not free to draw different numbers from *itself*: two runs at the same seed
    must still give the identical matrix, or the cached `.npz` a `Bunch` writes
    stops being meaningful. This is the one bit-exact assertion that survives on
    the randomized paths.
    """
    kwargs = {
        "window": 5,
        "dynamic_window": dynamic_window,
        "subsample": subsample,
        "subsample_factor": SUBSAMPLE_FACTOR,
        "seed": 4711,
    }
    first = hyperhyper.count_pairs(grid_corpus, **kwargs).toarray()
    second = hyperhyper.count_pairs(grid_corpus, **kwargs).toarray()
    np.testing.assert_array_equal(first, second)
    assert first.sum() > 0


# --------------------------------------------------------------------------
# the keep rate itself, pinned to the closed form rather than to the reference
# --------------------------------------------------------------------------


def test_keep_probability_is_sqrt_threshold_over_count():
    """
    `subsample="prob"` must keep a token with probability
    `sqrt(threshold / count)`, and leave anything at or below the threshold
    alone. Checked against the formula directly, so it stays a real constraint
    even if the reference snapshot is ever regenerated from broken code.
    """
    counts = {0: 5, 1: 50, 2: 500, 3: 5000}
    threshold = 50.0
    keep = pair_counts.subsample_keep_probabilities(counts, threshold)

    assert set(keep) == {2, 3}, "words at or below the threshold must be untouched"
    for word in keep:
        assert keep[word] == pytest.approx(math.sqrt(threshold / counts[word]))


@pytest.mark.slow
def test_prob_subsampling_matches_the_deter_scaling_in_expectation(uniform_corpus):
    """
    The end-to-end version of the keep rate, and the reason "prob" and "deter"
    are two spellings of one idea.

    On a uniform corpus every word occurs `c` times, so every keep probability
    is the same `p = sqrt(threshold / c)`. `iterate_tokens` blanks a dropped
    token in place rather than removing it, so window spans do not shift and a
    pair survives exactly when both of its endpoints do -- probability `p^2`.
    `subsample="deter"` multiplies every pair by the factor of both its words,
    i.e. by exactly `p^2`, with no randomness at all.

    So `E[total | "prob"]` must equal `total | "deter"`. That equality is only
    true if the keep rate really is `sqrt(threshold / count)`; get the exponent
    or the direction wrong and the two totals part company immediately.
    """
    counts = list(uniform_corpus.counts.values())
    assert len(set(counts)) == 1, "fixture is meant to be uniform"
    total_tokens = sum(counts)
    threshold = SUBSAMPLE_FACTOR * total_tokens
    p = math.sqrt(threshold / counts[0])
    assert 0.05 < p < 0.95, (
        f"keep probability {p:.3f} is too close to a degenerate 0 or 1 for this "
        f"test to have any power"
    )

    kwargs = {"window": 5, "dynamic_window": None, "subsample_factor": SUBSAMPLE_FACTOR}
    deter_total = hyperhyper.count_pairs(
        uniform_corpus, subsample="deter", **kwargs
    ).sum()

    prob_totals = np.array(
        [
            hyperhyper.count_pairs(
                uniform_corpus, subsample="prob", seed=seed, **kwargs
            ).sum()
            for seed in range(2000, 2000 + N_STAT_SEEDS)
        ],
        dtype=np.float64,
    )

    se = math.sqrt(prob_totals.var(ddof=1) / len(prob_totals))
    diff = abs(prob_totals.mean() - deter_total)
    assert diff <= SIGMA_TOLERANCE * se, (
        f'mean total under subsample="prob" is {prob_totals.mean():.6g}, but the '
        f"deterministic sqrt(threshold/count) scaling gives {deter_total:.6g} "
        f"(p={p:.4f}). Difference {diff:.6g} exceeds {SIGMA_TOLERANCE} standard "
        f"errors ({SIGMA_TOLERANCE * se:.6g}): the per-token keep rate is not "
        f"sqrt(threshold/count)."
    )
