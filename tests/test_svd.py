from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hyperhyper.svd import SVDEmbedding, calc_svd

# A diagonal matrix is the easiest SVD to do by hand: its singular values are
# the absolute diagonal entries, and its left singular vectors are the unit
# basis vectors.
DIAG = np.diag([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).astype(np.float32)


def wrap(dense):
    """`calc_svd` takes anything with a sparse `.m`."""
    return SimpleNamespace(m=csr_matrix(dense))


IMPLS = ["scipy", "gensim", "scikit"]


@pytest.mark.parametrize("impl", IMPLS)
def test_singular_values_are_descending(impl):
    """
    Regression test: `scipy.sparse.linalg.svds` returns the singular values
    ascending, while `gensim` and `scikit` return them descending. All three
    recover the same subspace, so similarity and analogy scores were unaffected
    -- but anything slicing `ut[:, :k]`, reading `s[0]` as the leading
    component or plotting the spectrum got the spectrum backwards, and
    `bunch.svd_matrix` persists these arrays to disk.
    """
    _, s, _ = calc_svd(wrap(DIAG), 3, impl, {})
    assert np.allclose(s, [6.0, 5.0, 4.0])


@pytest.mark.parametrize("impl", IMPLS)
def test_singular_vectors_follow_the_values(impl):
    """
    Reordering the values without reordering the columns of `ut` alongside them
    would be worse than not sorting at all.

    For a diagonal matrix the left singular vector belonging to the entry `d_i`
    is the unit vector `e_i` (up to sign), so the leading column has to be
    +/- e_0, the next +/- e_1, and so on.
    """
    ut, _, _ = calc_svd(wrap(DIAG), 3, impl, {})
    for i in range(3):
        expected = np.zeros(6)
        expected[i] = 1.0
        assert np.allclose(np.abs(ut[:, i]), expected, atol=1e-5)


def test_impls_agree_on_the_spectrum():
    spectra = {impl: calc_svd(wrap(DIAG), 3, impl, {})[1] for impl in IMPLS}
    for impl, s in spectra.items():
        assert np.allclose(s, spectra["scipy"], atol=1e-5), impl


def test_calc_svd_unknown_impl():
    with pytest.raises(ValueError):
        calc_svd(wrap(DIAG), 3, "nope", {})


def _rank_deficient(n=40, rank=5, seed=0):
    """An ``n``-by-``n`` matrix of exact rank ``rank`` (``rank`` << ``n``)."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, rank)) @ rng.random((rank, n))).astype(np.float32)


def _neighbours(impl, matrix, dim, query=0, n=6):
    ut, s, _ = calc_svd(wrap(matrix), dim, impl, {})
    return [idx for idx, _ in SVDEmbedding(ut, s, eig=0.0).most_similar(query, n=n)]


def test_rank_deficient_eig0_is_deterministic_across_runs():
    """
    Regression test for BUG 1: with ``dim`` above the matrix's numerical rank the
    surplus singular vectors are arbitrary null-space directions, and under the
    default ``eig=0`` they entered the embedding with full weight. `svds`
    (ARPACK) starts from an unseeded random vector, so the *same* matrix and
    parameters gave *different* nearest neighbours run to run. Dropping the
    negligible components makes it deterministic.
    """
    matrix = _rank_deficient()  # rank 5, dim 15 -> 10 null-space directions
    runs = [_neighbours("scipy", matrix, dim=15) for _ in range(5)]
    for i, run in enumerate(runs[1:], start=1):
        np.testing.assert_array_equal(run, runs[0], err_msg=f"run {i} differs")


def test_impls_agree_on_a_rank_deficient_matrix():
    """
    The other half of BUG 1: once the noise columns are dropped, all three
    backends keep the same significant subspace, so `eig=0` similarities -- which
    are invariant to the basis and sign of that subspace -- must agree.
    """
    matrix = _rank_deficient()
    ref = _neighbours("scipy", matrix, dim=15)
    for impl in IMPLS:
        np.testing.assert_array_equal(_neighbours(impl, matrix, dim=15), ref)


@pytest.mark.parametrize("impl", IMPLS)
def test_dim_over_matrix_rank_is_truncated_consistently(impl):
    """
    BUG 1/2: every backend must return exactly the number of *significant*
    components (5), not `dim` columns padded with null-space noise.
    """
    ut, s, _ = calc_svd(wrap(_rank_deficient()), 15, impl, {})
    assert ut.shape[1] == 5
    assert s.shape == (5,)


@pytest.mark.parametrize("impl", IMPLS)
def test_dim_at_or_above_min_shape_is_clamped(impl):
    """
    Regression test for BUG 2: `dim` was never clamped, so `dim >= min(shape)`
    crashed `svds` with an opaque `ValueError` while gensim and scikit silently
    returned different column counts. All three now clamp to `min(shape) - 1`
    (and then drop negligibles), so all three agree.
    """
    full_rank = np.diag(np.arange(1, 11)).astype(np.float32)  # 10x10, rank 10
    ut, s, _ = calc_svd(wrap(full_rank), 10, impl, {})
    assert ut.shape[1] == 9  # clamped from 10 to min(shape) - 1
    assert s.shape == (9,)


def test_all_zero_matrix_raises_a_clear_error():
    """
    BUG 2: an all-zero (S)PPMI matrix (e.g. from a huge `neg`) made `svds` raise
    the opaque `ArpackError: Starting vector is zero`. It must raise a clear,
    actionable error instead.
    """
    zero = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="all-zero"):
        calc_svd(wrap(zero), 5, "scipy", {})


@pytest.mark.parametrize("impl", IMPLS)
def test_full_rank_result_is_unchanged(impl):
    """
    The guard on BUG 1's fix: for a full-rank `dim < rank` request no column is
    negligible, so nothing is dropped and the recovered spectrum is exactly the
    top-`dim` singular values -- the existing, recorded behaviour.
    """
    rng = np.random.default_rng(1)
    matrix = rng.random((30, 30)).astype(np.float32)
    reference = np.linalg.svd(matrix, compute_uv=False)[:5]
    ut, s, _ = calc_svd(wrap(matrix), 5, impl, {})

    # The invariant the BUG-1 fix must hold is that nothing is dropped for a
    # full-rank request -- deterministic and identical across all backends.
    assert ut.shape[1] == 5
    assert len(s) == 5

    # Only `scipy` is an exact truncated SVD; `gensim` and `scikit` are
    # randomized and recover the trailing singular values only approximately
    # (measured worst case ~1.3e-4, and it varies with the BLAS build, which is
    # why a 1e-4 bound was flaky on 3.11 alone). Hold the exact backend tightly
    # and the randomized ones to a tolerance that covers their real spread.
    tol = 1e-4 if impl == "scipy" else 5e-3
    assert np.allclose(np.sort(s)[::-1], reference, atol=tol)


def _embedding():
    rng = np.random.default_rng(0)
    dense = rng.random((20, 20)).astype(np.float32)
    ut, s, _ = calc_svd(wrap(dense), 5, "scikit", {})
    return SVDEmbedding(ut, s)


def test_svd_most_similar_returns_index_then_score():
    """
    Regression test for the tuple order: `most_similar` used to return
    `(score, index)` while `most_similar_vectors` returned `(index, score)`.
    Both now return `(index, score)`.
    """
    embd = _embedding()
    for idx, score in embd.most_similar(0, n=5):
        assert isinstance(idx, int)
        assert embd.similarity(0, idx) == pytest.approx(score, rel=1e-5)

    # the word itself is its own nearest neighbour, at a similarity of 1
    first_idx, first_score = embd.most_similar(0, n=5)[0]
    assert first_idx == 0
    assert first_score == pytest.approx(1.0, rel=1e-5)


def test_svd_most_similar_agrees_with_most_similar_vectors():
    """
    `most_similar(w)` and `most_similar_vectors([w], [])` ask the same question,
    so with a consistent tuple order they must give the same answer.
    """
    embd = _embedding()
    by_index = embd.most_similar(3, n=5)
    by_vector = embd.most_similar_vectors([3], [], topn=5)
    assert [i for i, _ in by_index] == [i for i, _ in by_vector]
    for (_, a), (_, b) in zip(by_index, by_vector, strict=True):
        assert a == pytest.approx(b, rel=1e-5)


def test_svd_most_similar_is_sorted_descending():
    embd = _embedding()
    scores = [score for _, score in embd.most_similar(7, n=10)]
    assert scores == sorted(scores, reverse=True)


# right singular vectors + w+c (FEATURE 2)


@pytest.mark.parametrize("impl", IMPLS)
def test_calc_svd_returns_right_vectors(impl):
    """
    `calc_svd` now hands back `(ut, s, vt)`. For every backend -- including
    gensim, whose `stochastic_svd` returns only `(U, s)` so `vt` is reconstructed
    from `Vᵀ = diag(s)⁻¹·Uᵀ·M` -- the triplet must reconstruct the rank-`dim`
    truncation of the input.
    """
    ut, s, vt = calc_svd(wrap(DIAG), 3, impl, {})
    assert vt.shape == (3, 6)
    approx = ut @ np.diag(s) @ vt
    expected = np.zeros((6, 6))
    expected[0, 0], expected[1, 1], expected[2, 2] = 6.0, 5.0, 4.0
    assert np.allclose(approx, expected, atol=1e-3)


@pytest.mark.parametrize("impl", IMPLS)
def test_drop_negligible_truncates_vt_consistently(impl):
    """
    The negligible-component drop has to trim `vt`'s rows to the same kept set as
    `ut`'s columns and `s`, or `w+c` would sum mismatched subspaces.
    """
    ut, s, vt = calc_svd(wrap(_rank_deficient()), 15, impl, {})
    assert ut.shape[1] == s.shape[0] == vt.shape[0] == 5


def test_wplusc_adds_context_vectors():
    """
    `add_context=True` builds `w+c = U·S^eig + V·S^eig` (then normalizes).

    Uses an *asymmetric* square matrix so `U != V` and the sum genuinely differs
    from the word-only vectors, and checks the result against the by-hand
    construction.
    """
    rng = np.random.default_rng(3)
    dense = rng.random((12, 12)).astype(np.float32)  # asymmetric -> U != V
    ut, s, vt = calc_svd(wrap(dense), 5, "scipy", {})

    word_only = SVDEmbedding(ut, s, eig=0.5)
    wplusc = SVDEmbedding(ut, s, eig=0.5, vt=vt, add_context=True)

    assert wplusc.m.shape == word_only.m.shape == (12, 5)
    # the context vectors really were added: w+c differs from word-only
    assert not np.allclose(wplusc.m, word_only.m)

    # ... and equals U·S^0.5 + V·S^0.5, normalized, built by hand
    manual = np.power(s, 0.5) * ut + np.power(s, 0.5) * vt.T
    manual = manual / np.linalg.norm(manual, axis=1, keepdims=True)
    assert np.allclose(wplusc.m, manual, atol=1e-5)


def test_wplusc_requires_context_vectors():
    ut, s, _ = calc_svd(wrap(DIAG), 3, "scipy", {})
    with pytest.raises(ValueError, match="vt"):
        SVDEmbedding(ut, s, add_context=True)


def test_word_only_embedding_is_unchanged_by_the_new_signature():
    """
    The default `add_context=False` path must build the exact same matrix as
    before the feature was added (this is what keeps recorded results stable).
    """
    ut, s, vt = calc_svd(wrap(DIAG), 4, "scipy", {})
    for eig in (0.0, 0.5, 1.0):
        old_style = SVDEmbedding(ut, s, eig=eig)
        new_style = SVDEmbedding(ut, s, eig=eig, vt=vt, add_context=False)
        assert np.allclose(old_style.m, new_style.m)


# 3CosMul (FEATURE 1)


def _cosmul_toy():
    """
    Five unit vectors laid out so 3CosAdd and 3CosMul disagree on the analogy
    ``a:a_ :: b:?`` (positives ``[a_, b]``, negative ``[a]``).

    ``a_ = e1``, ``b = e2``, ``a = e3`` (mutually orthogonal), and two candidate
    answers:

      * ``d1`` leans hard on one positive: cos(·,a_)=0.95, cos(·,b)=0.10
      * ``d2`` is balanced:                cos(·,a_)=0.50, cos(·,b)=0.50

    both orthogonal to ``a``. 3CosAdd sums the similarities (0.95+0.10=1.05 >
    0.50+0.50=1.00) and picks ``d1``; 3CosMul, after mapping cosines to [0,1],
    multiplies them and rewards the balanced ``d2``. So the two objectives return
    different answers -- computable by hand.
    """
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # a_  index 0
            [0.0, 1.0, 0.0, 0.0],  # b   index 1
            [0.0, 0.0, 1.0, 0.0],  # a   index 2
            [0.95, 0.10, 0.0, 0.29580399],  # d1  index 3
            [0.50, 0.50, 0.0, 0.70710678],  # d2  index 4
        ]
    )
    return SVDEmbedding(vectors, np.ones(4), eig=0.0)


def test_svd_cosmul_differs_from_cosadd():
    embd = _cosmul_toy()
    exclusions = {0, 1, 2}

    def first(objective):
        guesses = embd.most_similar_vectors([0, 1], [2], topn=4, objective=objective)
        return next(int(i) for i, _ in guesses if int(i) not in exclusions)

    assert first("add") == 3  # 3CosAdd -> d1 (dominant single similarity)
    assert first("mul") == 4  # 3CosMul -> d2 (balanced)


def test_most_similar_vectors_rejects_unknown_objective():
    embd = _cosmul_toy()
    with pytest.raises(ValueError, match="objective"):
        embd.most_similar_vectors([0], [1], objective="nonsense")


def test_cosmul_composes_with_wplusc():
    """
    The two features compose: 3CosMul runs on a `w+c` SVD embedding and returns a
    valid, correctly shaped ranking.
    """
    rng = np.random.default_rng(5)
    dense = rng.random((15, 15)).astype(np.float32)
    ut, s, vt = calc_svd(wrap(dense), 6, "scipy", {})
    wplusc = SVDEmbedding(ut, s, eig=0.0, vt=vt, add_context=True)

    guesses = wplusc.most_similar_vectors([1, 2], [0], topn=4, objective="mul")
    assert len(guesses) == 4
    assert all(0 <= int(i) < wplusc.m.shape[0] for i, _ in guesses)
    scores = [sc for _, sc in guesses]
    assert scores == sorted(scores, reverse=True)
