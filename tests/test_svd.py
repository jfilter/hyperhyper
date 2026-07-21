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
    _, s = calc_svd(wrap(DIAG), 3, impl, {})
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
    ut, _ = calc_svd(wrap(DIAG), 3, impl, {})
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
    ut, s = calc_svd(wrap(matrix), dim, impl, {})
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
    ut, s = calc_svd(wrap(_rank_deficient()), 15, impl, {})
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
    ut, s = calc_svd(wrap(full_rank), 10, impl, {})
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
    ut, s = calc_svd(wrap(matrix), 5, impl, {})
    assert ut.shape[1] == 5
    assert np.allclose(np.sort(s)[::-1], reference, atol=1e-4)


def _embedding():
    rng = np.random.default_rng(0)
    dense = rng.random((20, 20)).astype(np.float32)
    ut, s = calc_svd(wrap(dense), 5, "scikit", {})
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
