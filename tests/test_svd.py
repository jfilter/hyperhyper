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
