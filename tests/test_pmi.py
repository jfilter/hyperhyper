import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hyperhyper.pmi import PPMIEmbedding, calc_pmi

# A 3x3 count matrix small enough to work out by hand.
# Its row sums are [3, 2, 4], its column sums [4, 3, 2] and its total is 9.
COUNTS = np.array(
    [
        [1.0, 2.0, 0.0],
        [0.0, 1.0, 1.0],
        [3.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# calc_pmi returns e^PMI, i.e. count[i, j] * total / (row_sum[i] * col_sum[j])
EXPECTED_PMI = np.array(
    [
        [1 * 9 / (3 * 4), 2 * 9 / (3 * 3), 0.0],
        [0.0, 1 * 9 / (2 * 3), 1 * 9 / (2 * 2)],
        [3 * 9 / (4 * 4), 0.0, 1 * 9 / (4 * 2)],
    ]
)


def test_calc_pmi_hand_computed():
    pmi = calc_pmi(csr_matrix(COUNTS), cds=1)
    assert np.allclose(pmi.toarray(), EXPECTED_PMI)


def test_calc_pmi_cds():
    """
    The context distribution smoothing exponent is applied to the column sums
    (and therefore also to the total they are summed to).
    """
    cds = 0.75
    sum_c = COUNTS.sum(axis=0) ** cds
    expected = COUNTS * sum_c.sum() / (COUNTS.sum(axis=1)[:, None] * sum_c[None, :])

    pmi = calc_pmi(csr_matrix(COUNTS), cds=cds)
    assert np.allclose(pmi.toarray(), expected)


def test_calc_pmi_keeps_float32():
    """
    The count matrix is the largest object in the pipeline; upcasting it to
    float64 doubles the memory use for nothing.
    """
    pmi = calc_pmi(csr_matrix(COUNTS), cds=0.75)
    assert pmi.dtype == np.float32


def test_calc_pmi_integer_counts():
    """
    Integer counts must not be silently floored to zero by an integer
    reciprocal.
    """
    pmi = calc_pmi(csr_matrix(COUNTS.astype(np.int64)), cds=1)
    assert np.allclose(pmi.toarray(), EXPECTED_PMI)


def test_calc_pmi_empty_row():
    """
    An all-zero row/column has a sum of zero; the reciprocal of that must not
    leak `inf`/`nan` into the matrix.
    """
    counts = COUNTS.copy()
    counts[1, :] = 0.0
    counts[:, 1] = 0.0

    pmi = calc_pmi(csr_matrix(counts), cds=1).toarray()
    assert np.isfinite(pmi).all()
    assert (pmi[1, :] == 0).all()
    assert (pmi[:, 1] == 0).all()


def test_ppmi_embedding_is_symmetric_and_normalized():
    embd = PPMIEmbedding(calc_pmi(csr_matrix(COUNTS), cds=1))
    for i in range(3):
        for j in range(3):
            assert embd.similarity(i, j) == pytest.approx(
                embd.similarity(j, i), rel=1e-5
            )
        # normalized rows have unit length, unless the row is empty
        assert embd.similarity(i, i) == pytest.approx(1.0, rel=1e-5)


def test_ppmi_most_similar_matches_similarity():
    embd = PPMIEmbedding(calc_pmi(csr_matrix(COUNTS), cds=1))
    for idx, score in embd.most_similar(0):
        assert embd.similarity(0, idx) == pytest.approx(score, rel=1e-5)


def test_ppmi_most_similar_returns_index_then_score():
    """
    Regression test for the tuple order: `most_similar` used to return
    `(score, index)` while `most_similar_vectors` returned `(index, score)`.
    Both now return `(index, score)`.
    """
    embd = PPMIEmbedding(calc_pmi(csr_matrix(COUNTS), cds=1))
    result = embd.most_similar(0, n=3)
    for idx, score in result:
        assert isinstance(idx, int)
        assert 0 <= idx < 3
        assert embd.similarity(0, idx) == pytest.approx(score, rel=1e-5)

    # the word itself comes first, at a similarity of 1
    assert result[0][0] == 0
    assert result[0][1] == pytest.approx(1.0, rel=1e-5)
    # and the scores are sorted descending
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_ppmi_most_similar_agrees_with_most_similar_vectors():
    """
    The two methods ask the same question, so with a consistent tuple order
    they must give the same answer.
    """
    embd = PPMIEmbedding(calc_pmi(csr_matrix(COUNTS), cds=1))
    by_index = embd.most_similar(1, n=3)
    by_vector = embd.most_similar_vectors([1], [], topn=3)
    assert [i for i, _ in by_index] == [i for i, _ in by_vector]
    for (_, a), (_, b) in zip(by_index, by_vector, strict=True):
        assert a == pytest.approx(b, rel=1e-5)


def test_ppmi_does_not_mutate_the_input_matrix():
    """
    Regression test: `__init__` used to bind the caller's matrix and rebind its
    `.data` to the log of itself, destroying it in place. `bunch.pmi_matrix()`
    hands that matrix to the user, so a second embedding built from it silently
    computed log(log(...)) and came out all zero instead of raising.

    counts [[8, 1], [1, 8]] have row/column sums of 9 and a total of 18, so
    e^PMI is [[8*18/81, 1*18/81], [1*18/81, 8*18/81]] = [[16/9, 2/9], ...].
    log(16/9) = 0.5753641... and log(2/9) < 0, which PPMI clips to 0.
    """
    counts = csr_matrix(np.array([[8.0, 1.0], [1.0, 8.0]], dtype=np.float32))
    pmi = calc_pmi(counts, cds=1)
    assert np.allclose(pmi.toarray(), [[16 / 9, 2 / 9], [2 / 9, 16 / 9]])

    before = pmi.toarray()
    expected = [[np.log(16 / 9), 0.0], [0.0, np.log(16 / 9)]]

    first = PPMIEmbedding(pmi, normalize=False)
    assert np.allclose(first.m.toarray(), expected)
    # the caller's matrix still holds e^PMI
    assert np.allclose(pmi.toarray(), before)

    # ... so a second embedding is identical, rather than log(log(...)) == 0
    second = PPMIEmbedding(pmi, normalize=False)
    assert np.allclose(second.m.toarray(), expected)


def test_ppmi_clips_stored_zeros_without_neg():
    """
    Regression test: `np.log` turns an explicitly stored zero into -inf. The
    clipping that removes it used to sit inside the `if neg is not None` branch,
    so with `neg=None` the -inf survived and wiped out the whole row in
    `normalize()`.

    Also pins the class down to *positive* PMI for every `neg`: a
    `PPMIEmbedding` that hands back signed PMI is a trap.
    """
    m = csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    m.data[1] = 0.0  # an explicitly stored zero, as pruning leaves behind
    assert (m.data == [1.0, 0.0, 3.0, 4.0]).all()

    embd = PPMIEmbedding(m.copy(), normalize=False, neg=None)
    assert np.isfinite(embd.m.data).all()
    # log(1) = 0 is dropped as well, so only log(3) and log(4) remain
    assert np.allclose(embd.m.toarray(), [[0.0, 0.0], [np.log(3), np.log(4)]])

    # and nothing negative leaks through, whatever `neg` is
    signed = csr_matrix(np.array([[0.5, 2.0], [3.0, 0.25]]))
    for neg in (None, 1, 5):
        e = PPMIEmbedding(signed.copy(), normalize=False, neg=neg)
        assert (e.m.data >= 0).all()

    # the -inf used to survive normalize() and destroy the row
    normalized = PPMIEmbedding(m.copy(), normalize=True, neg=None)
    assert np.isfinite(normalized.m.toarray()).all()


def _ppmi_with_rows(rows):
    """
    Build a `PPMIEmbedding` whose normalized rows are exactly ``rows``.

    ``rows`` must be non-negative and unit-norm. Passing ``e^row`` as the
    ``e^PMI`` input makes the constructor's ``log`` recover ``row``, the clip is a
    no-op (non-negative), and normalizing an already unit-norm row leaves it
    unchanged -- so we get an SPPMI embedding with hand-chosen cosine geometry
    through the real constructor.
    """
    data = np.where(rows > 0, np.exp(rows), 0.0)
    return PPMIEmbedding(csr_matrix(data))


def test_ppmi_cosmul_differs_from_cosadd():
    """
    3CosMul (Levy & Goldberg 2014) on the sparse SPPMI representation, with the
    answer worked out by hand.

    ``a_ = e1``, ``b = e2``, ``a = e3``; candidate ``d1`` (index 3) is dominated
    by a single positive (cos=0.95, 0.10) while ``d2`` (index 4) is balanced
    (0.50, 0.50). 3CosAdd sums (1.05 > 1.00) and picks ``d1``; 3CosMul multiplies
    (0.50*0.50 > 0.95*0.10) and picks ``d2``. Unlike the dense SVD case there is
    no ``(cos+1)/2`` remap -- SPPMI cosines are already non-negative.
    """
    rows = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # a_  index 0
            [0.0, 1.0, 0.0, 0.0],  # b   index 1
            [0.0, 0.0, 1.0, 0.0],  # a   index 2
            [0.95, 0.10, 0.0, 0.29580399],  # d1  index 3
            [0.50, 0.50, 0.0, 0.70710678],  # d2  index 4
        ]
    )
    embd = _ppmi_with_rows(rows)
    exclusions = {0, 1, 2}

    def first(objective):
        guesses = embd.most_similar_vectors([0, 1], [2], topn=4, objective=objective)
        return next(int(i) for i, _ in guesses if int(i) not in exclusions)

    assert first("add") == 3  # 3CosAdd -> d1
    assert first("mul") == 4  # 3CosMul -> d2


def test_ppmi_most_similar_vectors_rejects_unknown_objective():
    embd = PPMIEmbedding(calc_pmi(csr_matrix(COUNTS), cds=1))
    with pytest.raises(ValueError, match="objective"):
        embd.most_similar_vectors([0], [1], objective="nonsense")
