"""
implements PMI matrix (Pointwise mutual information)
See: https://en.wikipedia.org/wiki/Pointwise_mutual_information
"""

import heapq

import numpy as np
from gensim import matutils
from scipy import sparse

# Epsilon for the 3CosMul denominator, shared with the SVD implementation and
# taken verbatim from Levy's ``hyperwords`` (see ``svd.COSMUL_EPS``).
from .svd import COSMUL_EPS


def calc_pmi(counts, cds):
    """
    Compute ``e^PMI`` from a word-context co-occurrence matrix.

    Returns the exponentiated pointwise mutual information

        e^PMI(w, c) = P(w, c) / (P(w) * P(c))
                    = count(w, c) * total / (count(w) * count(c))

    i.e. PMI without the final ``log()`` -- the caller (``PPMIEmbedding``)
    takes the log itself. Working in the exponentiated domain keeps the matrix
    sparse: a zero co-occurrence stays a stored zero rather than becoming -inf.

    ``cds`` is the context-distribution smoothing exponent (Levy & Goldberg
    2015): context counts are raised to the power ``cds`` before normalization,
    which flattens the context distribution and dampens the PMI of rare
    contexts. ``cds=1`` recovers plain PMI; ``cds=0.75`` is the common default.

    Args:
        counts: sparse word-by-context co-occurrence count matrix.
        cds (float): context-distribution smoothing exponent.

    Returns:
        Sparse matrix of ``e^PMI`` values, in the same container flavor
        (``spmatrix`` vs. sparse array) as ``counts``.
    """

    sum_w = np.asarray(counts.sum(axis=1)).ravel()
    sum_c = np.asarray(counts.sum(axis=0)).ravel()
    if cds != 1:
        sum_c = sum_c**cds
    sum_total = sum_c.sum()
    # guard against integer input (integer reciprocal is floor division) and
    # against empty rows/columns (which would yield inf -> nan)
    sum_w = np.divide(
        1.0, sum_w, out=np.zeros_like(sum_w, dtype=np.float64), where=sum_w > 0
    )
    sum_c = np.divide(
        1.0, sum_c, out=np.zeros_like(sum_c, dtype=np.float64), where=sum_c > 0
    )

    counts = counts.tocsr()
    # float32 is the floor (integer counts have to become floats), but a caller
    # who hands in float64 keeps float64 - hard-casting to float32 here would
    # compute at single precision while `* sum_total` promotes the result back
    # to float64, i.e. a silent precision loss behind an honest-looking dtype.
    dtype = np.result_type(counts.dtype, np.float32)
    counts = counts.astype(dtype, copy=False)
    D_w = sparse.diags_array(sum_w.astype(dtype, copy=False), format="csr")
    D_c = sparse.diags_array(sum_c.astype(dtype, copy=False), format="csr")
    pmi = D_w @ counts @ D_c * sum_total
    # `diags_array` is sparse-array flavored, so the product would come back as a
    # csr_array. PPMIEmbedding's row slicing needs the 2-D spmatrix semantics,
    # so hand back the same container the caller passed in.
    if isinstance(counts, sparse.spmatrix):
        pmi = sparse.csr_matrix(pmi)
    return pmi


class PPMIEmbedding:
    """
    Explicit (sparse, high-dimensional) word representation over SPPMI values.

    Each row is a word vector whose entries are shifted positive pointwise
    mutual information (SPPMI) with each context, following Levy & Goldberg
    (2015). The input ``matrix`` is assumed to hold ``e^PMI`` values (as
    produced by ``calc_pmi``); the constructor takes the ``log`` and applies
    the negative-sampling shift and the positive clip:

        SPPMI(w, c) = max(PMI(w, c) - log(neg), 0)

    ``neg`` is the number of negative samples: shifting PMI down by ``log(neg)``
    mimics word2vec's SGNS with ``neg`` negative samples and drives more entries
    to the zero clip, making the matrix sparser. ``neg=1`` leaves PMI unshifted
    (plain PPMI); ``neg=None`` also applies no shift. Regardless of ``neg``,
    negative values are always clipped to zero, so the result is genuinely
    *positive* PMI.

    Args:
        matrix: sparse ``e^PMI`` matrix (see ``calc_pmi``). Copied, not
            mutated in place.
        normalize (bool): if True, L2-normalize the rows so that similarity
            reduces to a dot product / cosine similarity.
        neg (int | None): negative-sampling shift; PMI is shifted by
            ``-log(neg)`` before clipping.
    """

    def __init__(self, matrix, normalize=True, neg=1):
        # copy: rebinding `.data` below would otherwise destroy the caller's
        # matrix in place. `bunch.pmi_matrix()` hands that very matrix back to
        # the user, and building a second embedding from it used to silently
        # compute log(log(...)) instead of raising.
        self.m = matrix.copy()
        # a stored zero legitimately means "no co-occurrence"; the -inf it
        # produces is clipped away below, so the warning is only noise
        with np.errstate(divide="ignore"):
            self.m.data = np.log(self.m.data)

        if neg is not None:
            self.m.data -= np.log(neg)

        # Always clip, never only in the `neg` branch: an explicitly stored zero
        # makes np.log produce -inf, and with `neg=None` that -inf used to
        # survive and wipe out the whole row in normalize(). Clipping also makes
        # the class live up to its name -- *P*PMI -- for every `neg`.
        self.m.data[self.m.data < 0] = 0
        self.m.eliminate_zeros()

        if normalize:
            self.normalize()

    def normalize(self):
        """
        L2-normalize each row (word vector) in place.

        After this, every non-zero row has unit Euclidean norm, so dot products
        between rows are cosine similarities. All-zero rows are left at zero.
        """
        m2 = self.m.copy()
        m2.data **= 2
        sums = np.sqrt(np.asarray(m2.sum(axis=1)).ravel())
        norm = np.divide(
            1.0, sums, out=np.zeros_like(sums, dtype=np.float64), where=sums > 0
        )
        normalizer = sparse.diags_array(
            norm.astype(self.m.dtype, copy=False), format="csr"
        )
        # keep the caller's container flavor, see the note in calc_pmi()
        was_matrix = isinstance(self.m, sparse.spmatrix)
        self.m = normalizer @ self.m
        if was_matrix:
            self.m = sparse.csr_matrix(self.m)

    def represent(self, w_idx):
        """
        Return the SPPMI row vector for the word at index ``w_idx`` as a
        1-by-vocabulary sparse row.
        """
        return self.m[w_idx, :]

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w1).dot(self.represent(w2).T)[0, 0]

    def most_similar(self, w, n=10):
        """
        Assumes the vectors have been normalized.

        Returns a list of `(index, score)`, sorted by descending score -- the
        same order as `most_similar_vectors`.
        """
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        # data and indices have equal length by the CSR invariant.
        # rank on the score, but hand back (index, score)
        best = heapq.nlargest(n, zip(scores.data, scores.indices, strict=True))
        return [(int(idx), float(score)) for score, idx in best]

    def most_similar_vectors(self, positives, negatives, topn=10, objective="add"):
        """
        Analogy-style query over the ``positives`` and ``negatives`` word indices.

        With ``objective="add"`` (the default, 3CosAdd) words are ranked by
        similarity to the unit-normalized mean of the ``positives`` vectors minus
        the ``negatives`` vectors -- the classic "king - man + woman" arithmetic.

        With ``objective="mul"`` the 3CosMul objective of Levy & Goldberg (2014)
        is used instead (see ``_most_similar_mul``). Both share the
        ``(index, score)`` return convention.

        Assumes the vectors have been normalized.

        Args:
            positives: word indices to add.
            negatives: word indices to subtract.
            topn (int): number of results to return.
            objective (str): ``"add"`` (3CosAdd) or ``"mul"`` (3CosMul).

        Returns:
            List of ``(index, score)`` tuples sorted by descending score --
            the same order and shape as ``most_similar``.

        Some parts taken from gensim.
        https://github.com/RaRe-Technologies/gensim/blob/ea87470e4c065676d3d33df15b8db4192b30ebc1/gensim/models/keyedvectors.py#L690
        """
        if objective == "mul":
            return self._most_similar_mul(positives, negatives, topn)
        if objective != "add":
            raise ValueError(
                f"objective must be 'add' (3CosAdd) or 'mul' (3CosMul), "
                f"got {objective!r}"
            )
        mean = [np.squeeze(self.represent(x).toarray()) for x in positives] + [
            -1 * np.squeeze(self.represent(x).toarray()) for x in negatives
        ]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

        dists = self.m.dot(mean)

        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]

    def _most_similar_mul(self, positives, negatives, topn):
        """
        3CosMul (Levy & Goldberg 2014), for the sparse SPPMI representation.

        Ranks each candidate ``d`` by

            (∏ cos(d, p) for p in positives) / (∏ cos(d, n) for n in negatives + ε)

        with ``ε = COSMUL_EPS``, reproducing Levy's ``hyperwords``
        ``analogy_eval.py`` scoring (``sa_ * sb * reciprocal(sa + 0.01)`` for the
        analogy ``a:a_ :: b:?`` with ``positives=[a_, b]``, ``negatives=[a]``).

        Unlike the dense SVD case, hyperwords does *not* remap sparse
        similarities via ``(cos + 1) / 2``: the SPPMI rows are non-negative and
        L2-normalized, so their cosine similarities already lie in ``[0, 1]``.
        """

        def sims(idx):
            # cos(d, idx) for every row d, as a dense 1-D array
            col = self.m.dot(self.represent(idx).T)
            return np.squeeze(np.asarray(col.todense()))

        dists = np.ones(self.m.shape[0], dtype=np.float64)
        for x in positives:
            dists = dists * sims(x)
        denom = np.ones(self.m.shape[0], dtype=np.float64)
        for x in negatives:
            denom = denom * sims(x)
        dists = dists / (denom + COSMUL_EPS)

        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
