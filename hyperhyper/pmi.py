"""
implements PMI matrix (Pointwise mutual information)
See: https://en.wikipedia.org/wiki/Pointwise_mutual_information
"""

import heapq

import numpy as np
from gensim import matutils
from scipy import sparse


def calc_pmi(counts, cds):
    """
    Calculates e^PMI; PMI without the log().
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
    Base class for explicit representations. Assumes that the serialized input is e^PMI.

    Positive PMI (PPMI) with negative sampling (neg).
    Negative samples shift the PMI matrix before truncation.
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

    def most_similar_vectors(self, positives, negatives, topn=10):
        """
        Some parts taken from gensim.
        https://github.com/RaRe-Technologies/gensim/blob/ea87470e4c065676d3d33df15b8db4192b30ebc1/gensim/models/keyedvectors.py#L690
        """
        mean = [np.squeeze(self.represent(x).toarray()) for x in positives] + [
            -1 * np.squeeze(self.represent(x).toarray()) for x in negatives
        ]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

        dists = self.m.dot(mean)

        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
