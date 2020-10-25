"""
implements PMI matrix (Pointwise mutual information)
See: https://en.wikipedia.org/wiki/Pointwise_mutual_information
"""

import heapq

import numpy as np
from gensim import matutils
from scipy.sparse import csr_matrix, dok_matrix


def calc_pmi(counts, cds):
    """
    Calculates e^PMI; PMI without the log().
    """

    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]
    if cds != 1:
        sum_c = sum_c ** cds
    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = csr_matrix(counts)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi = pmi * sum_total
    return pmi


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


class PPMIEmbedding:
    """
    Base class for explicit representations. Assumes that the serialized input is e^PMI.

    Positive PMI (PPMI) with negative sampling (neg).
    Negative samples shift the PMI matrix before truncation.
    """

    def __init__(self, matrix, normalize=True, neg=1):
        self.m = matrix
        self.m.data = np.log(self.m.data)

        # not needed?
        # # self.normal = normalize

        if neg is not None:
            self.m.data -= np.log(neg)
            self.m.data[self.m.data < 0] = 0
            self.m.eliminate_zeros()

        if normalize:
            self.normalize()

    def normalize(self):
        m2 = self.m.copy()
        m2.data **= 2
        norm = np.reciprocal(np.sqrt(np.array(m2.sum(axis=1))[:, 0]))
        normalizer = dok_matrix((len(norm), len(norm)))
        normalizer.setdiag(norm)
        self.m = normalizer.tocsr().dot(self.m)

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
        """
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        return heapq.nlargest(n, zip(scores.data, scores.indices))


    # TODO: working?
    def most_similar_vectors(self, positives, negatives, topn=10):
        """
        Some parts taken from gensim.
        https://github.com/RaRe-Technologies/gensim/blob/ea87470e4c065676d3d33df15b8db4192b30ebc1/gensim/models/keyedvectors.py#L690
        """
        mean = [np.squeeze(self.represent(x).toarray()) for x in positives] + [-1 * np.squeeze(self.represent(x).toarray()) for x in negatives]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

        dists = self.m.dot(mean)

        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
