import heapq

import numpy as np
from scipy.sparse import csr_matrix, dok_matrix


class SVDEmbedding:
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    """

    def __init__(self, path, normalize=True, eig=0.0):
        ut = np.load(path + ".ut.npy")
        s = np.load(path + ".s.npy")

        if eig == 0.0:
            self.m = ut.T
        elif eig == 1.0:
            self.m = s * ut.T
        else:
            self.m = np.power(s, eig) * ut.T

        self.dim = self.m.shape[1]

        if normalize:
            self.normalize()

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]

    def represent(self, w_idx):
        return self.m[w_idx, :]

    def similarity(self, w_idx_1, w_idx_2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w_idx_1).dot(self.represent(w_idx_2))

    def closest(self, w_idx, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w_idx))
        # return heapq.nlargest(n, zip(scores, self.iw))
        return heapq.nlargest(n, zip(scores))


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

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        return heapq.nlargest(n, zip(scores.data, scores.indices))
