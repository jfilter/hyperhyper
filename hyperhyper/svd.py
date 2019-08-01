import heapq
import logging

import numpy as np
from gensim.models.lsimodel import stochastic_svd
from scipy.sparse import csr_matrix, dok_matrix, linalg

logger = logging.getLogger(__name__)


try:
    from sparsesvd import sparsesvd
except ImportError:
    logger.info("no sparsvd")

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    logger.info("no sklearn")


# Checkout the comparision: https://github.com/jfilter/sparse-svd-benchmark
# https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
# https://pypi.org/project/sparsesvd/
# https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/lsimodel.py


def calc_svd(matrix, dim, impl, impl_args):
    if impl == "sparsesvd":
        # originally used SVD implementation
        ut, s, _ = sparsesvd(matrix.m.tocsc(), dim)
        # returns in a different format
        ut = ut.T
    if impl == "scipy":
        ut, s, _ = linalg.svds(matrix.m, dim)
    # randomized (but fast) truncated SVD
    if impl == "gensim":
        # better default arguments
        args = {"power_iters": 5, "extra_dims": 10, **impl_args}
        ut, s = stochastic_svd(matrix.m, dim, matrix.m.shape[0], **args)
    if impl == "scikit":
        ut, s, _ = randomized_svd(matrix.m, dim, **impl_args)

    return ut, s


class SVDEmbedding:
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    """

    def __init__(self, ut, s, normalize=True, eig=0.0):
        if eig == 0.0:
            self.m = ut
        elif eig == 1.0:
            self.m = s * ut
        else:
            self.m = np.power(s, eig) * ut

        # not used?
        # self.dim = self.m.shape[1]

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

    def most_similar(self, w_idx, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w_idx))
        return heapq.nlargest(n, zip(scores, list(range(len(scores)))))
