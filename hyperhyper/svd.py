"""
apply SVD on a PPMI matrix to get low-dimensional word embeddings
"""

import heapq
import logging

import numpy as np
from gensim import matutils
from gensim.models.lsimodel import stochastic_svd
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)


try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None
    logger.info("no sklearn")


def calc_svd(matrix, dim, impl, impl_args):
    """
    apply truncated SVD with several implementations

    truncated SVD:
    scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

    randomized truncated SVD:
    gensim: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/lsimodel.py
    scikit: https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

    Check out the comparision: https://github.com/jfilter/sparse-svd-benchmark
    """
    if impl == "scipy":
        ut, s, _ = svds(matrix.m, k=dim, **impl_args)
        # `svds` returns the singular values ascending, `gensim`/`scikit` return
        # them descending. SVDEmbedding is insensitive to the order (it scales
        # columns and normalizes rows), but anything slicing `ut[:, :k]`, reading
        # `s[0]` as the leading component or plotting the spectrum is not -- and
        # `bunch.svd_matrix` persists these arrays to disk.
        order = np.argsort(s)[::-1]
        ut, s = ut[:, order], s[order]
    # randomized (but fast) truncated SVD
    elif impl == "gensim":
        # better default arguments
        args = {"power_iters": 5, "extra_dims": 10, **impl_args}
        ut, s = stochastic_svd(matrix.m, dim, matrix.m.shape[0], **args)
    elif impl == "scikit":
        if randomized_svd is None:
            raise ImportError(
                "impl='scikit' requires scikit-learn: pip install 'hyperhyper[full]'"
            )
        ut, s, _ = randomized_svd(matrix.m, dim, **impl_args)
    else:
        raise ValueError(f"unknown SVD impl: {impl!r}")

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

        if normalize:
            self.normalize()

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))[:, np.newaxis]
        # keep all-zero rows at zero instead of turning them into nan (which
        # would propagate into every later similarity computation)
        self.m = np.divide(self.m, norm, out=np.zeros_like(self.m), where=norm > 0)

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

        Returns a list of `(index, score)`, sorted by descending score -- the
        same order as `most_similar_vectors`.
        """
        scores = self.m.dot(self.represent(w_idx))
        # rank on the score, but hand back (index, score)
        best = heapq.nlargest(n, zip(scores, range(len(scores)), strict=True))
        return [(int(idx), float(score)) for score, idx in best]

    def most_similar_vectors(self, positives, negatives, topn=10):
        """
        Assumes the vectors have been normalized.
        """
        mean = [self.represent(x) for x in positives] + [
            -1 * self.represent(x) for x in negatives
        ]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        dists = self.m.dot(mean)
        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
