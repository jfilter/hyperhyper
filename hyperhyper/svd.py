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
    Factorize a (P)PMI matrix with truncated SVD, keeping ``dim`` components.

    Computes the rank-``dim`` truncated SVD ``M ≈ U · diag(s) · Vᵀ`` and returns
    the left singular vectors ``U`` together with the singular values ``s``.
    ``SVDEmbedding`` combines them (``U · diag(s)^eig``) into dense word vectors;
    this is the factorization step of Levy & Goldberg (2015).

    The singular values are always returned in descending order (``svds`` yields
    them ascending, so they are reordered to match ``gensim``/``scikit``).

    Args:
        matrix: object with an ``.m`` attribute holding the sparse matrix to
            factorize (e.g. a ``PPMIEmbedding``).
        dim (int): number of singular components to keep.
        impl (str): SVD backend -- ``"scipy"`` (exact truncated ``svds``),
            ``"gensim"`` (randomized ``stochastic_svd``, fast), or ``"scikit"``
            (randomized ``randomized_svd``; requires scikit-learn).
        impl_args (dict): extra keyword arguments forwarded to the backend.

    Returns:
        Tuple ``(ut, s)``: left singular vectors (``dim`` columns) and the
        matching singular values in descending order.

    Raises:
        ImportError: if ``impl="scikit"`` but scikit-learn is not installed.
        ValueError: if ``impl`` is not a recognized backend.

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
    Dense, low-dimensional word embedding from a truncated SVD factorization.

    Built from the left singular vectors ``ut`` and singular values ``s``
    returned by ``calc_svd``. Each row is a word vector formed by weighting the
    singular vectors with the singular values raised to the ``eig`` exponent
    (Levy & Goldberg 2015):

        W = U · diag(s) ** eig

    ``eig`` controls how much the singular values scale the components:

        * ``eig = 0.0`` -- ignore singular values, use the raw singular vectors
          ``U`` (often the best-performing choice for word similarity).
        * ``eig = 0.5`` -- symmetric weighting ``U · diag(s)^0.5``.
        * ``eig = 1.0`` -- full weighting ``U · diag(s)``, closest to the
          classic LSA representation.

    Args:
        ut: left singular vectors (``dim`` columns), one row per word.
        s: singular values.
        normalize (bool): if True, L2-normalize the rows so that dot products
            are cosine similarities.
        eig (float): exponent applied to the singular values before weighting.
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
        """
        L2-normalize each row (word vector) in place.

        After this, every non-zero row has unit Euclidean norm, so dot products
        between rows are cosine similarities. All-zero rows are left at zero
        rather than turned into ``nan``.
        """
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))[:, np.newaxis]
        # keep all-zero rows at zero instead of turning them into nan (which
        # would propagate into every later similarity computation)
        self.m = np.divide(self.m, norm, out=np.zeros_like(self.m), where=norm > 0)

    def represent(self, w_idx):
        """
        Return the dense embedding vector for the word at index ``w_idx``.
        """
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
        Analogy-style query: rank words by similarity to the mean of the
        ``positives`` vectors minus the ``negatives`` vectors.

        The (unit-normalized) mean of the selected word vectors is compared
        against every row; useful for "king - man + woman" style queries.
        Assumes the vectors have been normalized.

        Args:
            positives: word indices to add.
            negatives: word indices to subtract.
            topn (int): number of results to return.

        Returns:
            List of ``(index, score)`` tuples sorted by descending score --
            the same order and shape as ``most_similar``.
        """
        mean = [self.represent(x) for x in positives] + [
            -1 * self.represent(x) for x in negatives
        ]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        dists = self.m.dot(mean)
        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
