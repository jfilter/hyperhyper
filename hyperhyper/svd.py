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


# Epsilon added to the 3CosMul denominator. Taken verbatim from Levy's
# ``hyperwords/analogy_eval.py`` (``guess``), which computes
# ``mul_sim = sa_ * sb * np.reciprocal(sa + 0.01)`` -- i.e. 0.01, *not* the
# 0.001 that some descriptions of 3CosMul quote.
COSMUL_EPS = 0.01


def calc_svd(matrix, dim, impl, impl_args):
    """
    Factorize a (P)PMI matrix with truncated SVD, keeping ``dim`` components.

    Computes the rank-``dim`` truncated SVD ``M ≈ U · diag(s) · Vᵀ`` and returns
    the left singular vectors ``U``, the singular values ``s`` and the right
    singular vectors ``Vᵀ``. ``SVDEmbedding`` combines them (``U · diag(s)^eig``,
    optionally plus ``V · diag(s)^eig`` for the ``w+c`` representation) into
    dense word vectors; this is the factorization step of Levy & Goldberg (2015).

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

    ``dim`` is clamped to ``min(shape) - 1`` and any component whose singular
    value is numerical noise is dropped (see ``_drop_negligible_components``), so
    the result may have *fewer* than ``dim`` columns for a rank-deficient or
    over-large request; a full-rank ``dim < rank`` request is untouched.

    Returns:
        Tuple ``(ut, s, vt)``: left singular vectors (up to ``dim`` columns), the
        matching singular values in descending order, and the right singular
        vectors ``Vᵀ`` (up to ``dim`` rows). ``gensim``'s ``stochastic_svd`` does
        not return ``Vᵀ``, so for that backend it is reconstructed from the
        identity ``Vᵀ = diag(s)⁻¹ · Uᵀ · M`` (see ``_right_vectors_from_left``).

    Raises:
        ImportError: if ``impl="scikit"`` but scikit-learn is not installed.
        ValueError: if ``impl`` is not a recognized backend, or the matrix is
            empty/all-zero/numerically rank 0.

    truncated SVD:
    scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

    randomized truncated SVD:
    gensim: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/lsimodel.py
    scikit: https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html

    Check out the comparision: https://github.com/jfilter/sparse-svd-benchmark
    """
    # Clamp the request *here*, at the single point all three backends pass
    # through, rather than in `bunch.svd_matrix`: `calc_svd` is also called
    # directly (the tests do, with a bare `.m` wrapper and no `Bunch`), and each
    # backend reacts differently to an over-large `dim` -- `svds` raises
    # `ValueError: k must satisfy 0 < k < min(A.shape)`, while gensim and scikit
    # silently return a *different* number of columns. Clamping to
    # `min(shape) - 1` (the most ARPACK can resolve) makes all three agree on the
    # effective dim, and the arrays `svd_matrix` keys on disk with it. The cache
    # path stays keyed on the *requested* dim, which is harmless: a clamped run
    # is deterministic, so it always reproduces the same cached arrays.
    n_rows, n_cols = matrix.m.shape
    max_dim = min(n_rows, n_cols) - 1
    if max_dim < 1:
        raise ValueError(
            f"cannot factorize a {n_rows}x{n_cols} matrix: at least a 2x2 "
            f"matrix is needed to extract one singular component"
        )
    if matrix.m.nnz == 0:
        # An all-zero (S)PPMI matrix (e.g. a `neg` large enough to shift every
        # entry below the clip) has no singular directions at all; `svds` would
        # raise the opaque `ArpackError: Starting vector is zero`.
        raise ValueError(
            "cannot factorize an all-zero matrix (every singular value is "
            "zero); check `neg`/`min_count`, which may have emptied the "
            "(S)PPMI matrix"
        )
    dim = min(dim, max_dim)

    if impl == "scipy":
        ut, s, vt = svds(matrix.m, k=dim, **impl_args)
        # `svds` returns the singular values ascending, `gensim`/`scikit` return
        # them descending. SVDEmbedding is insensitive to the order (it scales
        # columns and normalizes rows), but anything slicing `ut[:, :k]`, reading
        # `s[0]` as the leading component or plotting the spectrum is not -- and
        # `bunch.svd_matrix` persists these arrays to disk. The right vectors
        # `vt` have to follow the same reordering as `ut`/`s`, row for row.
        order = np.argsort(s)[::-1]
        ut, s, vt = ut[:, order], s[order], vt[order, :]
    # randomized (but fast) truncated SVD
    elif impl == "gensim":
        # better default arguments
        args = {"power_iters": 5, "extra_dims": 10, **impl_args}
        ut, s = stochastic_svd(matrix.m, dim, matrix.m.shape[0], **args)
        # `stochastic_svd` returns only `(U, s)`; recover the right singular
        # vectors so the `w+c` representation works on this backend too.
        vt = _right_vectors_from_left(ut, s, matrix.m)
    elif impl == "scikit":
        if randomized_svd is None:
            raise ImportError(
                "impl='scikit' requires scikit-learn: pip install 'hyperhyper[full]'"
            )
        ut, s, vt = randomized_svd(matrix.m, dim, **impl_args)
    else:
        raise ValueError(f"unknown SVD impl: {impl!r}")

    return _drop_negligible_components(ut, s, vt, matrix.m)


def _right_vectors_from_left(ut, s, m):
    """
    Recover the right singular vectors ``Vᵀ`` from ``U``, ``s`` and ``M``.

    ``gensim``'s ``stochastic_svd`` returns only the left factors ``(U, s)``. For
    a truncated SVD ``M = U · diag(s) · Vᵀ`` the identity ``Uᵀ · M = diag(s) · Vᵀ``
    holds, so ``Vᵀ = diag(s)⁻¹ · Uᵀ · M`` reconstructs the right vectors that are
    consistent with the ``U``/``s`` gensim returned. Components with a zero
    singular value (numerical null space, dropped downstream anyway) map to a
    zero row rather than a division by zero.
    """
    # `M.T · U` is `(n_context x dim)` and dense; transpose to the `Vᵀ` layout.
    sv_t = np.asarray(m.T.dot(ut)).T  # == diag(s) · Vᵀ, shape (dim, n_context)
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
    return (inv_s[:, None] * sv_t).astype(ut.dtype, copy=False)


def _drop_negligible_components(ut, s, vt, m):
    """
    Discard singular vectors whose singular value is numerical noise.

    When ``dim`` exceeds the matrix's numerical rank, the surplus singular values
    are ~0 and their singular *vectors* are arbitrary null-space directions: a
    different one every time, because ``svds`` (ARPACK) starts from an unseeded
    random vector. Under the default ``eig=0`` ``SVDEmbedding`` uses those columns
    verbatim, so the same matrix and parameters gave *different* nearest
    neighbours run to run, and the three backends disagreed.

    Dropping the negligible columns -- matching gensim's own truncation -- makes
    ``eig=0`` deterministic (pairwise similarities are invariant to the
    orthonormal basis and to the per-run sign/rotation of the kept subspace, so
    once every backend keeps the *same* set of significant directions they agree)
    and cannot re-enter regardless of ``eig``.

    The tolerance is ``s.max() * max(shape) * eps`` -- the standard numerical-rank
    threshold (cf. ``numpy.linalg.matrix_rank``). ``eps`` is taken at the matrix's
    own precision (float32 for a PPMI matrix), which is where the ~1e-6 noise
    floor actually comes from. For a full-rank ``dim < rank`` request no column
    is negligible and the arrays are returned bit-unchanged.

    The right singular vectors ``vt`` (rows indexed by component) are truncated
    to exactly the same kept set, so ``U``, ``s`` and ``Vᵀ`` always describe the
    same subspace -- the ``w+c`` representation depends on it.
    """
    if s.size == 0:
        raise ValueError("SVD returned no components")
    dtype = m.dtype
    eps = np.finfo(dtype if np.issubdtype(dtype, np.floating) else np.float32).eps
    tol = s.max() * max(m.shape) * eps
    keep = s > tol
    if not keep.any():
        raise ValueError(
            "the matrix is numerically rank 0 (all singular values are "
            "negligible); there is nothing to embed"
        )
    if keep.all():
        # full rank at this dim -- hand back the exact arrays, untouched
        return ut, s, vt
    return ut[:, keep], s[keep], vt[keep, :]


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

    With ``add_context=True`` the ``w+c`` representation of Levy & Goldberg
    (2015) is built instead: the context vectors are added to the word vectors,

        W = U · diag(s) ** eig + V · diag(s) ** eig

    (the sum is normalized afterwards, if ``normalize``). This requires the right
    singular vectors ``vt`` (``= Vᵀ``). The paper applies ``w+c`` only to dense
    SVD embeddings, never to PPMI.

    Args:
        ut: left singular vectors (``dim`` columns), one row per word.
        s: singular values.
        normalize (bool): if True, L2-normalize the rows so that dot products
            are cosine similarities.
        eig (float): exponent applied to the singular values before weighting.
        vt: right singular vectors ``Vᵀ`` (``dim`` rows), needed only for
            ``add_context``.
        add_context (bool): if True, add the context vectors ``V · diag(s)^eig``
            to the word vectors (the ``w+c`` representation).
    """

    def __init__(self, ut, s, normalize=True, eig=0.0, vt=None, add_context=False):
        self.m = self._scale(ut, s, eig)
        if add_context:
            if vt is None:
                raise ValueError(
                    "add_context=True needs the context vectors `vt`; obtain "
                    "them from calc_svd (which now returns `(ut, s, vt)`) or "
                    "Bunch.svd(add_context=True)"
                )
            # `vt` is (dim x n_context); its transpose holds one context vector
            # per row, aligned with the word rows of `ut`. w+c = W + C.
            self.m = self.m + self._scale(vt.T, s, eig)

        if normalize:
            self.normalize()

    @staticmethod
    def _scale(u, s, eig):
        """
        Weight singular vectors ``u`` (rows) by ``diag(s) ** eig``.

        ``eig == 0.0`` returns ``u`` untouched (the array is not copied, matching
        the original behaviour), ``eig == 1.0`` avoids the ``pow``.
        """
        if eig == 0.0:
            return u
        if eig == 1.0:
            return s * u
        return np.power(s, eig) * u

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

    def most_similar_vectors(self, positives, negatives, topn=10, objective="add"):
        """
        Analogy-style query over the ``positives`` and ``negatives`` word indices.

        With ``objective="add"`` (the default, 3CosAdd) words are ranked by
        similarity to the unit-normalized mean of the ``positives`` vectors minus
        the ``negatives`` vectors -- the classic "king - man + woman" arithmetic.

        With ``objective="mul"`` the 3CosMul objective of Levy & Goldberg (2014)
        is used instead (see ``_most_similar_mul``); it is often materially better
        on analogies. Both share the ``(index, score)`` return convention.

        Assumes the vectors have been normalized.

        Args:
            positives: word indices to add.
            negatives: word indices to subtract.
            topn (int): number of results to return.
            objective (str): ``"add"`` (3CosAdd) or ``"mul"`` (3CosMul).

        Returns:
            List of ``(index, score)`` tuples sorted by descending score --
            the same order and shape as ``most_similar``.
        """
        if objective == "mul":
            return self._most_similar_mul(positives, negatives, topn)
        if objective != "add":
            raise ValueError(
                f"objective must be 'add' (3CosAdd) or 'mul' (3CosMul), "
                f"got {objective!r}"
            )
        mean = [self.represent(x) for x in positives] + [
            -1 * self.represent(x) for x in negatives
        ]
        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        dists = self.m.dot(mean)
        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]

    def _most_similar_mul(self, positives, negatives, topn):
        """
        3CosMul (Levy & Goldberg 2014), for dense SVD vectors.

        Ranks each candidate ``d`` by the multiplicative combination

            (∏ cos(d, p) for p in positives) / (∏ cos(d, n) for n in negatives + ε)

        with ``ε = COSMUL_EPS``. This is exactly Levy's ``hyperwords``
        ``analogy_eval.py`` scoring: for the analogy ``a:a_ :: b:?`` it passes
        ``positives=[a_, b]``, ``negatives=[a]`` and computes
        ``sa_ * sb * reciprocal(sa + 0.01)``.

        hyperwords maps *dense* cosine similarities from ``[-1, 1]`` onto
        ``[0, 1]`` via ``(cos + 1) / 2`` before multiplying (its ``else`` branch
        in ``prepare_similarities``), so that a single strongly negative cosine
        cannot dominate or flip the sign of the product. That remapping is
        reproduced here; the PPMI variant does not remap, matching hyperwords.
        """

        def sims(idx):
            return (self.m.dot(self.represent(idx)) + 1.0) / 2.0

        dists = np.ones(self.m.shape[0], dtype=np.float64)
        for x in positives:
            dists = dists * sims(x)
        denom = np.ones(self.m.shape[0], dtype=np.float64)
        for x in negatives:
            denom = denom * sims(x)
        dists = dists / (denom + COSMUL_EPS)

        best = matutils.argsort(dists, topn=topn, reverse=True)
        return [(best_idx, float(dists[best_idx])) for best_idx in best]
