"""
Which SVD backend should you use?

    python bench/bench_svd.py [n_sentences] [vocab] [dims...]

`calc_svd` offers three backends and the package has never said which to pick:

    scipy    scipy.sparse.linalg.svds -- exact truncated SVD (ARPACK)
    gensim   gensim stochastic_svd    -- randomized
    scikit   sklearn randomized_svd   -- randomized, needs the `full` extra

Randomized SVD is an *approximation*, so the choice is not free: it trades
accuracy for time. This script measures both halves of that trade on real PPMI
matrices built by this package's own pipeline, so the spectrum is the one the
backends actually see rather than a random matrix's.

WHAT IS MEASURED, AND WHY THESE MEASURES
========================================

**Time** -- wall clock of `calc_svd` alone, best of several repeats.

**Fidelity, in the terms the package actually consumes.** Comparing raw
singular vectors would be misleading: they are only defined up to sign (and, for
tied singular values, up to a rotation within the tied subspace), so two correct
factorizations can differ arbitrarily column by column. What `hyperhyper`
*consumes* is the embedding built from them -- and after `SVDEmbedding`
normalizes rows, everything the package reports is a cosine between rows. So
fidelity is measured there:

    sv_rel_err   max relative error of the singular values vs the exact ones.
                 A direct measure of how well the randomized range-finder
                 captured the top-`dim` subspace.

    cos_spearman Spearman correlation between this backend's cosine similarities
                 and scipy's, over a fixed random sample of word pairs. This is
                 exactly the quantity `eval_similarity` correlates against human
                 scores, so a value near 1.0 means "this backend would report the
                 same word-similarity score".

    nn_overlap   mean overlap of the top-10 nearest neighbours with scipy's, over
                 a sample of words. `most_similar` is the other thing users read,
                 and it is harsher than the correlation: a small perturbation
                 reorders neighbours long before it moves a rank correlation.

`scipy` is the reference for the fidelity columns because it is the exact
truncated SVD -- not because it is "the right answer" in some larger sense.

WHAT THIS DOES NOT MEASURE
==========================
Not whether the *approximation* is worse for a downstream task. A randomized
backend could in principle score higher on word similarity than the exact one
(truncation is itself a denoising step, and there is no law saying the exact
top-`dim` subspace maximizes agreement with human judgement). Measuring that
needs a real corpus with real vocabulary and a gold set -- this script uses
synthetic Zipfian text, whose "words" have no meaning to correlate against. What
it answers is narrower and still worth knowing: **how much does the backend
change the answer, and what do you get for it.**
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hyperhyper import pmi, svd
from hyperhyper.pair_counts import make_count_closure

SEED = 20260722
REPEATS = 3
N_PAIR_SAMPLES = 20_000
N_NEIGHBOUR_SAMPLES = 300
TOP_N = 10


class _Vocab:
    def __init__(self, size):
        self.size = size


class _Corpus:
    """The bit of a corpus `make_count_closure` needs."""

    def __init__(self, size):
        self.vocab = _Vocab(size)


def build_ppmi(n_sentences, vocab_size, window=5):
    """A PPMI matrix from synthetic Zipfian text, via the package's own path."""
    rng = np.random.default_rng(SEED)
    ids = np.clip(rng.zipf(1.15, size=n_sentences * 15), 1, vocab_size) - 1
    texts, i = [], 0
    while i < len(ids):
        n = int(rng.integers(10, 25))
        texts.append([int(x) for x in ids[i : i + n]])
        i += n

    closure = make_count_closure(
        _Corpus(vocab_size),
        window=window,
        dynamic_window="deter",
        decay_rate=0.25,
        delete_oov=True,
        subsample=None,
        subsampler_prob=None,
        seed=SEED,
    )
    counts = closure.count_texts(texts, None)
    return pmi.PPMIEmbedding(pmi.calc_pmi(counts, cds=0.75), neg=1, normalize=False)


def embedding(ut, s):
    """What the user ends up holding: row-normalized vectors at the default eig=0."""
    return svd.SVDEmbedding(ut, s, eig=0.0, normalize=True).m


def fidelity(reference, candidate, ref_s, cand_s, rng):
    """Compare a backend's embedding with the exact one; see the module docstring."""
    k = min(len(ref_s), len(cand_s))
    sv_rel_err = float(
        np.max(np.abs(cand_s[:k] - ref_s[:k]) / np.maximum(ref_s[:k], 1e-30))
    )

    n_words = reference.shape[0]
    left = rng.integers(0, n_words, N_PAIR_SAMPLES)
    right = rng.integers(0, n_words, N_PAIR_SAMPLES)
    keep = left != right
    left, right = left[keep], right[keep]
    ref_cos = np.einsum("ij,ij->i", reference[left], reference[right])
    cand_cos = np.einsum("ij,ij->i", candidate[left], candidate[right])
    cos_spearman = float(spearmanr(ref_cos, cand_cos).statistic)

    probes = rng.choice(n_words, size=min(N_NEIGHBOUR_SAMPLES, n_words), replace=False)
    overlaps = []
    for w in probes:
        ref_top = np.argpartition(-(reference @ reference[w]), TOP_N + 1)[: TOP_N + 1]
        cand_top = np.argpartition(-(candidate @ candidate[w]), TOP_N + 1)[: TOP_N + 1]
        ref_top = set(ref_top) - {w}
        cand_top = set(cand_top) - {w}
        overlaps.append(len(ref_top & cand_top) / max(len(ref_top), 1))
    return sv_rel_err, cos_spearman, float(np.mean(overlaps))


def run(matrix, dim):
    results = {}
    for impl in ("scipy", "gensim", "scikit"):
        try:
            best, out = None, None
            for _ in range(REPEATS):
                start = time.perf_counter()
                out = svd.calc_svd(matrix, dim, impl, {})
                elapsed = time.perf_counter() - start
                best = elapsed if best is None else min(best, elapsed)
            results[impl] = (best, out)
        except ImportError as e:
            print(f"  {impl}: skipped ({e})")
    if "scipy" not in results:
        return

    ref_ut, ref_s, _ = results["scipy"][1]
    reference = embedding(ref_ut, ref_s)

    print(
        f"{'backend':10s} {'time':>8s} {'speedup':>8s} {'components':>11s} "
        f"{'sv_rel_err':>11s} {'cos_spearman':>13s} {'nn_overlap@10':>14s}"
    )
    base = results["scipy"][0]
    for impl, (elapsed, (ut, s, _vt)) in results.items():
        if impl == "scipy":
            print(
                f"{impl:10s} {elapsed:7.3f}s {1.0:7.2f}x {len(s):11d} "
                f"{'(exact)':>11s} {'(reference)':>13s} {'(reference)':>14s}"
            )
            continue
        err, corr, overlap = fidelity(
            reference, embedding(ut, s), ref_s, s, np.random.default_rng(SEED)
        )
        print(
            f"{impl:10s} {elapsed:7.3f}s {base / elapsed:7.2f}x {len(s):11d} "
            f"{err:11.2e} {corr:13.4f} {overlap:14.3f}"
        )


def main(argv):
    n_sentences = int(argv[0]) if len(argv) > 0 else 20_000
    vocab_size = int(argv[1]) if len(argv) > 1 else 5_000
    dims = [int(d) for d in argv[2:]] or [100, 300, 500]

    print("hyperhyper SVD backend benchmark")
    print(f"  seed        {SEED}")
    print(f"  sentences   {n_sentences}")
    print(f"  vocab       {vocab_size}")
    print(f"  repeats     {REPEATS} (best of)")
    start = time.perf_counter()
    matrix = build_ppmi(n_sentences, vocab_size)
    print(f"  PPMI matrix {matrix.m.shape}, nnz {matrix.m.nnz:,}")
    print(f"  build       {time.perf_counter() - start:.2f}s\n")

    for dim in dims:
        print(f"dim = {dim}")
        run(matrix, dim)
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
