"""
FROZEN SNAPSHOT of the pair-counting implementation.

    source:  hyperhyper/pair_counts.py
    git SHA: 671c91b  (branch `modernize-2026`)

`hyperhyper/pair_counts.py` is unchanged from 671c91b through fa3127b (the
commits in between touch only pyproject.toml, uv.lock and CI), so this snapshot
is current for any of them. Verify with:

    git diff 671c91b HEAD -- hyperhyper/pair_counts.py

This file is a deliberate, verbatim-as-possible copy of the counting code as it
stood at that revision. It exists so that an optimized rewrite of
`hyperhyper/pair_counts.py` can be checked against the behaviour it is supposed
to preserve, by `tests/test_pair_counts_equivalence.py`.

RULES FOR THIS FILE
-------------------
* Do NOT "fix", tidy or modernize anything in here. Bugs are part of the
  snapshot; the point is to detect *changes*, not to be right.
* Do NOT import the counting logic from `hyperhyper.pair_counts`. The moment
  this file imports the thing it is supposed to check, it checks nothing.
  (`_default_workers` is the one exception, see `_merge_order` below.)
* Only update the snapshot together with a deliberate, reviewed decision that
  the output of `count_pairs` is *meant* to change -- and bump the SHA above.

WHY THE PARALLEL DRIVER IS REPLACED BY A SERIAL ONE
---------------------------------------------------
The live `count_pairs_parallel` fans the per-chunk work out to a process pool
and then adds the partial matrices back together in a fixed order. Spinning up
a second process pool for every one of the ~100 configurations in the
equivalence grid is slow, and pickling a closure defined in this module into
spawned workers is fragile.

Instead `_count_pairs_serial` below reproduces that summation order exactly, in
a single process. That matters for bit-identity because float32 addition is not
associative: adding the same chunk matrices in a different order gives a
different matrix. The order the live driver produces is

    * chunks are submitted in `texts_paths` order until more than
      MAX_JOBS_IN_QUEUE = _default_workers() * 2 are in flight,
    * that whole batch is drained before anything else is submitted,
    * and the drained batch is added in `sorted()` (lexicographic path) order.

so the accumulation order is: consecutive batches of `_default_workers() * 2 + 1`
chunks, each batch internally sorted by path. Note that lexicographic order is
*not* numeric order once a corpus has ten or more chunks (`texts_10.pkl` sorts
before `texts_2.pkl`), which is precisely why this is replicated rather than
assumed to be a no-op.
"""

import random
from collections import defaultdict
from math import e, fabs, sqrt
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix

# The one import from live code: this only decides how the chunk list is cut
# into batches, it is not part of the counting arithmetic, and the reference has
# to agree with whatever the live driver does on *this* machine for the
# summation order to line up.
from hyperhyper.utils import _default_workers, read_pickle

# --------------------------------------------------------------------------
# frozen: hyperhyper/pair_counts.py @ 671c91b
# --------------------------------------------------------------------------


def decay(distance, rate):
    """
    simple exponential decay
    """
    distance -= 1  # the returned value is 1 when the distance is 1
    return e ** -(rate * distance)


def to_count_matrix(pair_counts, vocab_size):
    """
    transforms the counts into a sparse matrix
    """
    cols = []
    rows = []
    data = []
    for k, v in pair_counts.items():
        rows.append(k[0])
        cols.append(k[1])
        data.append(v)
    count_matrix = coo_matrix(
        (data, (rows, cols)), shape=(vocab_size + 1, vocab_size + 1), dtype=np.float32
    )
    return count_matrix.tocsr()


def iterate_tokens(
    tokens,
    window,
    dynamic_window_prob,
    dynamic_window_deter,
    dynamic_window_decay,
    delete_oov,
    subsampler_prob,
    unkown_id,
    rng,
):
    """
    iterate over tokens in a sentence and counting pairs
    """
    if delete_oov:
        tokens = [t for t in tokens if t != unkown_id]

    if subsampler_prob is not None:
        tokens = [
            t
            if t not in subsampler_prob or rng.random() <= subsampler_prob[t]
            else None
            for t in tokens
        ]

    len_tokens = len(tokens)
    res = []
    for i, tok in enumerate(tokens):
        if tok is not None:
            offset = rng.randint(1, window) if dynamic_window_prob else window
            start = i - offset
            if start < 0:
                start = 0
            end = i + offset + 1
            if end > len_tokens:
                end = len_tokens
            for j in range(start, end):
                if j != i and tokens[j] is not None:
                    count = 1
                    # the variations are exclusive
                    if dynamic_window_deter:
                        distance = fabs(i - j)
                        count = (window + 1 - distance) / window
                    if dynamic_window_decay is not None:
                        distance = fabs(i - j)
                        count = decay(distance, dynamic_window_decay)
                    res.append((tok, tokens[j], count))
    return res


def subsample_keep_probabilities(counts, threshold):
    """
    The word2vec subsampling factor `sqrt(threshold / count)` for every word
    frequent enough to be subsampled at all.
    """
    return {
        word: sqrt(threshold / count)
        for word, count in counts.items()
        if count > threshold
    }


VALID_MODES = frozenset({"deter", "prob", "off", "decay"})


# --------------------------------------------------------------------------
# serial stand-in for count_pairs_parallel / CountPairsClosure
# --------------------------------------------------------------------------


def _count_one_chunk(text_path, params):
    """
    Mirrors `CountPairsClosure.__call__` @ 671c91b, including the per-file RNG
    derived from a stable string (so it does not depend on chunk *order*).
    """
    texts = read_pickle(text_path)
    rng = random.Random(f"{params['seed']}-{Path(text_path).name}")
    counter = defaultdict(int)
    for t in texts:
        for pair in iterate_tokens(
            t,
            params["window"],
            params["dynamic_window_prob"],
            params["dynamic_window_deter"],
            params["dynamic_window_decay"],
            params["delete_oov"],
            params["subsampler_prob"],
            params["vocab_size"],  # <UKN> id
            rng,
        ):
            counter[pair[0], pair[1]] += pair[2]
    return to_count_matrix(counter, params["vocab_size"])


def _merge_order(texts_paths):
    """
    The order in which `count_pairs_parallel` adds the per-chunk matrices.

    See the module docstring: batches of `_default_workers() * 2 + 1` chunks in
    submission order, each batch internally sorted by path.
    """
    batch = _default_workers() * 2 + 1
    order = []
    for start in range(0, len(texts_paths), batch):
        order.extend(sorted(texts_paths[start : start + batch]))
    return order


def _count_pairs_serial(texts_paths, params):
    res = None
    for path in _merge_order(list(texts_paths)):
        m = _count_one_chunk(path, params)
        if res is None:
            res = m
        else:
            res += m
    return res


def reference_count_pairs(
    corpus,
    window=2,
    dynamic_window="deter",
    decay_rate=0.25,
    delete_oov=True,
    subsample="deter",
    subsample_factor=1e-5,
    seed=1312,
    min_count=0,
):
    """
    Frozen copy of `count_pairs` @ 671c91b, with the process pool swapped for
    the equivalent serial accumulation.
    """
    for x in (dynamic_window, subsample):
        if x is not None and x is not False and x not in VALID_MODES:
            raise ValueError(
                f"expected one of {sorted(VALID_MODES)} or None/False, got {x!r}"
            )

    total_tokens = sum(corpus.counts.values())
    subsample_value = subsample_factor * total_tokens

    subsampler_prob = None
    if subsample == "prob":
        subsampler_prob = subsample_keep_probabilities(corpus.counts, subsample_value)

    count_matrix = _count_pairs_serial(
        corpus.texts,
        {
            "window": window,
            "dynamic_window_prob": dynamic_window == "prob",
            "dynamic_window_deter": dynamic_window == "deter",
            "dynamic_window_decay": decay_rate if dynamic_window == "decay" else None,
            "delete_oov": delete_oov,
            "subsampler_prob": subsampler_prob,
            "vocab_size": corpus.vocab.size,
            "seed": seed,
        },
    )

    if min_count is not None and min_count > 0:
        count_matrix.data *= count_matrix.data >= min_count
        count_matrix.eliminate_zeros()

    if subsample == "deter":
        subsampler = np.ones(corpus.vocab.size + 1, dtype=np.float32)
        keep = subsample_keep_probabilities(corpus.counts, subsample_value)
        for word, factor in keep.items():
            subsampler[word] = factor

        d = sparse.diags_array(subsampler, format="csr")
        scaled = d @ count_matrix.tocsr() @ d
        if isinstance(count_matrix, sparse.spmatrix):
            scaled = sparse.csr_matrix(scaled)
        count_matrix = scaled
    return count_matrix
