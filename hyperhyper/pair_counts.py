"""
construct a co-occurrence matrix by counting word pairs (co-locations of words)
"""

import logging
import random
import time
from collections import defaultdict
from concurrent import futures
from contextlib import closing
from itertools import islice
from math import e, fabs, sqrt
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix
from tqdm import tqdm

from .utils import _default_workers, read_pickle

logger = logging.getLogger(__name__)


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
    # setting to float is important, +1 for UNK
    # COO matrix is the fastest for constructing the matrix since we have all
    # the data already
    count_matrix = coo_matrix(
        (data, (rows, cols)), shape=(vocab_size + 1, vocab_size + 1), dtype=np.float32
    )
    # CSR matrices support more arithmetic operations and are more efficient
    return count_matrix.tocsr()


# What a `ProcessPoolExecutor` costs before it counts anything. A bare
# 10-worker pool is 0.1s; the rest is that every child has to import
# `hyperhyper` from scratch under macOS/spawn just to unpickle
# `CountPairsClosure`. Measured on a 10-core M1 Pro, Python 3.12:
#
#     7.0s   when importing the package still pulled in spacy eagerly
#     2.2s   now that `preprocessing` imports spacy lazily
#
# 3.0 is the measured 2.2 plus headroom for a cold page cache. It does not need
# to be accurate: it is compared against an estimate carrying a 1.3x margin, so
# it only decides the outcome for corpora within ~30% of the break-even, where
# the two routes are within a second of each other anyway. Being wrong either
# way costs about one pool startup.
POOL_STARTUP_SECONDS = 3.0

# How much faster the pool has to look before it is worth the risk. At 1.0 we
# would start a pool to save nothing.
POOL_SPEEDUP_MARGIN = 1.3

# Sentences counted in-process to estimate the per-sentence cost. Large enough
# to be above timer noise, small enough that the estimate is a rounding error
# against any corpus where the answer is not obvious.
PROBE_SENTENCES = 2000


def merge_order(texts_paths, workers):
    """
    The order the partial matrices are summed in -- a hard-wired contract, not
    an implementation detail.

    float32 addition is not associative, so the summation order is part of the
    answer: on this package's own corpora, reordering it moves ~500 cells of the
    matrix by ~2e-4 for any configuration whose counts are not integers
    (`dynamic_window` of "deter" with window>2, or "decay"). The order below --
    consecutive groups of `2 * workers + 1` files in corpus order, sorted within
    each group -- is what the previous scheduling loop happened to produce, and
    `tests/test_pair_counts_equivalence.py` holds the result to it bit for bit.

    Pinning it here is what makes the rest of this module free to schedule
    however it likes: *which* worker finishes *when* no longer has any influence
    on the result, because the merge no longer consults completion order at all.

    Known wart, inherited and deliberately not changed: `workers` is the local
    core count, so two machines with different core counts still produce
    matrices that differ in the last bits. Fixing that means picking a fixed
    group size, which would change every cached matrix.
    """
    group = workers * 2 + 1
    order = []
    for i in range(0, len(texts_paths), group):
        order.extend(sorted(texts_paths[i : i + group]))
    return order


def _estimate_serial_seconds(texts_paths, count_pairs_closure):
    """
    Guess what counting the whole corpus in this process would cost, by counting
    a slice of the first chunk and extrapolating.

    Measured rather than derived from the corpus size because the per-token cost
    is not a property of the corpus: on the same 5M-token corpus the default
    configuration runs at 4.3 us/token and `subsample="prob"` at 0.54 us/token,
    an 8x spread that no single token-count threshold can straddle. Getting this
    wrong in the "prob" direction is exactly the case the pool loses on.
    """
    texts = read_pickle(texts_paths[0])
    if not texts:
        return 0.0
    sample = texts[:PROBE_SENTENCES]
    # a throwaway RNG: the sample's counts are discarded, only its timing is used
    start = time.perf_counter()
    count_pairs_closure.count_texts(sample, random.Random(0))
    elapsed = time.perf_counter() - start
    # the last chunk may be short, which makes this a slight overestimate --
    # erring towards the pool, the right way round for large corpora
    total_sentences = len(texts) * len(texts_paths)
    return elapsed / len(sample) * total_sentences


def _pool_is_worth_starting(texts_paths, count_pairs_closure, workers):
    """
    Whether spreading the counting over a process pool beats just doing it here.
    """
    if len(texts_paths) < 2 or workers < 2:
        return False
    serial = _estimate_serial_seconds(texts_paths, count_pairs_closure)
    parallel = POOL_STARTUP_SECONDS + serial / min(workers, len(texts_paths))
    logger.info(
        "counting %s chunks: serial ~%.2fs, pool ~%.2fs",
        len(texts_paths),
        serial,
        parallel,
    )
    return parallel * POOL_SPEEDUP_MARGIN < serial


def _serial_results(texts_paths, count_pairs_closure):
    for path in texts_paths:
        yield path, count_pairs_closure(path)


def _parallel_results(texts_paths, count_pairs_closure, workers):
    """
    Yield `(path, matrix)` as workers finish, keeping `2 * workers` tasks in
    flight the whole time.

    The loop this replaced submitted a round of jobs, drained *all* of them, and
    only then submitted the next round, so every core sat idle from the moment
    it finished its last task of a round until the slowest task in that round
    came back. Topping the queue up on each individual completion removes the
    barrier; the window still bounds how many partial matrices can be resident,
    which is what the round-based version was really for.
    """
    with futures.ProcessPoolExecutor(workers) as executor:
        paths = iter(texts_paths)
        pending = {
            executor.submit(count_pairs_closure, p): p
            for p in islice(paths, workers * 2)
        }
        while pending:
            finished, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for job in finished:
                path = pending.pop(job)
                # re-arm before handing the result over: the consumer merges
                # between yields, and a worker must not wait on that
                for nxt in islice(paths, 1):
                    pending[executor.submit(count_pairs_closure, nxt)] = nxt
                yield path, job.result()


def count_pairs_parallel(texts_paths, count_pairs_closure):
    """
    Count pairs chunk by chunk, in a process pool when that pays for itself.

    Chunks are loaded from disk one at a time (rather than held in memory) and
    the partial matrices are merged and released as soon as a merge group is
    complete, so peak memory is set by the in-flight window, not by the corpus.
    """
    texts_paths = list(texts_paths)
    if not texts_paths:
        return None

    workers = _default_workers()
    order = merge_order(texts_paths, workers)
    group = workers * 2 + 1

    if _pool_is_worth_starting(texts_paths, count_pairs_closure, workers):
        produce = _parallel_results(texts_paths, count_pairs_closure, workers)
    else:
        produce = _serial_results(texts_paths, count_pairs_closure)

    res = None
    buffer = {}
    # `closing` so that an exception raised while merging tears the pool down
    # here, rather than leaving it to whenever the generator is collected
    with (
        tqdm(total=len(texts_paths), desc="generating pairs") as pbar,
        closing(produce) as results,
    ):
        # `order` is a concatenation of merge groups; consume just enough
        # results to complete the next group, merge it, and drop it again
        for i in range(0, len(order), group):
            merge_group = order[i : i + group]
            while any(path not in buffer for path in merge_group):
                path, matrix = next(results)
                buffer[path] = matrix
                pbar.update(1)
            for path in merge_group:
                matrix = buffer.pop(path)
                if res is None:
                    res = matrix
                else:
                    res += matrix
    return res


class CountPairsClosure:
    """
    creating a closure, has to be an object to be pickle-able when doing
    multiprocessing
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, text_path):
        texts = read_pickle(text_path)
        # a per-file RNG, derived from a stable string, so the randomized parts
        # are reproducible no matter how the workers are started (spawn does not
        # inherit the parent's RNG state) or in which order the futures complete
        rng = random.Random(f"{self.seed}-{Path(text_path).name}")
        return self.count_texts(texts, rng)

    def count_texts(self, texts, rng):
        """
        The counting itself, split out from `__call__` so that a caller can time
        a slice of a chunk without going near the filesystem or the RNG scheme.
        """
        counter = defaultdict(int)
        for t in texts:
            for pair in iterate_tokens(
                t,
                self.window,
                self.dynamic_window_prob,
                self.dynamic_window_deter,
                self.dynamic_window_decay,
                self.delete_oov,
                self.subsampler_prob,
                self.vocab_size,  # <UKN> id
                rng,
            ):
                counter[pair[0], pair[1]] += pair[2]
        return to_count_matrix(counter, self.vocab_size)


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


# storing the default values here again to re-use them when writing to the db
# TODO: implement in a more elegant way
default_pair_args = MappingProxyType(
    {
        "window": 2,
        "dynamic_window": "deter",
        "decay_rate": 0.25,
        "delete_oov": True,
        "subsample": "deter",
        "subsample_factor": 1e-5,
    }
)

VALID_MODES = frozenset({"deter", "prob", "off", "decay"})


def subsample_keep_probabilities(counts, threshold):
    """
    The word2vec subsampling factor `sqrt(threshold / count)` for every word
    frequent enough to be subsampled at all.

    This is the probability of *keeping* a token, so it falls as the count
    rises. `subsample="deter"` applies the very same factor, but by scaling the
    counts instead of by dropping tokens.
    """
    return {
        word: sqrt(threshold / count)
        for word, count in counts.items()
        if count > threshold
    }


def count_pairs(
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
    counting pairs in a corpus

    TODO: instead of giving a subsample_factor, give a portion of tokens to apply subsample
    """
    for x in (dynamic_window, subsample):
        if x is not None and x is not False and x not in VALID_MODES:
            raise ValueError(
                f"expected one of {sorted(VALID_MODES)} or None/False, got {x!r}"
            )

    # word2vec (and hyperwords) define the subsampling threshold as
    # `sample * train_words`, i.e. relative to the total number of TOKENS, and
    # Levy & Goldberg 3.1 put `t` on the same scale as the frequency `f` it is
    # compared against. `corpus.size` is the number of SENTENCES, so using it
    # here made the threshold too small by the average sentence length -- which
    # is not a harmless rescale: it changes *which* words get subsampled at all,
    # dragging rare words the paper leaves alone into an f^-1/2 reweighting.
    total_tokens = sum(corpus.counts.values())
    subsample_value = subsample_factor * total_tokens

    subsampler_prob = None
    if subsample == "prob":
        # `iterate_tokens` compares against this as a *keep* probability, so it
        # has to fall as the count rises. It used to be `1 - sqrt(...)`, the
        # word2vec *discard* probability, which kept frequent words and dropped
        # rare ones -- exactly backwards, and the opposite of "deter" below.
        subsampler_prob = subsample_keep_probabilities(corpus.counts, subsample_value)

    count_matrix = count_pairs_parallel(
        corpus.texts,
        CountPairsClosure(
            window=window,
            dynamic_window_prob=dynamic_window == "prob",
            dynamic_window_deter=dynamic_window == "deter",
            dynamic_window_decay=decay_rate if dynamic_window == "decay" else None,
            delete_oov=delete_oov,
            subsampler_prob=subsampler_prob,
            vocab_size=corpus.vocab.size,
            seed=seed,
        ),
    )

    # already prunning with a `min_count` of 1 can greatly reduces memory usage
    logger.info("Sparseness rate: %s", count_matrix.nnz / (corpus.vocab.size**2))
    if min_count is not None and min_count > 0:
        count_matrix.data *= count_matrix.data >= min_count
        count_matrix.eliminate_zeros()
        logger.info(
            "Sparseness rate after pruning: %s",
            count_matrix.nnz / (corpus.vocab.size**2),
        )

    # down sample in a deterministic way
    if subsample == "deter":
        # construct array with appropriate factor
        logger.info("creating array for the subsampling")
        subsampler = np.ones(corpus.vocab.size + 1, dtype=np.float32)
        # the identical factor the "prob" branch uses as a keep probability
        keep = subsample_keep_probabilities(corpus.counts, subsample_value)
        for word, factor in keep.items():
            subsampler[word] = factor
        num_sub = len(keep)
        logger.info(
            "subsampling applied to %s of the tokens", num_sub / corpus.vocab.size
        )

        logger.info("scaling with the subsampler: start")
        # `outer(s, s) * M` is the same as `diag(s) @ M @ diag(s)` but the latter
        # never materializes a dense (V+1)x(V+1) matrix
        d = sparse.diags_array(subsampler, format="csr")
        scaled = d @ count_matrix.tocsr() @ d
        # `diags_array` is sparse-array flavored, so the product comes back as a
        # csr_array. PPMIEmbedding relies on 2-D spmatrix row slicing, so keep
        # returning the same container flavor we were handed.
        if isinstance(count_matrix, sparse.spmatrix):
            scaled = sparse.csr_matrix(scaled)
        count_matrix = scaled
        logger.info("scaling with the subsampler: done")
    return count_matrix
