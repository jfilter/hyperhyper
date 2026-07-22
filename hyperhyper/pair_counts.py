"""
construct a co-occurrence matrix by counting word pairs (co-locations of words)
"""

import inspect
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

from .utils import (
    _MISSING_MAIN_GUARD,
    BrokenProcessPool,
    _default_workers,
    read_pickle,
)

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

# Ceiling on the events (word, context, weight triples) the vectorized counter
# will materialize at once. It holds several arrays of this length -- positions,
# distances, keys, the `np.unique` sort -- so the cap is really a memory budget,
# roughly 300-400 MB at this value, *per worker*.
#
# A chunk over the cap falls back to the Python loop, which streams. That path
# is slower but produces the identical matrix, so the fallback is invisible in
# the result. `_auto_text_chunk_size` sizes chunks well under this for any
# corpus it controls; the cap is there for an explicit `text_chunk_size` or an
# input-file-shaped chunk that is far larger.
MAX_VECTORIZED_EVENTS = 5_000_000

# How much faster the pool has to look before it is worth the risk. At 1.0 we
# would start a pool to save nothing.
POOL_SPEEDUP_MARGIN = 1.3

# Sentences counted in-process to estimate the per-sentence cost. Large enough
# to be above timer noise, small enough that the estimate is a rounding error
# against any corpus where the answer is not obvious.
PROBE_SENTENCES = 2000


def merge_order(texts_paths):
    """
    The order the partial matrices are summed in -- a hard-wired contract, not
    an implementation detail.

    float32 addition is not associative, so the summation order is part of the
    answer: on this package's own corpora, reordering it moves ~500 cells of the
    matrix by up to ~2e-4 (about 2 ulps, ~3e-7 relative) for any configuration
    whose counts are not integers (`dynamic_window` of "deter" with window>2, or
    "decay"). Configurations whose counts are exact in float32 -- window=1, or
    `dynamic_window=None`, or window=2 with "deter", whose counts are multiples
    of 0.5 -- are unaffected by any reordering at all.

    Sorting by path is one single canonical order: it depends on nothing but the
    chunk file names, so it is the same on every machine, on every run, at any
    core count. That is exact, not probabilistic -- the same numbers are added in
    the same sequence everywhere.

    Pinning it here is what makes the rest of this module free to schedule
    however it likes: *which* worker finishes *when* has no influence on the
    result, because the merge does not consult completion order at all. Note
    that lexicographic order is not numeric order once a corpus has ten or more
    chunks (`texts_10.pkl` sorts before `texts_2.pkl`); that is fine, all that is
    required of the order is that it be fixed.
    """
    return sorted(texts_paths)


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
                try:
                    result = job.result()
                except futures.process.BrokenProcessPool as e:
                    # the stdlib message names a symptom and gives the reader
                    # nothing to act on; see `utils.BrokenProcessPool`
                    raise BrokenProcessPool(_MISSING_MAIN_GUARD) from e
                yield path, result


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
    order = merge_order(texts_paths)
    # purely a memory knob: the merge is a strict left fold over `order`, so how
    # that order is sliced into groups changes only how many partial matrices
    # are resident at once, never the sum
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


def make_count_closure(
    corpus,
    *,
    window,
    dynamic_window,
    decay_rate,
    delete_oov,
    subsample,
    subsampler_prob,
    seed,
):
    """
    Translate `count_pairs`' mode strings into a worker closure.

    This exists so there is exactly **one** place that turns the public
    parameters into the booleans `iterate_tokens` reads. It used to be inline in
    `count_pairs`, with `bench/bench_pair_counts.py` keeping a hand-written copy
    whose docstring said "if the rewrite changes how a worker is configured, fix
    it here" -- an invitation to drift that was duly accepted: when the `dirty`
    subsampling variant landed, the benchmark's copy was not updated and every
    run died with `AttributeError: 'CountPairsClosure' object has no attribute
    'subsampler_dirty'`. A benchmark that cannot run is worse than none, because
    it looks like a safety net.
    """
    return CountPairsClosure(
        window=window,
        dynamic_window_prob=dynamic_window == "prob",
        dynamic_window_deter=dynamic_window == "deter",
        dynamic_window_decay=decay_rate if dynamic_window == "decay" else None,
        delete_oov=delete_oov,
        subsampler_prob=subsampler_prob,
        subsampler_dirty=subsample == "dirty",
        vocab_size=corpus.vocab.size,
        seed=seed,
    )


class CountPairsClosure:
    """
    creating a closure, has to be an object to be pickle-able when doing
    multiprocessing

    The fields are named explicitly rather than swallowed through `**kwargs`.
    A `self.__dict__.update(kwargs)` accepts a caller that forgets a field and
    defers the failure to the middle of counting -- possibly inside a worker
    process, where the traceback is least useful. Naming them makes an
    out-of-date construction site fail at construction, with the missing
    argument named.
    """

    def __init__(
        self,
        *,
        window,
        dynamic_window_prob,
        dynamic_window_deter,
        dynamic_window_decay,
        delete_oov,
        subsampler_prob,
        subsampler_dirty,
        vocab_size,
        seed,
    ):
        self.window = window
        self.dynamic_window_prob = dynamic_window_prob
        self.dynamic_window_deter = dynamic_window_deter
        self.dynamic_window_decay = dynamic_window_decay
        self.delete_oov = delete_oov
        self.subsampler_prob = subsampler_prob
        self.subsampler_dirty = subsampler_dirty
        self.vocab_size = vocab_size
        self.seed = seed

    def __call__(self, text_path):
        texts = read_pickle(text_path)
        # a per-file RNG, derived from a stable string, so the randomized parts
        # are reproducible no matter how the workers are started (spawn does not
        # inherit the parent's RNG state) or in which order the futures complete
        rng = random.Random(f"{self.seed}-{Path(text_path).name}")
        return self.count_texts(texts, rng)

    def uses_rng(self):
        """
        Whether counting this configuration draws random numbers.

        `subsample="deter"` does not: it is a post-hoc scaling of the finished
        matrix, so it never reaches `iterate_tokens`. Only `"prob"`/`"dirty"`
        (which set `subsampler_prob`) and `dynamic_window="prob"` draw.
        """
        return self.subsampler_prob is not None or self.dynamic_window_prob

    def is_vectorizable(self):
        """
        Whether `count_texts_vectorized` can handle this configuration.

        Two families qualify, and what they have in common is that the *draw
        stream* can be reproduced exactly -- either because there is none, or
        because it is one draw per token in token order, which a list
        comprehension over the flattened chunk gives verbatim:

        * everything deterministic (no draws at all);
        * `dynamic_window="prob"` **without** subsampling: exactly one
          `randint(1, window)` per token, in order.

        `subsample="prob"`/`"dirty"` stay on the loop. Not because their stream
        could not be reproduced -- it could, with more care, since the
        subsampling draws for a sentence all precede its radius draws -- but
        because they are already the *fastest* configurations by a wide margin:
        they discard so many tokens that few pairs survive, so vectorizing them
        would optimize the cheap case and buy a second code path to keep in
        sync.
        """
        return not self.uses_rng() or (
            self.dynamic_window_prob and self.subsampler_prob is None
        )

    def count_texts(self, texts, rng):
        """
        The counting itself, split out from `__call__` so that a caller can time
        a slice of a chunk without going near the filesystem or the RNG scheme.

        Deterministic configurations take a vectorized path that produces a
        bit-identical matrix several times faster; everything else keeps the
        Python loop. See `count_texts_vectorized`.
        """
        if self.is_vectorizable():
            vectorized = self.count_texts_vectorized(texts, rng)
            if vectorized is not None:
                return vectorized

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
                self.subsampler_dirty,
                self.vocab_size,  # <UKN> id
                rng,
            ):
                counter[pair[0], pair[1]] += pair[2]
        return to_count_matrix(counter, self.vocab_size)

    def count_texts_vectorized(self, texts, rng=None):
        """
        Count a whole chunk with numpy instead of the per-token Python loop.

        Returns the same CSR matrix as the loop, **bit for bit**, or `None` when
        the chunk is too large to hold its event arrays (the caller then falls
        back to the loop, which is slower but streams).

        Handles the configurations `is_vectorizable` accepts: everything
        deterministic, plus `dynamic_window="prob"` without subsampling.

        The `"prob"` window keeps its exact draw stream rather than switching to
        a numpy generator. `iterate_tokens` draws one `randint(1, window)` per
        token in token order, and a comprehension over the flattened chunk draws
        the same numbers in the same order -- so this stays bit-identical to the
        loop *and* to every result recorded before it existed, which a numpy RNG
        could never be. The draws themselves are not the expensive part
        (~0.08s per 200k tokens against ~0.9s for the emission they gate).

        How bit-identity is kept
        ------------------------
        Two things do it, and both are easy to lose:

        * **Accumulate in float64, cast once at the end.** The loop accumulates
          into `defaultdict(int)` with *Python* floats -- which are float64 --
          and only `to_count_matrix` narrows to float32. Accumulating in float32
          here would round at every addition instead of once at the end, and the
          matrix would differ in the low bits.
        * **Emit events in the loop's order.** float addition is not
          associative, so the sequence of additions *to a given cell* is part of
          the answer. The loop is centre-major with the context position
          ascending, and the ragged expansion below reproduces exactly that;
          `np.add.at` then applies the additions in index order.

        The decay weights come from the same scalar `decay()` the loop uses, via
        a lookup table over the `window + 1` possible distances -- not from
        `np.exp`, which is free to differ in the last bit.
        """
        arrays = []
        for tokens in texts:
            sentence = np.asarray(tokens, dtype=np.int64)
            if self.delete_oov:
                # `vocab_size` is the <UNK> id; dropping it CLOSES the window
                # over that slot, exactly as the loop's list comprehension does
                sentence = sentence[sentence != self.vocab_size]
            arrays.append(sentence)

        lengths = np.array([len(a) for a in arrays], dtype=np.int64)
        if not len(lengths) or lengths.sum() == 0:
            return to_count_matrix({}, self.vocab_size)
        flat = np.concatenate(arrays)

        # per position: the bounds of the sentence it belongs to
        sentence_start = np.repeat(np.cumsum(lengths) - lengths, lengths)
        sentence_end = sentence_start + np.repeat(lengths, lengths)
        centre = np.arange(len(flat), dtype=np.int64)

        if self.dynamic_window_prob:
            # one draw per token, in token order -- exactly what the loop does
            radius = np.fromiter(
                (rng.randint(1, self.window) for _ in range(len(flat))),
                dtype=np.int64,
                count=len(flat),
            )
        else:
            radius = self.window

        lo = np.maximum(sentence_start, centre - radius)
        hi = np.minimum(sentence_end, centre + radius + 1)
        span = hi - lo  # still includes the centre itself
        n_events = int(span.sum())
        if n_events > MAX_VECTORIZED_EVENTS:
            return None

        # ragged expansion: for each centre, its context positions ascending.
        # This is the loop's `for j in range(start, end)`, all centres at once.
        offsets = np.arange(n_events, dtype=np.int64) - np.repeat(
            np.cumsum(span) - span, span
        )
        context = np.repeat(lo, span) + offsets
        centre = np.repeat(centre, span)
        keep = context != centre  # the loop's `if j != i`
        context = context[keep]
        centre = centre[keep]
        # peak memory is what bounds `MAX_VECTORIZED_EVENTS`, and these three are
        # each one event-length array that nothing below needs
        del offsets, keep, lo, hi, span, sentence_start, sentence_end

        if self.dynamic_window_deter:
            distance = np.abs(centre - context).astype(np.float64)
            weights = (self.window + 1 - distance) / self.window
        elif self.dynamic_window_decay is not None:
            table = np.array(
                [
                    decay(float(d), self.dynamic_window_decay)
                    for d in range(self.window + 1)
                ],
                dtype=np.float64,
            )
            weights = table[np.abs(centre - context)]
        else:
            weights = np.ones(len(context), dtype=np.float64)

        # one integer per (word, context) cell, so the accumulator is the size
        # of the *observed* pairs rather than (V+1)^2
        stride = self.vocab_size + 1
        keys = flat[centre] * stride + flat[context]
        unique, inverse = np.unique(keys, return_inverse=True)
        totals = np.zeros(len(unique), dtype=np.float64)
        np.add.at(totals, inverse.ravel(), weights)

        matrix = coo_matrix(
            (totals, (unique // stride, unique % stride)),
            shape=(stride, stride),
            dtype=np.float32,
        )
        return matrix.tocsr()


def iterate_tokens(
    tokens,
    window,
    dynamic_window_prob,
    dynamic_window_deter,
    dynamic_window_decay,
    delete_oov,
    subsampler_prob,
    subsampler_dirty,
    unkown_id,
    rng,
):
    """
    iterate over tokens in a sentence and counting pairs

    `subsampler_prob` is a *keep* probability per token id (see
    `count_pairs`). When it is set, two treatments of the dropped tokens are
    possible and `subsampler_dirty` selects between them:

      * clean (``subsampler_dirty`` False): a dropped token is blanked to
        `None` but keeps its position, so the window still spans its slot and
        does NOT reach past it. The inner loop skips the `None` slots.
      * dirty (``subsampler_dirty`` True): a dropped token is removed from the
        sequence before any window is built, so the window closes up and
        reaches one word further. This is the variant Levy, Goldberg & Dagan
        (2015, section 3.1) report, and it is what hyperwords' `corpus2pairs.py`
        does (`if random() > keep_prob: continue`, before pairs are emitted).

    Both treatments draw exactly one `rng.random()` per subsample-eligible
    token, in token order, so the per-token random stream has the same shape
    regardless of which one is chosen.
    """
    if delete_oov:
        tokens = [t for t in tokens if t != unkown_id]

    if subsampler_prob is not None:
        if subsampler_dirty:
            # dirty: drop the token outright, so the window later closes up
            tokens = [
                t
                for t in tokens
                if t not in subsampler_prob or rng.random() <= subsampler_prob[t]
            ]
        else:
            # clean: blank the token but keep its slot, so the window still
            # counts the slot and does not reach past it
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


VALID_MODES = frozenset({"deter", "prob", "dirty", "off", "decay"})


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

    `subsample_factor` is word2vec's `sample` parameter: a word is subsampled
    once its count exceeds `subsample_factor * total_tokens`, and kept with
    probability `sqrt(threshold / count)` above that. Keeping the word2vec
    convention (rather than, say, a fraction of the vocabulary) is deliberate,
    so a value can be carried over from a word2vec/hyperwords setup unchanged.

    `subsample` selects how frequent-word subsampling is applied. Levy,
    Goldberg & Dagan (2015, section 3.1) distinguish two ways of dropping
    tokens, which differ in whether a dropped token's slot still bounds the
    context window:

      * ``"prob"`` -- CLEAN subsampling. A dropped token is blanked but keeps
        its position, so the window still counts its slot and does NOT reach
        past it. Randomized per token, one draw per subsample-eligible token.
      * ``"dirty"`` -- DIRTY subsampling, the variant the paper REPORTS. A
        dropped token is removed from the sequence BEFORE any window is built,
        so the window closes up and reaches one word further, creating
        co-occurrences between words that were separated only by the dropped
        token. This matches hyperwords' `corpus2pairs.py`, which drops tokens
        (`if random() > keep_prob: continue`) before emitting any pairs.
        Randomized, using the same per-token draw pattern as ``"prob"``.
      * ``"deter"`` (default) -- a deterministic post-hoc approximation:
        the final count matrix is scaled by
        ``diag(sqrt(t/f)) M diag(sqrt(t/f))``. No tokens are dropped, so this
        is neither the clean nor the dirty variant, only their expectation.
      * ``"off"``/``None`` -- no subsampling.
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
    if subsample in ("prob", "dirty"):
        # `iterate_tokens` compares against this as a *keep* probability, so it
        # has to fall as the count rises. It used to be `1 - sqrt(...)`, the
        # word2vec *discard* probability, which kept frequent words and dropped
        # rare ones -- exactly backwards, and the opposite of "deter" below.
        # "prob" and "dirty" share this exact same keep map; they differ only in
        # how `iterate_tokens` treats the dropped tokens (see its docstring).
        subsampler_prob = subsample_keep_probabilities(corpus.counts, subsample_value)

    count_matrix = count_pairs_parallel(
        corpus.texts,
        make_count_closure(
            corpus,
            window=window,
            dynamic_window=dynamic_window,
            decay_rate=decay_rate,
            delete_oov=delete_oov,
            subsample=subsample,
            subsampler_prob=subsampler_prob,
            seed=seed,
        ),
    )

    # `count_pairs_parallel` returns None when there are no text chunks at all,
    # i.e. an empty corpus. Raise a clear error here -- where the corpus is in
    # scope -- rather than returning a degenerate (1x1) zero matrix that only
    # detonates two layers downstream in PMI/SVD with a far less obvious message.
    if count_matrix is None:
        raise ValueError(
            "cannot count pairs on an empty corpus: no text chunks were "
            "produced. Did `Corpus.from_texts([])` receive any documents?"
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


# The pair-count defaults, for `record` to write alongside a result in the db.
# Derived from the signature rather than restated, so the two cannot drift: a
# hand-written copy here used to omit `seed` and `min_count`, so a run using
# their defaults recorded no value for them and `results(query={"min_count": 0})`
# found nothing. `corpus` is dropped -- it has no default and is not a parameter.
default_pair_args = MappingProxyType(
    {
        name: param.default
        for name, param in inspect.signature(count_pairs).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
)
