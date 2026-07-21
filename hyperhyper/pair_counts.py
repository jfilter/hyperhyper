"""
construct a co-occurrence matrix by counting word pairs (co-locations of words)
"""

import logging
import random
from collections import defaultdict
from concurrent import futures
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


def count_pairs_parallel(texts_paths, count_pairs_closure):
    """
    count pairs in parallel by loading and processing files to keep memory
    consumption low
    """
    # Ensure that memory is freed when a job completes.
    res = None
    with futures.ProcessPoolExecutor(_default_workers()) as executor:
        # A dictionary which will contain a list the future info in the key, and the filename in the value
        jobs = {}
        files_left = len(texts_paths)
        files_iter = iter(texts_paths)

        MAX_JOBS_IN_QUEUE = _default_workers() * 2  # heuristic ;)

        with tqdm(total=len(texts_paths), desc="generating pairs") as pbar:
            while files_left:
                for this_file in files_iter:
                    job = executor.submit(count_pairs_closure, this_file)
                    jobs[job] = this_file
                    if len(jobs) > MAX_JOBS_IN_QUEUE:
                        break  # limit the job submission for now job

                # Get the completed jobs whenever they are done.
                # Buffer the partial matrices instead of adding them straight
                # into `res`: `as_completed` yields in worker-completion order,
                # and float32 addition is not associative, so accumulating as
                # results arrive makes the cached `.npz` differ bit for bit
                # between runs. `corpus.py` was hardened against exactly this
                # for the vocab merge; this is the same fix for the pair counts.
                # The buffer only ever holds one submission round (bounded by
                # MAX_JOBS_IN_QUEUE), so memory stays where it was.
                done = {}
                for job in futures.as_completed(jobs):
                    files_left -= 1
                    pbar.update(1)
                    done[jobs[job]] = job.result()
                    del jobs[job]

                # add in a fixed order, mirroring `Corpus.from_text_files`
                for this_file in sorted(done):
                    m = done.pop(this_file)
                    if res is None:
                        res = m
                    else:
                        res += m
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
