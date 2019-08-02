import logging
import os
import random
from array import array
from collections import Counter, defaultdict
from concurrent import futures
from math import ceil, fabs, sqrt, e

import numpy as np
from gensim.utils import SaveLoad
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from tqdm import tqdm

from .utils import chunks, read_pickle

logger = logging.getLogger(__name__)


def decay(x, rate):
    x -= 1  # so the value is 1 when the distance is 1
    return e ** -(rate * x)


class PairCounts(SaveLoad):
    counter = defaultdict(int)

    def __getitem__(self, key):
        return self.counter[key]

    def __setitem__(self, key, value):
        self.counter[key] = value

    def items(self):
        return self.counter.items()

    def __str__(self):
        return str(self.counter)


def to_count_matrix(pair_counts, vocab_size):
    """
    Transforms the counts into a sparse matrix (CSR).
    """
    cols = []
    rows = []
    data = []
    for k, v in pair_counts.items():
        rows.append(k[0])
        cols.append(k[1])
        data.append(v)
    # why is float32 so important?
    # +1 for UNK
    count_matrix = coo_matrix(
        (data, (rows, cols)), shape=(vocab_size + 1, vocab_size + 1), dtype=np.float32
    )
    # logger.info(f"num non-zeros: {count_matrix.count_nonzero()}")
    return count_matrix


def count_pairs_parallel(texts_paths, count_pairs_closure, low_memory):
    # Ensure that memory is freed when a job completes.
    res = None
    with futures.ProcessPoolExecutor() as executor:
        # A dictionary which will contain a list the future info in the key, and the filename in the value
        jobs = {}
        files_left = len(texts_paths)
        files_iter = iter(texts_paths)

        if low_memory:
            MAX_JOBS_IN_QUEUE = os.cpu_count()
        else:
            MAX_JOBS_IN_QUEUE = os.cpu_count() * 2  # heuristic ;)

        with tqdm(total=len(texts_paths), desc="generating pairs") as pbar:
            while files_left:
                for this_file in files_iter:
                    job = executor.submit(count_pairs_closure, this_file)
                    jobs[job] = this_file
                    if len(jobs) > MAX_JOBS_IN_QUEUE:
                        break  # limit the job submission for now job

                # Get the completed jobs whenever they are done
                for job in futures.as_completed(jobs):
                    files_left -= 1
                    pbar.update(1)
                    m = job.result()
                    if res is None:
                        res = m
                    else:
                        res += m

                    del jobs[job]
    return res


# The 'pythonic' of creating closures (irony).
# Used for pickeling when doing multiprocessing.
class CountPairsClosure(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, text_path):
        texts = read_pickle(text_path)
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
            ):
                counter[pair[0], pair[1]] += pair[2]
        return to_count_matrix(counter, self.vocab_size)


# iterate over tokens in a sentence
def iterate_tokens(
    tokens,
    window,
    dynamic_window_prob,
    dynamic_window_deter,
    dynamic_window_decay,
    delete_oov,
    subsampler_prob,
    unkown_id,
):
    if delete_oov:
        tokens = [t for t in tokens if t != unkown_id]

    if not subsampler_prob is None:
        tokens = [
            t
            if t not in subsampler_prob or random.random() <= subsampler_prob[t]
            else None
            for t in tokens
        ]

    len_tokens = len(tokens)
    res = []
    for i, tok in enumerate(tokens):
        if tok is not None:
            if dynamic_window_prob:
                offset = random.randint(1, window)
            else:
                offset = window
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
                    if not dynamic_window_decay is None:
                        distance = fabs(i - j)
                        count = decay(distance, dynamic_window_decay)
                    res.append((tok, tokens[j], count))
    return res


def count_pairs(
    corpus,
    window=2,
    dynamic_window="deter",
    decay_rate=0.25,
    delete_oov=True,
    subsample="deter",
    subsample_factor=1e-5,
    seed=1312,
    low_memory=False,
):
    for x in [dynamic_window, subsample]:
        if not x is None and not x == False:
            assert x in ("deter", "prob", "off", "decay")

    random.seed(seed)

    subsampler_prob = None
    if subsample == "prob":
        subsampler_prob = subsample_factor * corpus.size
        subsampler_prob = {
            word: 1 - sqrt(subsampler_prob / count)
            for word, count in corpus.counts.items()
            if count > subsampler_prob
        }

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
        ),
        low_memory=low_memory,
    )

    # down sample in a deterministic way
    if subsample == "deter":
        # construct array with appropriate factor
        subsample_value = subsample_factor * corpus.size
        subsampler = np.ones(corpus.vocab.size + 1, dtype=np.float32)

        logger.info('creating subsampler matrix')
        num_sub = 0
        for word, count in corpus.counts.items():
            if count > subsample_value:
                subsampler[word] = sqrt(subsample_value / count)
                num_sub += 1
        if low_memory:
            # this requires less memory but more time
            indices = zip(*count_matrix.nonzero())
            total = count_matrix.count_nonzero()
            count_matrix = count_matrix.todok()
            for i, j in tqdm(indices, desc="subsample deterministic (low memory)", total=total):
                count_matrix[(i, j)] *= subsampler[i] * subsampler[j]
        else:
            print(f"subsampling applied to {num_sub / corpus.vocab.size} of the tokens")
            # to 2d matrix
            subsampler = subsampler.reshape((-1, 1)).dot(subsampler.reshape(1, -1))
            logger.info('multiply elementwise: start')
            # elementwise muplication of 2 matrices
            count_matrix = count_matrix.multiply(subsampler)
            # had to convert to dense for a minute
            logger.info('multiply elementwise: done')

    return count_matrix.tocsr()
