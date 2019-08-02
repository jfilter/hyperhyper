import logging
import os
import pickle
import random
from array import array
from collections import defaultdict
from concurrent import futures
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.utils import SaveLoad
from tqdm import tqdm

from .preprocessing import simple_tokenizer, texts_to_sents
from .utils import chunks, dsum, map_pool, read_pickle, to_pickle

random.seed(1312)
logger = logging.getLogger(__name__)


class Vocab(Dictionary):
    def __init__(self, texts=None, **kwargs):
        super().__init__(texts)
        if not texts is None:
            self.filter(**kwargs)

    def filter(self, no_below=0, no_above=1, keep_n=50000, keep_tokens=None):
        self.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens
        )

    @property
    def size(self):
        return len(self.token2id)

    @property
    def tokens(self):
        # return tokens as array (in order of id)
        return [tup[0] for tup in sorted(self.token2id.items(), key=lambda x: x[1])]


class TransformToIndicesClosure(object):
    # a closure that is pickable
    # <UNK> is the last ID (thus vocab_size)
    # https://docs.python.org/3/library/array.html

    def __init__(self, c):
        self.vocab_size = c.vocab.size
        self.d = c.vocab.doc2idx
        if self.vocab_size <= 65535:
            self.size = "H"
        else:
            self.size = "L"

    def __call__(self, texts):
        return array(self.size, self.d(texts, self.vocab_size))


def count_tokens(texts):
    counts = defaultdict(int)
    for text in texts:
        for token in text:
            counts[token] += 1
    return counts


def _texts_to_ids(args):
    f, to_indices, recount = args[0], args[1], args[2]
    texts = read_pickle(f)
    transformed = [to_indices(t) for t in texts]
    to_pickle(transformed, f)
    if recount:
        counts = count_tokens(transformed)
    else:
        counts = {}
    return len(transformed), counts



def texts_to_ids(input_text_fns, to_indices, recount):
    total_len = 0
    all_counts = []
    with futures.ProcessPoolExecutor() as executor:
        # A dictionary which will contain a list the future info in the key, and the filename in the value
        jobs = {}
        files_left = len(input_text_fns)
        files_iter = iter(input_text_fns)
        MAX_JOBS_IN_QUEUE = os.cpu_count() * 2

        with tqdm(total=len(input_text_fns), desc="generating pairs") as pbar:
            while files_left:
                for this_file in files_iter:
                    job = executor.submit(
                        _texts_to_ids, [this_file, to_indices, recount]
                    )
                    jobs[job] = this_file
                    if len(jobs) > MAX_JOBS_IN_QUEUE:
                        break  # limit the job submission for now job

                # Get the completed jobs whenever they are done
                for job in futures.as_completed(jobs):
                    files_left -= 1
                    pbar.update(1)
                    num_sents, counts = job.result()
                    all_counts.append(counts)
                    total_len += num_sents
                    del jobs[job]

    return total_len, dsum(*all_counts)


def _build_vocab_from_file(args):
    f, preproc_func, view_fraction = args[0], args[1], args[2]

    texts = f.read_text().split("\n")
    texts = preproc_func(texts)

    # temporary save processed files to continue working later
    to_pickle(texts, f.with_suffix(".pkl"))

    # skip at random
    if 0.999 > view_fraction < random.random():
        return Vocab()
    return Vocab(texts)


class Corpus(SaveLoad):
    def __init__(
        self, vocab, preproc_fun, texts=None, input_text_fns=None, recount=False
    ):
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.preproc_fun = preproc_fun

        if texts is None:
            to_indices = TransformToIndicesClosure(self)
            self.size, counts = texts_to_ids(input_text_fns, to_indices, recount)
            # count properly if the vocab was constructed on a fraction of the data
            if recount:
                self.vocab.dfs = counts
            self.input_text_fns = input_text_fns
            self.texts = None
        else:
            to_indices = TransformToIndicesClosure(self)
            transformed = [
                to_indices(t) for t in tqdm(texts, desc="transform to indices")
            ]
            self.texts = transformed
            self.size = len(transformed)

    @property
    def counts(self):
        return self.vocab.dfs

    def texts_to_file(self, dir, text_chunk_size):
        if self.texts is None:
            # just use the ones we created
            self.texts = self.input_text_fns
            # for i, f in enumerate(self.input_text_fns):
            #     pass
        else:
            # can't do in init because we don't have a file location yet
            fns = []
            for i, c in enumerate(chunks(self.texts, text_chunk_size)):
                fn = Path(f"{dir}/texts_{i}.pkl").resolve()
                to_pickle(c, fn)
                fns.append(fn)
            self.texts = fns

    @staticmethod
    def from_file(input, limit=None, **kwargs):
        """
        Reads a file where each line represent a sentence.
        """

        logger.info("reading file")
        text = Path(input).read_text()
        lines = text.splitlines()
        if limit is not None:
            lines = lines[:limit]
        logger.info("done reading file")
        return Corpus.from_sents(lines, **kwargs)

    @staticmethod
    def from_sents(
        texts, vocab=None, preproc_func=simple_tokenizer, preproc_single=False, **kwargs
    ):
        assert preproc_func is not None
        if preproc_single:
            texts = preproc_func(texts)
        else:
            texts = map_pool(texts, preproc_func, desc="preprocessing texts")

        if vocab is None:
            vocab = Vocab(texts, **kwargs)
        corpus = Corpus(vocab, preproc_func, texts=texts)
        return corpus

    @staticmethod
    def from_texts(texts, preproc_func=texts_to_sents, preproc_single=True, **kwargs):
        return Corpus.from_sents(
            texts, preproc_func=preproc_func, preproc_single=preproc_single, **kwargs
        )

    @staticmethod
    def from_text_files(
        base_dir,
        preproc_func=texts_to_sents,
        preproc_single=True,
        view_fraction=1,
        recount=True,
        **kwargs,
    ):
        voc = Vocab()
        input_text_fns = list(Path(base_dir).glob("*.txt"))
        proc_fns = [f.with_suffix(".pkl") for f in input_text_fns]

        with futures.ProcessPoolExecutor() as executor:
            jobs = {}
            files_left = len(input_text_fns)
            files_iter = iter(input_text_fns)
            MAX_JOBS_IN_QUEUE = os.cpu_count() * 2

            with tqdm(total=len(input_text_fns), desc="build up vocab") as pbar:
                while files_left:
                    for this_file in files_iter:
                        job = executor.submit(
                            _build_vocab_from_file,
                            [this_file, preproc_func, view_fraction],
                        )
                        jobs[job] = this_file
                        if len(jobs) > MAX_JOBS_IN_QUEUE:
                            break

                    for job in futures.as_completed(jobs):
                        files_left -= 1
                        pbar.update(1)
                        # update document frequencies
                        voc.merge_with(job.result())
                        del jobs[job]

        # only consider most frequent terms
        voc.filter(**kwargs)

        if view_fraction > 0.999:
            return Corpus(voc, preproc_func, input_text_fns=proc_fns, recount=False)

        if recount:
            return Corpus(voc, preproc_func, input_text_fns=proc_fns, recount=True)

        # extrapolate
        factor = 1 / view_fraction
        for key, value in voc.dfs.items():
            voc.dfs[key] = value * factor
        return Corpus(voc, preproc_func, input_text_fns=proc_fns, recount=False)
