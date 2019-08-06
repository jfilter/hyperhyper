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

from .preprocessing import texts_to_sents, tokenize_texts, tokenize_texts_parallel
from .utils import chunks, dsum, read_pickle, to_pickle

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
    # count again, gensim's dictionary only provides document frequencies
    counts = defaultdict(int)
    for text in texts:
        for token in text:
            counts[token] += 1
    return counts


def _texts_to_ids(args):
    f, to_indices = args[0], args[1]
    texts = read_pickle(f)
    transformed = [to_indices(t) for t in texts]
    to_pickle(transformed, f)
    counts = count_tokens(transformed)
    return len(transformed), counts


def texts_to_ids(input_text_fns, to_indices):
    total_len = 0
    all_counts = []
    with futures.ProcessPoolExecutor() as executor:
        # A dictionary which will contain a list the future info in the key, and the filename in the value
        jobs = {}
        files_left = len(input_text_fns)
        files_iter = iter(input_text_fns)
        MAX_JOBS_IN_QUEUE = os.cpu_count() * 2

        with tqdm(total=len(input_text_fns), desc="texts to ids") as pbar:
            while files_left:
                for this_file in files_iter:
                    job = executor.submit(_texts_to_ids, [this_file, to_indices])
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
    def __init__(self, vocab, preproc_fun, texts=None, input_text_fns=None, lang="en"):
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.lang = lang
        self.preproc_fun = preproc_fun

        if texts is None:
            to_indices = TransformToIndicesClosure(self)
            self.size, self.counts = texts_to_ids(input_text_fns, to_indices)
            self.input_text_fns = input_text_fns
            self.texts = None
        else:
            to_indices = TransformToIndicesClosure(self)
            transformed = [
                to_indices(t) for t in tqdm(texts, desc="transform to indices")
            ]
            self.texts = transformed
            self.counts = count_tokens(transformed)
            self.size = len(transformed)

    def texts_to_file(self, dir, text_chunk_size):
        if self.texts is None:
            # just use the ones we created
            self.texts = self.input_text_fns
            fns = []
            Path(dir).mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(self.input_text_fns):
                # TODO: make use of chunk size?
                new_path = Path(f"{dir}/texts_{i}.pkl").resolve()
                f.rename(new_path)
                fns.append(new_path)
            self.texts = fns
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
        texts, vocab=None, preproc_func=tokenize_texts_parallel, lang="en", **kwargs
    ):
        texts = preproc_func(texts)
        if vocab is None:
            vocab = Vocab(texts, **kwargs)
        corpus = Corpus(vocab, preproc_func, texts=texts, lang=lang)
        return corpus

    @staticmethod
    def from_texts(texts, preproc_func=texts_to_sents, **kwargs):
        return Corpus.from_sents(texts, preproc_func=preproc_func, **kwargs)

    @staticmethod
    def from_text_files(
        base_dir, preproc_func=texts_to_sents, view_fraction=1, lang="en", **kwargs
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
            return Corpus(voc, preproc_func, input_text_fns=proc_fns, lang=lang)

        return Corpus(voc, preproc_func, input_text_fns=proc_fns, lang=lang)

