import logging
import os
from collections import defaultdict
from pathlib import Path
from array import array

from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_non_alphanum,
    strip_numeric,
)
from gensim.utils import SaveLoad
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import map_pool

logger = logging.getLogger(__name__)

num_cpu = os.cpu_count()


def default_preprocess_string(text):
    CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_numeric]
    return preprocess_string(text, CUSTOM_FILTERS)


class Vocab(Dictionary):
    def __init__(self, texts, **kwargs):
        logger.info("creating vocab")
        super().__init__(texts)
        logger.info("filtering extremes")
        self.filter_extremes(**kwargs)
        logger.info("done with vocab")

    @property
    def size(self):
        return len(self.token2id)

    @property
    def tokens(self):
        # return tokens as array (in order of id)
        return [tup[0] for tup in sorted(self.token2id.items(), key=lambda x: x[1])]


# def create(x):
#     def to_idx(texts):
#         if x.vocab_size <= 65535:
#             size = "H"
#         else:
#             size = "L"
#         return [
#             array(size, x.vocab.doc2idx(t, x.vocab_size))
#             for t in texts

#             # for t in tqdm(texts, desc="converting texts to indices")
#         ]

#     return to_idx


# a closure that is pickable
# <UNK> is the last ID (thus vocab_size)
# https://docs.python.org/3/library/array.html
class TransformToIndicesClosure(object):
    def __init__(self, c):
        self.vocab_size = c.vocab.size
        self.d = c.vocab.doc2idx
        if self.vocab_size <= 65535:
            self.size = "H"
        else:
            self.size = "L"

    def __call__(self, texts):
        return array(self.size, self.d(texts, self.vocab_size))


class Corpus(SaveLoad):
    def __init__(self, vocab, texts, preproc_fun):
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.preproc_fun = preproc_fun

        # parallel is slower (due to large vocab?)
        # self.texts = map(texts, TransformToIndicesClosure(self))
        toIndices = TransformToIndicesClosure(self)
        self.texts = [toIndices(t) for t in tqdm(texts, desc="transform to indices")]

        self.size = len(self.texts)
        frequency = defaultdict(int)
        for text in tqdm(self.texts, desc="counting frequency of tokens"):
            for token in text:
                frequency[token] += 1

        self.counts = frequency

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
        return Corpus.from_texts(lines, **kwargs)

    # TODO: tokenzation?
    @staticmethod
    def from_texts(
        texts,
        no_below=0,
        no_above=1,
        keep_n=50000,
        keep_tokens=None,
        vocab=None,
        preproc_func=default_preprocess_string,
    ):
        assert preproc_func is not None
        texts = map_pool(texts, preproc_func, desc="preprocessing texts")
        # texts = Parallel(n_jobs=num_cpu)(
        #     delayed(preproc_func)(t) for t in tqdm(texts, desc="preprocessing texts")
        # )
        if vocab is None:
            vocab = Vocab(
                texts,
                no_below=no_below,
                keep_n=keep_n,
                keep_tokens=keep_tokens,
                no_above=no_above,
            )
        corpus = Corpus(vocab, texts, preproc_func)
        return corpus
