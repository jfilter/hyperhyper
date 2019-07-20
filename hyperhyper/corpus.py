import logging
import os
import pickle
from array import array
from collections import defaultdict
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.utils import SaveLoad
from joblib import Parallel, delayed
from tqdm import tqdm

from .preprocessing import simple_tokenizer, texts_to_sents
from .utils import chunks, map_pool, to_pickle

logger = logging.getLogger(__name__)

num_cpu = os.cpu_count()


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
        texts = [toIndices(t) for t in tqdm(texts, desc="transform to indices")]

        self.size = len(texts)
        self.counts = defaultdict(int)
        for text in tqdm(texts, desc="counting frequency of tokens"):
            for token in text:
                self.counts[token] += 1
        self.texts = texts

    # can't do in init because we don't have a file location yet
    def texts_to_file(self, dir, text_chunk_size):
        fns = []
        for i, c in enumerate(chunks(self.texts, text_chunk_size)):
            fn = Path(f"{dir}/texts_1{i}.pkl").resolve()
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
        return Corpus.from_texts(lines, **kwargs)

    @staticmethod
    def from_sents(
        texts,
        no_below=0,
        no_above=1,
        keep_n=50000,
        keep_tokens=None,
        vocab=None,
        preproc_func=simple_tokenizer,
        preproc_single=False,
    ):
        assert preproc_func is not None
        if preproc_single:
            texts = preproc_func(texts)
        else:
            texts = map_pool(texts, preproc_func, desc="preprocessing texts")

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

    @staticmethod
    def from_texts(texts, preproc_func=texts_to_sents, preproc_single=True, **kwargs):
        return Corpus.from_sents(
            texts, preproc_func=preproc_func, preproc_single=preproc_single, **kwargs
        )
