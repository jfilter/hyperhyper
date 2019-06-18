import logging
import os
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim.utils import SaveLoad
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import map_chunks

logger = logging.getLogger(__name__)

num_cpu = os.cpu_count()

from collections import defaultdict


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


class Corpus(SaveLoad):
    def __init__(self, vocab, texts):
        self.vocab = vocab
        logger.info("converting texts to indices")

        def to_idx(texts):
            return [self.vocab.doc2idx(t) for t in texts]

        self.texts = map_chunks(texts, to_idx)

        self.size = len(self.texts)
        logger.info("counting appearences of tokens")

        frequency = defaultdict(int)
        for text in tqdm(self.texts):
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
        preproc_func=preprocess_string,
    ):
        if preproc_func is not None:
            logger.info("preprocessing texts")
            texts = Parallel(n_jobs=num_cpu)(
                delayed(preproc_func)(t) for t in tqdm(texts)
            )
            logger.info("done preprocessing")
        if vocab is None:
            vocab = Vocab(
                texts,
                no_below=no_below,
                keep_n=keep_n,
                keep_tokens=keep_tokens,
                no_above=no_above,
            )
        corpus = Corpus(vocab, texts)
        return corpus
