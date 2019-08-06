import logging
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

import dataset

from . import evaluation, pair_counts, pmi, svd
from .corpus import Corpus
from .experiment import record
from .utils import delete_folder, load_arrays, load_matrix, save_arrays, save_matrix

logger = logging.getLogger(__name__)


class Bunch:
    def __init__(
        self, path, corpus=None, force_overwrite=False, text_chunk_size=100000
    ):
        self.db = None
        self.path = Path(path)

        if force_overwrite:
            delete_folder(self.path)

        if not corpus is None and not force_overwrite:
            if Path(self.path / "corpus.pkl").is_file():
                raise ValueError(
                    "There is already another corpus file saved. Set `force_overwrite` to True if you want to override it."
                )

        if corpus is None:
            self.corpus = Corpus.load(str(self.path / "corpus.pkl"))
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self.corpus = corpus
            self.corpus.texts_to_file(self.path / "texts", text_chunk_size)
            self.corpus.save(str(self.path / "corpus.pkl"))

    def get_db(self):
        if self.db is None:
            # connecting to a SQLite database
            self.db = dataset.connect(f"sqlite:///{self.path}/results.db")
        return self.db

    def dict_to_path(self, folder, dict):
        # cast integer floats to ints
        for k, v in dict.items():
            if type(v) is float:
                if v.is_integer():
                    dict[k] = int(v)

        filenames = [f"{k}_{v}".lower() for k, v in dict.items()]
        filename = "_".join(sorted(filenames))
        if len(filename) == 0:
            filename = "default"

        filename += ".npz"
        full_path = self.path / folder / filename
        return full_path

    def pair_counts(self, **kwargs):
        pair_path = self.dict_to_path("pair_counts", kwargs)
        if pair_path.is_file():
            try:
                logger.info("retrieved already saved pair count")
                return load_matrix(pair_path)
            except Exception as e:
                logger.info(f"creating pair counts, error while loading files: {e}")

        print("create new pair counts")
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        count_matrix = pair_counts.count_pairs(self.corpus, **kwargs)
        save_matrix(pair_path, count_matrix)
        return count_matrix

    def pmi_matrix(self, cds=0.75, pair_args={}, **kwargs):
        pmi_path = self.dict_to_path("pmi", {"cds": cds, **pair_args})
        if pmi_path.is_file():
            try:
                logger.info("retrieved already saved pmi")
                return load_matrix(pmi_path)
            except Exception as e:
                logger.info(f"creating new pmi, error while loading files: {e}")

        print("create new pmi")
        counts = self.pair_counts(**pair_args, **kwargs)

        start = timer()
        pmi_matrix = pmi.calc_pmi(counts, cds)

        end = timer()
        logger.info("pmi took " + str(round(end - start, 2)) + " seconds")

        pmi_path.parent.mkdir(parents=True, exist_ok=True)
        save_matrix(pmi_path, pmi_matrix)
        logger.info("matrix saved")

        return pmi_matrix

    @record
    def pmi(
        self,
        neg=1,
        cds=0.75,
        pair_args={},
        keyed_vectors=False,
        evaluate=True,
        **kwargs,
    ):
        m = self.pmi_matrix(cds, pair_args, **kwargs)
        embd = pmi.PPMIEmbedding(m, neg=neg)
        if evaluate:
            eval_results = self.eval_sim(embd)
        if keyed_vectors:
            return self.to_keyed_vectors(embd.m.todense(), m.shape[0])
        if evaluate:
            return embd, eval_results
        return embd

    def svd_matrix(
        self, impl, impl_args={}, dim=500, neg=1, cds=0.75, pair_args={}, **kwargs
    ):
        assert impl in ["scipy", "gensim", "scikit", "sparsesvd"]

        svd_path = self.dict_to_path(
            "svd",
            {
                "impl": impl,
                **impl_args,
                "neg": neg,
                "cds": cds,
                "dim": dim,
                **pair_args,
            },
        )
        logger.debug(f"looking up the file: {svd_path}")
        if svd_path.is_file():
            try:
                logger.info("retrieved already saved svd")
                return load_arrays(svd_path)
            except Exception as e:
                logger.info(f"creating new svd, error while loading files: {e}")

        print("creating new svd")
        m = self.pmi_matrix(cds, pair_args, **kwargs)
        m = pmi.PPMIEmbedding(m, neg=neg, normalize=False)

        start = timer()
        ut, s = svd.calc_svd(m, dim, impl, impl_args)
        end = timer()
        logger.info("svd took " + str(round((end - start) / 60, 2)) + " minutes")

        svd_path.parent.mkdir(parents=True, exist_ok=True)
        save_arrays(svd_path, ut, s)
        logger.info("svd arrays saved")

        return ut, s

    @record
    def svd(
        self,
        dim=500,
        eig=0,
        neg=1,
        cds=0.75,
        impl="scipy",
        impl_args={},
        pair_args={},
        keyed_vector=False,
        evaluate=True,
        **kwargs,
    ):
        ut, s = self.svd_matrix(
            impl=impl,
            impl_args=impl_args,
            dim=dim,
            neg=neg,
            cds=cds,
            pair_args=pair_args,
            **kwargs,
        )
        embedding = svd.SVDEmbedding(ut, s, eig=eig)

        if evaluate:
            eval_results = self.eval_sim(embedding)
        if keyed_vector:
            embedding = self.to_keyed_vectors(embedding.m, dim)
        if evaluate:
            return embedding, eval_results
        return embedding

    def to_keyed_vectors(self, embd_matrix, dim):
        # https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
        vectors = WordEmbeddingsKeyedVectors(vector_size=dim)

        # delete last row (for <UNK> token)
        embd_matrix = np.delete(embd_matrix, (-1), axis=0)

        vectors.add(self.corpus.vocab.tokens, embd_matrix)
        return vectors

    def eval_sim(self, embd, **kwargs):
        return evaluation.eval_similarity(
            embd,
            self.corpus.vocab.token2id,
            self.corpus.preproc_fun,
            lang=self.corpus.lang,
            **kwargs,
        )
