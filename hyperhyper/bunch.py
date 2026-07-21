"""
The heart of the package. This combines all the function and also exposes
the funtionality to the user. The `bunch` is the location where all the
resulting files are stored.
"""

import hashlib
import inspect
import json
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from timeit import default_timer as timer

import dataset
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from . import evaluation, pair_counts, pmi, svd
from .corpus import Corpus
from .experiment import record, results_from_db
from .utils import (
    _default_workers,
    delete_folder,
    load_arrays,
    load_matrix,
    save_arrays,
    save_matrix,
)

logger = logging.getLogger(__name__)

VALID_IMPLS = frozenset({"scipy", "gensim", "scikit"})

# Bumped whenever the meaning of a cache file changes -- a different default in
# `count_pairs`, a different matrix layout, a different key scheme. Old entries
# then simply stop being found instead of being served under a name whose
# meaning has silently moved.
CACHE_FORMAT = "v2"

# what may appear verbatim in the human-readable part of a cache file name
_UNSAFE_IN_NAME = re.compile(r"[^0-9A-Za-z.=+-]+")

# keep file names comfortably below any file system limit
_READABLE_PREFIX_MAX = 60

# How many text chunks to aim for per worker. The chunk files are the *only*
# unit of parallelism `count_pairs` has -- one task per chunk -- so the chunk
# count, not the chunk size, is what decides whether the machine is used. The
# old fixed 100k default produced 3 chunks for a 250k-sentence corpus, which on
# a 10-core machine is a hard ceiling of 3x however many cores are present, and
# measured out at 1.8x because the three chunks did not take equal time.
#
# Swept on 250k- and 500k-sentence corpora, 1 to 8 chunks per worker: the wall
# clock is flat to within noise from 1 to 6 (5.58-5.70s at 250k) and only starts
# to slip at 8 (6.12s), where the per-chunk overhead -- a pickled sparse matrix
# back to the parent, plus one more step in a merge that has to stay a strict
# left fold -- begins to show. So this is not a peak, it is the middle of a
# plateau, chosen for the headroom rather than for a measured advantage: chunks
# do not cost the same (the three chunks of the 250k corpus took 7.0s, 10.9s and
# 3.8s, a 2.9x spread), and at one chunk per worker that spread lands directly
# on the wall clock with nothing left to rebalance with.
CHUNKS_PER_WORKER = 4

# Floor: below this, a chunk stops being worth a task at all -- the round trip
# through the pool costs more than the counting.
MIN_TEXT_CHUNK_SIZE = 500

# Ceiling: one chunk is loaded into a worker whole, so this is what bounds a
# worker's peak memory. It is the old fixed default, kept in that role.
MAX_TEXT_CHUNK_SIZE = 100_000


def _auto_text_chunk_size(n_texts, workers=None):
    """
    Pick a chunk size that gives the pool a sensible number of chunks to spread.
    """
    workers = _default_workers() if workers is None else workers
    target_chunks = max(1, workers * CHUNKS_PER_WORKER)
    size = math.ceil(n_texts / target_chunks) if n_texts > 0 else MIN_TEXT_CHUNK_SIZE
    return max(MIN_TEXT_CHUNK_SIZE, min(size, MAX_TEXT_CHUNK_SIZE))


def _canonical(value):
    """
    Normalise a parameter value so that equivalent values serialise identically.

    Integral floats become ints (`dim=500.0` and `dim=500` are the same run),
    and the same is done inside nested containers.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    return value


def _readable_prefix(params):
    """
    A short, lossy, human-readable hint so a cache directory can be eyeballed.

    Purely decorative: only the digest that follows it identifies the entry.
    """
    parts = [
        _UNSAFE_IN_NAME.sub("-", f"{k}={v}")
        for k, v in sorted(params.items())
        if not isinstance(v, Mapping | list | tuple)
    ]
    return "_".join(parts)[:_READABLE_PREFIX_MAX].strip("_")


class Bunch:
    def __init__(self, path, corpus=None, force_overwrite=False, text_chunk_size=None):
        """
        `text_chunk_size` is how many sentences go into one on-disk text chunk.
        It still means exactly what it always did and an explicit value is still
        honoured verbatim; only the default has changed, from a fixed 100000 to
        `None`, which sizes the chunks from the corpus and the core count (see
        `_auto_text_chunk_size`). The old default was a parallelism ceiling in
        disguise: it is a *size*, and what `count_pairs` needs is a *count*.
        """
        self.db = None
        self.path = Path(path)

        # Deleting has to be guarded by "is there a replacement?". Wiping first
        # and only then discovering that there is no corpus to put back loses
        # every cached matrix *and* the results database of an existing bunch.
        if corpus is None:
            if force_overwrite:
                raise ValueError(
                    "`force_overwrite=True` needs a `corpus` to replace the old one. "
                    "Reopening an existing bunch without a corpus would delete it."
                )
            self.corpus = Corpus.load(str(self.path / "corpus.pkl"))
            return

        if not force_overwrite and Path(self.path / "corpus.pkl").is_file():
            raise ValueError(
                "There is already another corpus file saved. Set `force_overwrite` to True if you want to override it."
            )

        if force_overwrite and self.path.exists():
            delete_folder(self.path)

        if text_chunk_size is None:
            text_chunk_size = _auto_text_chunk_size(corpus.size)
        elif text_chunk_size < 1:
            raise ValueError(
                f"text_chunk_size must be a positive number of sentences or None "
                f"to size it automatically, got {text_chunk_size!r}"
            )
        logger.info("writing text chunks of %s sentences", text_chunk_size)

        self.path.mkdir(parents=True, exist_ok=True)
        self.corpus = corpus
        self.corpus.texts_to_file(self.path / "texts", text_chunk_size)
        self.corpus.save(str(self.path / "corpus.pkl"))

    def get_db(self):
        """
        Connecting to a SQLite database.
        """
        if self.db is None:
            self.db = dataset.connect(
                f"sqlite:///{self.path}/results.db",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )
        return self.db

    def close(self):
        """
        Dispose of the SQLite connection pool.

        Without this a long parameter sweep accumulates one pool per `Bunch`
        and keeps the database file open for the lifetime of the process.
        """
        db, self.db = self.db, None
        if db is not None:
            db.executable.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def dict_to_path(self, folder, params):
        """
        Return a file path for an embedding based on parameters.

        The name is a digest of a canonical serialisation of `params`, so two
        different parameter sets can never land in the same file. Nested dicts
        keep their own namespace, which the old flattened `k_v` scheme did not:
        `impl_args` and `pair_args` sharing a key name mapped to one slot.
        """
        canonical = _canonical(dict(params))
        blob = json.dumps(canonical, sort_keys=True, default=repr)
        digest = hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()

        prefix = _readable_prefix(canonical)
        stem = (
            f"{prefix}_{CACHE_FORMAT}-{digest}"
            if prefix
            else f"{CACHE_FORMAT}-{digest}"
        )
        return self.path / folder / f"{stem}.npz"

    def _effective_pair_args(self, pair_args=None, **kwargs):
        """
        Resolve the *full* argument set `count_pairs` will actually run with.

        Keying a cache on what the caller happened to spell leaves every
        defaulted parameter -- and everything routed through `**kwargs` -- out
        of the key, so a matrix computed with one window is served for another.
        Binding against the real signature makes `pair_counts()` and
        `pair_counts(window=2)` one entry, and `window=2` and `window=10` two.
        """
        pair_args = {} if pair_args is None else pair_args
        # `bind` raises on unknown or duplicated names, exactly as the real call
        # would, so a typo cannot silently create a second cache entry.
        bound = inspect.signature(pair_counts.count_pairs).bind(
            self.corpus, **pair_args, **kwargs
        )
        bound.apply_defaults()
        effective = dict(bound.arguments)
        effective.pop("corpus", None)
        return effective

    def pair_counts(self, **kwargs):
        """
        Count pairs.
        """
        pair_path = self.dict_to_path(
            "pair_counts", self._effective_pair_args(**kwargs)
        )
        if pair_path.is_file():
            try:
                logger.info("retrieved already saved pair count")
                return load_matrix(pair_path)
            except Exception as e:
                logger.info("creating pair counts, error while loading files: %s", e)

        logger.info("create new pair counts")
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        count_matrix = pair_counts.count_pairs(self.corpus, **kwargs)
        save_matrix(pair_path, count_matrix)
        return count_matrix

    def pmi_matrix(self, cds=0.75, pair_args=None, **kwargs):
        """
        Create a PMI matrix.
        """
        pair_args = {} if pair_args is None else pair_args
        # `**kwargs` reaches `count_pairs` too, so it has to be in the key
        pmi_path = self.dict_to_path(
            "pmi",
            {"cds": cds, "pair_args": self._effective_pair_args(pair_args, **kwargs)},
        )
        if pmi_path.is_file():
            try:
                logger.info("retrieved already saved pmi")
                return load_matrix(pmi_path)
            except Exception as e:
                logger.info("creating new pmi, error while loading files: %s", e)

        logger.info("create new pmi")
        counts = self.pair_counts(**pair_args, **kwargs)

        start = timer()
        pmi_matrix = pmi.calc_pmi(counts, cds)

        end = timer()
        logger.info("pmi took %s seconds", round(end - start, 2))

        pmi_path.parent.mkdir(parents=True, exist_ok=True)
        save_matrix(pmi_path, pmi_matrix)
        logger.info("matrix saved")

        return pmi_matrix

    @record
    def pmi(
        self,
        neg=1,
        cds=0.75,
        pair_args=None,
        keyed_vectors=False,
        evaluate=True,
        **kwargs,
    ):
        """
        Gets the PMI matrix.
        """
        pair_args = {} if pair_args is None else pair_args
        m = self.pmi_matrix(cds, pair_args, **kwargs)
        embd = pmi.PPMIEmbedding(m, neg=neg)
        if evaluate:
            eval_results = self.eval_sim(embd)
        if keyed_vectors:
            # because of the large dimensions, the matrix will get huge!
            embd = self.to_keyed_vectors(embd.m.toarray(), m.shape[0])
        if evaluate:
            return embd, eval_results
        return embd

    def svd_matrix(
        self, impl, impl_args=None, dim=500, neg=1, cds=0.75, pair_args=None, **kwargs
    ):
        """
        Do the actual SVD computation.
        """
        if impl not in VALID_IMPLS:
            raise ValueError(f"impl must be one of {sorted(VALID_IMPLS)}, got {impl!r}")

        impl_args = {} if impl_args is None else impl_args
        pair_args = {} if pair_args is None else pair_args

        svd_path = self.dict_to_path(
            "svd",
            {
                "impl": impl,
                "impl_args": impl_args,
                "neg": neg,
                "cds": cds,
                "dim": dim,
                # `**kwargs` reaches `count_pairs` too, so it has to be in the key
                "pair_args": self._effective_pair_args(pair_args, **kwargs),
            },
        )
        logger.debug("looking up the file: %s", svd_path)
        if svd_path.is_file():
            try:
                logger.info("retrieved already saved svd")
                return load_arrays(svd_path)
            except Exception as e:
                logger.info("creating new svd, error while loading files: %s", e)

        logger.info("creating new svd")
        m = self.pmi_matrix(cds, pair_args, **kwargs)
        m = pmi.PPMIEmbedding(m, neg=neg, normalize=False)

        start = timer()
        ut, s = svd.calc_svd(m, dim, impl, impl_args)
        end = timer()
        logger.info("svd took %s minutes", round((end - start) / 60, 2))

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
        impl_args=None,
        pair_args=None,
        keyed_vectors=False,
        evaluate=True,
        **kwargs,
    ):
        """
        Gets and SVD embedding.
        """
        impl_args = {} if impl_args is None else impl_args
        pair_args = {} if pair_args is None else pair_args
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
        if keyed_vectors:
            embedding = self.to_keyed_vectors(embedding.m, dim)
        if evaluate:
            return embedding, eval_results
        return embedding

    def to_keyed_vectors(self, embd_matrix, dim, delete_unknown=True):
        """
        Transform to gensim's keyed vectors structure for further usage.
        https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/keyedvectors.py
        """
        embd_matrix = np.asarray(embd_matrix)
        vectors = KeyedVectors(vector_size=dim)
        tokens = self.corpus.vocab.tokens
        if delete_unknown:
            # delete last row (for <UNK> token)
            embd_matrix = np.delete(embd_matrix, (-1), axis=0)
        else:
            # the last token is the UNK token so append it
            tokens.append("<UNK>")

        vectors.add_vectors(tokens, embd_matrix)
        return vectors

    def eval_sim(self, embd, **kwargs):
        """
        Evaluate the performance on word similarity datasets.
        NB: The corpus has to be initialized with the correct language.
        """
        return evaluation.eval_similarity(
            embd,
            self.corpus.vocab.token2id,
            self.corpus.preproc_fun,
            lang=self.corpus.lang,
            **kwargs,
        )

    def eval_analogy(self, embd, **kwargs):
        """
        Evaluate the performance on word analogies datasets.
        NB: The corpus has to be initialized with the correct language.
        """
        return evaluation.eval_analogies(
            embd,
            self.corpus.vocab.token2id,
            self.corpus.preproc_fun,
            lang=self.corpus.lang,
            **kwargs,
        )

    def results(self, **kwargs):
        """
        Retrieve evaluation results from the database.
        """
        return results_from_db(self.get_db(), **kwargs)
