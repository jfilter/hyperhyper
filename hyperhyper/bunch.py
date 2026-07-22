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
    delete_folder,
    load_matrix,
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

# How many text chunks to cut a corpus into. The chunk files are the *only*
# unit of parallelism `count_pairs` has -- one task per chunk -- so the chunk
# count, not the chunk size, is what decides whether the machine is used. The
# old fixed 100k-sentence default produced 3 chunks for a 250k-sentence corpus,
# which on a 10-core machine is a hard ceiling of 3x however many cores are
# present, and measured out at 1.8x because the three chunks did not take equal
# time.
#
# This is a FIXED number and deliberately not `workers * k`. The chunk layout is
# part of the *answer*, not part of the schedule: each chunk is counted into its
# own float32 matrix and those are then summed, so cutting the same corpus into
# a different number of pieces gives a different matrix. Deriving the count from
# the local core count would mean an 8-core and a 16-core machine building the
# same bunch from scratch could not reproduce each other's numbers. The pool
# size is still `_default_workers()` -- only the data partitioning is pinned.
#
# 40 is what a 10-core machine produced under the previous `workers * 4` rule,
# and it is the value the 1.68x speedup on the 250k corpus (3 chunks -> 40) was
# measured with. It was swept on 250k- and 500k-sentence corpora at 1 to 8
# chunks per worker: the wall clock is flat to within noise from 1 to 6
# (5.58-5.70s at 250k) and only starts to slip at 8 (6.12s), where the per-chunk
# overhead -- a pickled sparse matrix back to the parent, plus one more step in
# a merge that has to stay a strict left fold -- begins to show. 40 sits in the
# middle of that plateau, so it keeps its headroom on machines with fewer or
# more cores than the one it was measured on: chunks do not cost the same (the
# three chunks of the 250k corpus took 7.0s, 10.9s and 3.8s, a 2.9x spread), and
# a chunk count near the core count leaves nothing to rebalance that spread with.
TARGET_TEXT_CHUNKS = 40

# Floor: below this, a chunk stops being worth a task at all -- the round trip
# through the pool costs more than the counting.
MIN_TEXT_CHUNK_SIZE = 500

# Ceiling: one chunk is loaded into a worker whole, so this is what bounds a
# worker's peak memory. It is the old fixed default, kept in that role.
MAX_TEXT_CHUNK_SIZE = 100_000


def _auto_text_chunk_size(n_texts):
    """
    Pick a chunk size that gives the pool a sensible number of chunks to spread.

    A pure function of the corpus size: the same corpus is cut the same way on
    every machine, so a bunch built from scratch elsewhere counts the same
    partial matrices (see `TARGET_TEXT_CHUNKS`).
    """
    size = (
        math.ceil(n_texts / TARGET_TEXT_CHUNKS) if n_texts > 0 else MIN_TEXT_CHUNK_SIZE
    )
    return max(MIN_TEXT_CHUNK_SIZE, min(size, MAX_TEXT_CHUNK_SIZE))


def _canonical(value):
    """
    Normalise a parameter value so that equivalent values serialise identically.

    Integral floats become ints (`dim=500.0` and `dim=500` are the same run),
    and the same is done inside nested containers.

    Sets and frozensets are turned into a *sorted* list: a `set`'s iteration
    order is hash-randomised (it changes with `PYTHONHASHSEED`), so serialising
    one via `repr` gave a different digest -- and a silent cache miss -- on every
    run. Anything `json.dumps` cannot serialise natively is rejected rather than
    stringified through `repr` (whose output is likewise unstable for many
    types), so a cache key is always reproducible or is loudly refused.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        items = [_canonical(v) for v in value]
        # sort by a canonical serialisation so heterogeneous but JSON-able
        # elements (which need not be mutually `<`-comparable) still order
        # deterministically
        return sorted(items, key=lambda e: json.dumps(e, sort_keys=True))
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    raise TypeError(
        f"cannot build a stable cache key from a {type(value).__name__!r} "
        f"value ({value!r}): cache parameters must be JSON-serialisable "
        f"(str, int, float, bool, None, or mappings/sequences/sets of those)"
    )


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
        `None`, which sizes the chunks from the corpus alone (see
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
        # no `default=` fallback: `_canonical` has already turned every value
        # into something `json.dumps` handles natively, or raised. A `repr`
        # fallback here is what let a hash-randomised `set` through in the first
        # place.
        blob = json.dumps(canonical, sort_keys=True)
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
        self,
        impl,
        impl_args=None,
        dim=500,
        neg=1,
        cds=0.75,
        add_context=False,
        pair_args=None,
        **kwargs,
    ):
        """
        Do the actual SVD computation.

        Returns ``(ut, s, vt)``: the left singular vectors, the singular values
        and (only when ``add_context``) the right singular vectors ``Vᵀ`` needed
        for the ``w+c`` representation; ``vt`` is ``None`` otherwise.

        ``add_context`` participates in the cache key, so the ``w+c`` factorization
        (which persists three arrays) and the word-only one (two arrays) never
        share a cache entry. The word-only file keeps the exact ``a1``/``a2``
        layout the previous ``save_arrays`` wrote, so pre-existing caches still
        load.
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
                "add_context": add_context,
                # `**kwargs` reaches `count_pairs` too, so it has to be in the key
                "pair_args": self._effective_pair_args(pair_args, **kwargs),
            },
        )
        logger.debug("looking up the file: %s", svd_path)
        if svd_path.is_file():
            try:
                logger.info("retrieved already saved svd")
                loader = np.load(str(svd_path))
                vt = loader["a3"] if "a3" in loader.files else None
                return loader["a1"], loader["a2"], vt
            except Exception as e:
                logger.info("creating new svd, error while loading files: %s", e)

        logger.info("creating new svd")
        m = self.pmi_matrix(cds, pair_args, **kwargs)
        m = pmi.PPMIEmbedding(m, neg=neg, normalize=False)

        start = timer()
        ut, s, vt = svd.calc_svd(m, dim, impl, impl_args)
        end = timer()
        logger.info("svd took %s minutes", round((end - start) / 60, 2))

        svd_path.parent.mkdir(parents=True, exist_ok=True)
        # `a1`/`a2` match the historical `save_arrays` layout; `a3` (the context
        # vectors) is only written -- and only needed -- for `w+c`.
        arrays = {"a1": ut, "a2": s}
        if add_context:
            arrays["a3"] = vt
        np.savez_compressed(str(svd_path), **arrays)
        logger.info("svd arrays saved")

        return ut, s, (vt if add_context else None)

    @record
    def svd(
        self,
        dim=500,
        eig=0,
        neg=1,
        cds=0.75,
        impl="scipy",
        impl_args=None,
        add_context=False,
        pair_args=None,
        keyed_vectors=False,
        evaluate=True,
        **kwargs,
    ):
        """
        Gets and SVD embedding.

        ``add_context=True`` builds the ``w+c`` representation of Levy & Goldberg
        (2015) -- the context vectors are added to the word vectors. It defaults
        off, so the standard word-only embedding (and every recorded result) is
        unchanged.
        """
        impl_args = {} if impl_args is None else impl_args
        pair_args = {} if pair_args is None else pair_args
        ut, s, vt = self.svd_matrix(
            impl=impl,
            impl_args=impl_args,
            dim=dim,
            neg=neg,
            cds=cds,
            add_context=add_context,
            pair_args=pair_args,
            **kwargs,
        )
        embedding = svd.SVDEmbedding(ut, s, eig=eig, vt=vt, add_context=add_context)

        if evaluate:
            eval_results = self.eval_sim(embedding)
        if keyed_vectors:
            # `calc_svd` may hand back fewer columns than `dim` (a rank-deficient
            # or over-large request is clamped/truncated), so take the width from
            # the embedding itself -- for a normal `dim < rank` request this is
            # exactly `dim`.
            embedding = self.to_keyed_vectors(embedding.m, embedding.m.shape[1])
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

        Pass `data_dir` (and optionally `include_bundled=False`) to evaluate on
        your own analogy datasets; see `evaluation.read_test_data`.
        """
        return evaluation.eval_analogies(
            embd,
            self.corpus.vocab.token2id,
            self.corpus.preproc_fun,
            lang=self.corpus.lang,
            **kwargs,
        )

    def dataset_coverage(self, kind="ws", **kwargs):
        """
        Report the in-vocabulary fraction of each evaluation dataset.

        Lets you learn, before training an embedding, which bundled or custom
        (`data_dir`) test sets this corpus's vocabulary can actually be scored
        on. Computed under the corpus's own preprocessing, so it matches what
        the evaluator will see. See `evaluation.dataset_coverage`.
        """
        return evaluation.dataset_coverage(
            self.corpus.vocab.token2id,
            self.corpus.preproc_fun,
            lang=self.corpus.lang,
            kind=kind,
            **kwargs,
        )

    def results(self, **kwargs):
        """
        Retrieve evaluation results from the database.
        """
        return results_from_db(self.get_db(), **kwargs)
