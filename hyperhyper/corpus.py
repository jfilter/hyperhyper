"""
represent a collection of texts
"""

import importlib
import json
import logging
import random
from array import array
from collections import Counter, defaultdict
from concurrent import futures
from pathlib import Path

from gensim.corpora import Dictionary
from gensim.utils import SaveLoad
from tqdm import tqdm

from .preprocessing import tokenize_texts_parallel_v2
from .utils import (
    _default_workers,
    atomic_path,
    chunks,
    dsum,
    load_id_chunk,
    read_pickle,
    save_id_chunk,
    to_pickle,
)

logger = logging.getLogger(__name__)

# Version of `corpus.json`; a reader refuses any other value.
CORPUS_FORMAT = "corpus-v1"


def _function_identity(f):
    """
    `{"module", "qualname"}` naming `f`, checked to resolve back to `f` itself.

    The same contract pickle enforced by reference: a top-level, importable
    function. A lambda, a nested function or a `functools.partial` is refused.
    """
    identity = {
        "module": getattr(f, "__module__", None),
        "qualname": getattr(f, "__qualname__", None),
    }
    message = (
        f"preproc_func {f!r} cannot be stored by name: it must be a top-level "
        f"function importable as module + qualified name (not a lambda, a "
        f"nested function or a functools.partial)"
    )
    try:
        resolved = _resolve_function(identity)
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        raise TypeError(message) from e
    if resolved is not f:
        raise TypeError(message)
    return identity


def _resolve_function(identity):
    """Import `identity["module"]` and walk `identity["qualname"]` inside it."""
    obj = importlib.import_module(identity["module"])
    for part in identity["qualname"].split("."):
        obj = getattr(obj, part)
    return obj


class Vocab(Dictionary):
    """
    Holds mapping for the integer ids to the tokens (words).
    """

    def __init__(self, texts=None, **kwargs):
        super().__init__(texts)
        if texts is not None:
            self.filter(**kwargs)

    def add_documents(self, documents, prune_at=2000000):
        """
        gensim's `add_documents`, producing the identical dictionary ~1.45x faster.

        gensim tallies each document in a Python loop and builds three throwaway
        collections per document; this tallies with `Counter` and keeps the
        id assignment exactly: a document's new tokens get the next ids in sorted
        order. The pruning checkpoints are gensim's, so `filter_extremes` sees
        the same state. `tests/test_vocab.py` checks it against gensim.
        """
        for docno, document in enumerate(documents):
            if docno % 10000 == 0:
                if prune_at is not None and len(self) > prune_at:
                    self.filter_extremes(no_below=0, no_above=1.0, keep_n=prune_at)
                logger.info("adding document #%i to %s", docno, self)
            if isinstance(document, str):
                raise TypeError(
                    "doc2bow expects an array of unicode tokens on input, not a single string"
                )
            # `iter`: given a mapping, `Counter` would read its values as counts
            counter = Counter(iter(document))
            if not all(type(w) is str for w in counter):
                # gensim decodes `bytes` tokens; leave that rare path to it.
                # `elements()` replays first-occurrence order, as gensim tallies.
                self.doc2bow(list(counter.elements()), allow_update=True)
                continue

            token2id, cfs, dfs = self.token2id, self.cfs, self.dfs
            for w in sorted(w for w in counter if w not in token2id):
                token2id[w] = len(token2id)
            for w, freq in counter.items():
                i = token2id[w]
                cfs[i] = cfs.get(i, 0) + freq
                dfs[i] = dfs.get(i, 0) + 1
            self.num_docs += 1
            self.num_pos += sum(counter.values())
            self.num_nnz += len(counter)
        logger.info(
            "built %s from %i documents (total %i corpus positions)",
            self,
            self.num_docs,
            self.num_pos,
        )

    def filter(self, no_below=0, no_above=1, keep_n=50000, keep_tokens=None):
        """
        Filter extremes with sane defaults.
        """
        self.filter_extremes(
            no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens
        )

    @property
    def size(self):
        return len(self.token2id)

    @property
    def tokens(self):
        """
        Return tokens as array (in order of id).
        """
        return [tup[0] for tup in sorted(self.token2id.items(), key=lambda x: x[1])]


class TransformToIndicesClosure:
    """
    A closure that is pickable, usefull for multiprocessing.
    <UNK> is the last ID (thus vocab_size)
    for sizes: https://docs.python.org/3/library/array.html
    """

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
    """
    Count token frequencies since gensim's dictionary only provides document frequencies.
    """
    counts = defaultdict(int)
    for text in texts:
        for token in text:
            counts[token] += 1
    return counts


def _texts_to_ids(args):
    """
    Turn one file's token lists into id sequences, replacing the file in place.

    The input is the `.pkl` of *token strings* written while the vocabulary was
    built; the output is the `.npz` id chunk the counter reads (`utils.IdChunk`).
    The intermediate pickle is removed, so a bunch ends up holding only the id
    chunks -- and only data, not pickled objects.
    """
    f, to_indices = args[0], args[1]
    texts = read_pickle(f)
    transformed = [to_indices(t) for t in texts]
    save_id_chunk(f.with_suffix(".npz"), transformed)
    f.unlink()
    counts = count_tokens(transformed)
    return len(transformed), counts


def texts_to_ids(input_text_fns, to_indices):
    """
    transform the raw texts to integer ids
    """
    total_len = 0
    all_counts = []
    with futures.ProcessPoolExecutor(_default_workers()) as executor:
        # A dictionary which will contain a list the future info in the key, and the filename in the value
        jobs = {}
        files_left = len(input_text_fns)
        files_iter = iter(input_text_fns)
        MAX_JOBS_IN_QUEUE = _default_workers() * 2

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
    f, preproc_func, view_fraction, seed = args[0], args[1], args[2], args[3]

    # a per-file RNG, seeded with a stable string, so the sampling below is
    # reproducible no matter which worker process picks up which file
    rng = random.Random(f"{seed}-{Path(f).name}")

    texts = f.read_text().split("\n")
    texts = preproc_func(texts)

    # temporary save processed files to continue working later
    to_pickle(texts, f.with_suffix(".pkl"))

    # skip at random
    if 0.999 > view_fraction < rng.random():
        return Vocab()
    return Vocab(texts)


class Corpus(SaveLoad):
    """
    An object to hold text.
    """

    def __init__(self, vocab, preproc_fun, texts=None, input_text_fns=None, lang="en"):
        self.vocab = vocab
        self.vocab_size = vocab.size
        self.lang = lang
        self.preproc_fun = preproc_fun

        if texts is None:
            to_indices = TransformToIndicesClosure(self)
            self.size, self.counts = texts_to_ids(input_text_fns, to_indices)
            # `_texts_to_ids` replaced each token-string `.pkl` with the `.npz`
            # id chunk beside it, so that is where the texts now live
            self.input_text_fns = [f.with_suffix(".npz") for f in input_text_fns]
            self.texts = None
        else:
            to_indices = TransformToIndicesClosure(self)
            transformed = [
                to_indices(t) for t in tqdm(texts, desc="transform to indices")
            ]
            self.texts = transformed
            self.counts = count_tokens(transformed)
            self.size = len(transformed)

    def texts_are_on_disk(self):
        """
        Whether `self.texts` holds paths to pickled chunks rather than the
        token lists themselves.
        """
        return bool(self.texts) and all(isinstance(t, Path) for t in self.texts)

    def texts_to_file(self, directory, text_chunk_size):
        """
        If we haven't created the temporay text files yet, do it here.
        We could't do it earlier since we only have location on the filesystem
        through the `bunch`.

        Safe to call repeatedly: one corpus may be handed to several `Bunch`es,
        and each keeps its texts in its own directory. Once the texts are on
        disk `self.texts` holds paths, so a later call reads them back instead
        of pickling the paths themselves.
        """
        directory = Path(directory)
        if self.texts is None:
            # Re-use the per-input-file pickles written during vocab building,
            # one chunk per input file. `text_chunk_size` is deliberately NOT
            # applied here: re-splitting would change the chunk filenames, and
            # the pair-count RNG is seeded per file (`random.Random(f"{seed}-
            # {name}")`), so `subsample="prob"` results would shift; the moved
            # chunk boundaries would also change the float32 merge for the
            # deterministic modes. The file-based path's granularity is the
            # user's input-file granularity by design. Split the input up front
            # to give the pool more chunks.
            fns = []
            directory.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(self.input_text_fns):
                new_path = (directory / f"texts_{i}.npz").resolve()
                # only works if data and bunch are on same file system
                f.rename(new_path)
                fns.append(new_path)
            self.texts = fns
            return

        if self.texts_are_on_disk():
            if {t.parent for t in self.texts} == {directory.resolve()}:
                # already written to this very directory
                return
            # a second bunch: recover the token lists from the first one's chunks
            texts = [t for fn in self.texts for t in load_id_chunk(fn)]
        else:
            texts = self.texts

        fns = []
        for i, c in enumerate(chunks(texts, text_chunk_size)):
            fn = (directory / f"texts_{i}.npz").resolve()
            save_id_chunk(fn, c)
            fns.append(fn)
        self.texts = fns

    def save_json(self, path):
        """
        Write the corpus as `corpus.json`: plain data, no pickle.

        `preproc_fun` is stored as module + qualified name, exactly what pickle
        stored by reference, and the text chunks relative to the file's
        directory, so a bunch can be moved. The texts must already be on disk.
        """
        path = Path(path)
        if not self.texts_are_on_disk():
            raise ValueError("write the texts to disk first (`texts_to_file`)")
        root = path.parent.resolve()
        v = self.vocab
        data = {
            "format": CORPUS_FORMAT,
            "lang": self.lang,
            "size": self.size,
            "vocab_size": self.vocab_size,
            "preproc_fun": _function_identity(self.preproc_fun),
            # `relative_to` raises for a chunk outside the bunch: no silent copy
            "texts": [str(Path(t).resolve().relative_to(root)) for t in self.texts],
            "counts": list(self.counts.items()),
            "vocab": {
                "token2id": list(v.token2id.items()),
                "cfs": list(v.cfs.items()),
                "dfs": list(v.dfs.items()),
                "num_docs": v.num_docs,
                "num_pos": v.num_pos,
                "num_nnz": v.num_nnz,
            },
        }
        with atomic_path(path) as tmp:
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load_json(path):
        """Read a corpus written by `save_json`."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("format") != CORPUS_FORMAT:
            raise ValueError(
                f"{path}: unsupported corpus format {data.get('format')!r}, "
                f"expected {CORPUS_FORMAT!r}"
            )
        vocab = Vocab()
        v = data["vocab"]
        vocab.token2id.update((t, i) for t, i in v["token2id"])
        vocab.cfs.update((i, c) for i, c in v["cfs"])
        vocab.dfs.update((i, c) for i, c in v["dfs"])
        vocab.num_docs, vocab.num_pos, vocab.num_nnz = (
            v["num_docs"],
            v["num_pos"],
            v["num_nnz"],
        )

        corpus = Corpus.__new__(Corpus)
        corpus.vocab = vocab
        corpus.vocab_size = data["vocab_size"]
        corpus.lang = data["lang"]
        identity = data["preproc_fun"]
        try:
            corpus.preproc_fun = _resolve_function(identity)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"{path}: the corpus was tokenized with "
                f"{identity['module']}.{identity['qualname']}, which cannot be "
                f"imported here; reopen the bunch where that function is importable"
            ) from e
        root = path.parent.resolve()
        corpus.texts = [root / t for t in data["texts"]]
        corpus.counts = defaultdict(int, ((i, c) for i, c in data["counts"]))
        corpus.size = data["size"]
        return corpus

    @staticmethod
    def from_file(input_path, limit=None, **kwargs):
        """
        Construct a Corpus from a text file with newline-delimited sentences.
        """
        logger.info("reading file")
        text = Path(input_path).read_text()
        lines = text.splitlines()
        if limit is not None:
            lines = lines[:limit]
        logger.info("done reading file")
        return Corpus.from_sents(lines, **kwargs)

    @staticmethod
    def from_sents(
        texts, vocab=None, preproc_func=tokenize_texts_parallel_v2, lang="en", **kwargs
    ):
        """
        Construct corpus from lists of sentences.
        """
        texts = preproc_func(texts)
        if vocab is None:
            vocab = Vocab(texts, **kwargs)
        corpus = Corpus(vocab, preproc_func, texts=texts, lang=lang)
        return corpus

    @staticmethod
    def from_texts(texts, preproc_func=tokenize_texts_parallel_v2, **kwargs):
        """
        Construct corpus from list of texts.

        Defaults to the lightweight v2 whitespace tokenizer (ADR 0002); pass
        `preproc_func=texts_to_sents` for the spaCy sentence-splitting /
        lemmatizing path (needs the `full` extra + a model).
        """
        return Corpus.from_sents(texts, preproc_func=preproc_func, **kwargs)

    @staticmethod
    def from_text_files(
        base_dir,
        preproc_func=tokenize_texts_parallel_v2,
        view_fraction=1,
        lang="en",
        seed=1312,
        **kwargs,
    ):
        """
        Construct a corpus from a folder of text files.
        The size of the text files determine the working memory size later on.
        This is usefull for larger amount of text.

        Args:
            base_dir (str): The directory with the text files.
            preproc_func (fun): The funcation to preprocess texts into sentences.
            view_fraction (float): Option to only look at portions of the text to determine the most frequent words.
            lang (str): The language of the texts, defaults to "en".
            seed (int): Seed for the random sampling of `view_fraction`.

        Returns:
            Corpus
        """
        voc = Vocab()
        # sorted: `glob` yields in filesystem order, which differs between
        # machines and would make the vocabulary depend on it
        input_text_fns = sorted(Path(base_dir).glob("*.txt"))
        proc_fns = [f.with_suffix(".pkl") for f in input_text_fns]
        built = {}

        with futures.ProcessPoolExecutor(_default_workers()) as executor:
            jobs = {}
            files_left = len(input_text_fns)
            files_iter = iter(input_text_fns)
            MAX_JOBS_IN_QUEUE = _default_workers() * 2

            with tqdm(total=len(input_text_fns), desc="build up vocab") as pbar:
                while files_left:
                    for this_file in files_iter:
                        job = executor.submit(
                            _build_vocab_from_file,
                            [this_file, preproc_func, view_fraction, seed],
                        )
                        jobs[job] = this_file
                        if len(jobs) > MAX_JOBS_IN_QUEUE:
                            break

                    for job in futures.as_completed(jobs):
                        files_left -= 1
                        pbar.update(1)
                        # buffer instead of merging here: `as_completed` yields
                        # in worker-completion order, and `filter_extremes`
                        # breaks document-frequency ties by insertion order, so
                        # merging as results arrive keeps a *different set of
                        # tokens* on every run
                        built[jobs[job]] = job.result()
                        del jobs[job]

        # merge into one vocab in a deterministic order
        for f in sorted(built):
            voc.merge_with(built[f])

        # only consider most frequent terms etc.
        voc.filter(**kwargs)

        return Corpus(voc, preproc_func, input_text_fns=proc_fns, lang=lang)
