"""
utility functions for i/o and other general funtionality
"""

import logging
import os
import pickle
import tempfile
from collections import defaultdict
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BrokenProcessPool(RuntimeError):
    """
    A worker pool died. Nearly always a missing ``if __name__ == "__main__":``.

    Raised in place of `concurrent.futures.process.BrokenProcessPool`, whose own
    message ("A process in the process pool was terminated abruptly") names a
    symptom and gives the reader nothing to act on.
    """


_MISSING_MAIN_GUARD = """\
a worker process died while `hyperhyper` was parallelising.

The overwhelmingly common cause is a script without a main guard. On macOS and
Windows (and anywhere `multiprocessing` uses the "spawn" start method) each
worker re-imports your script to reach the function it must run. Without a
guard, that re-import runs your script *again* from the top -- including the
`hyperhyper` call that started the pool -- so every worker starts its own pool,
and the recursion collapses.

Wrap the top-level code of your script:

    def main():
        corpus = hyperhyper.Corpus.from_texts(texts)
        ...

    if __name__ == "__main__":
        main()

In a notebook or REPL this cannot happen, and there is nothing to change.

If your script already has the guard, the worker died for another reason -- most
often it ran out of memory. The original exception is chained below.\
"""


def _default_workers():
    """
    Number of usable CPUs, respecting affinity/cgroup limits where available.
    """
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def _with_npz(f):
    """The path as a `str` ending in `.npz` (numpy appends it; we need it up front)."""
    f = str(f)
    return f if f.endswith(".npz") else f + ".npz"


@contextmanager
def atomic_path(final):
    """
    Yield a temporary path in the same directory, renamed over `final` on success.

    Every artifact in a bunch directory is a *cache*: it is written once and read
    back on the next run. Writing in place makes an interrupted run (Ctrl-C, a
    full disk, an OOM kill during a long SVD) leave a **truncated file that still
    looks like a valid cache entry** -- the next run finds the path, tries to
    load it, and either raises deep inside numpy/scipy or, worse, silently
    reports a stale-looking failure. Writing to a sibling temporary file and
    `os.replace`-ing it means the destination either does not exist or is
    complete; `os.replace` is atomic within a filesystem, and staying in the same
    directory guarantees that.

    The temporary file is removed if the writer raises, so a failed write leaves
    no litter behind either.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=final.parent, prefix=f".{final.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp)
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_arrays(f, a1, a2):
    """
    Save two numpy arrays to a single compressed ``.npz`` file at ``f``.

    Used to persist the SVD outputs ``(ut, s)``. ``f`` may be a path-like
    object; the ``.npz`` extension is appended if missing. The write is atomic
    (see `atomic_path`). Read back with ``load_arrays``.
    """
    final = _with_npz(f)
    with atomic_path(final) as tmp:
        # numpy would append `.npz` to a name that lacks it, defeating the rename
        np.savez_compressed(str(tmp) + ".npz", a1=a1, a2=a2)
        os.replace(str(tmp) + ".npz", tmp)


def load_arrays(f):
    """
    Load the two arrays written by ``save_arrays`` from ``f``.

    Accepts a path with or without the ``.npz`` extension (it is appended if
    missing) and returns the arrays as a ``(a1, a2)`` tuple.
    """
    # `allow_pickle=False` is numpy's default, but it is stated explicitly here
    # because it is a *security* property, not a preference: an `.npz` may carry
    # object arrays that execute code on load. These files are plain numeric
    # caches, so nothing legitimate needs the pickle path (ADR 0002).
    loader = np.load(_with_npz(f), allow_pickle=False)
    return loader["a1"], loader["a2"]


def save_matrix(f, m):
    """
    Save a scipy sparse matrix ``m`` to a compressed ``.npz`` file at ``f``.

    ``f`` may be a path-like object. The write is atomic (see `atomic_path`).
    Read back with ``load_matrix``.
    """
    with atomic_path(_with_npz(f)) as tmp:
        # scipy appends `.npz` unless the name already has it
        save_npz(str(tmp) + ".npz", m, compressed=True)
        os.replace(str(tmp) + ".npz", tmp)


def load_matrix(f):
    """
    Load a scipy sparse matrix from ``f`` (as written by ``save_matrix``).

    Accepts a path with or without the ``.npz`` extension (appended if missing).
    First tries scipy's ``load_npz``; if that fails with ``ValueError`` (an
    older file written in the legacy, hand-rolled CSR layout with separate
    ``data``/``indices``/``indptr``/``shape`` arrays), it falls back to
    reconstructing a ``csr_matrix`` from those arrays.
    """
    f = _with_npz(f)
    try:
        return load_npz(f)
    except ValueError:
        # fall back to the legacy, hand-rolled layout (CSR only)
        loader = np.load(f, allow_pickle=False)
        return csr_matrix(
            (loader["data"], loader["indices"], loader["indptr"]),
            shape=loader["shape"],
        )


class IdChunk:
    """
    One on-disk chunk of a corpus: ragged token-id sentences, stored flat.

    `.npz` cannot hold a ragged array, so a chunk is kept as a single flat id
    array plus the sentence boundaries -- which is *also* the layout the
    vectorized counter builds for itself, so reading a chunk in this form lets
    it skip the flattening entirely.

    Iterating yields ordinary Python lists, one per sentence, so the
    non-vectorized counting loop sees exactly what it saw when chunks were
    pickled `array('H')`s. `flat`/`offsets` are there for the callers that want
    the whole chunk at once.
    """

    __slots__ = ("flat", "offsets")

    def __init__(self, flat, offsets):
        self.flat = flat
        self.offsets = offsets

    @classmethod
    def from_sentences(cls, sentences):
        """Build a chunk from any iterable of id sequences."""
        sentences = [np.asarray(s, dtype=np.int64) for s in sentences]
        lengths = np.array([len(s) for s in sentences], dtype=np.int64)
        flat = np.concatenate(sentences) if sentences else np.zeros(0, dtype=np.int64)
        # the smallest width that holds every id, mirroring the `array('H')`
        # / `array('L')` choice the pickled format made
        dtype = np.uint16 if (flat.size and flat.max() <= 65535) else np.uint32
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        return cls(flat.astype(dtype, copy=False), offsets)

    def __len__(self):
        return len(self.offsets) - 1

    def __iter__(self):
        flat, offsets = self.flat, self.offsets
        for i in range(len(offsets) - 1):
            # `tolist()` gives Python ints, so the counting loop's dict keys and
            # comparisons are exactly what they were with `array('H')`
            yield flat[offsets[i] : offsets[i + 1]].tolist()

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            # a view of the boundaries, not a copy of every sentence: the probe
            # in `_estimate_serial_seconds` slices a couple of thousand
            # sentences off a chunk that may hold a hundred thousand
            return IdChunk(
                self.flat[self.offsets[start] : self.offsets[stop]],
                self.offsets[start : stop + 1] - self.offsets[start],
            )
        return self.flat[self.offsets[index] : self.offsets[index + 1]].tolist()


def save_id_chunk(path, sentences):
    """Write a corpus chunk as `.npz` (see `IdChunk`), atomically."""
    chunk = (
        sentences
        if isinstance(sentences, IdChunk)
        else IdChunk.from_sentences(sentences)
    )
    with atomic_path(path) as tmp:
        np.savez(str(tmp) + ".npz", flat=chunk.flat, offsets=chunk.offsets)
        os.replace(str(tmp) + ".npz", tmp)


def load_id_chunk(path):
    """
    Read a corpus chunk, in either format.

    `.npz` is the current one; `.pkl` is what bunches built before the switch
    hold, and they keep working -- the format is chosen by extension, never by
    sniffing, the same rule the evaluation data follows (ADR 0002). A pickled
    chunk is returned as-is (a list of `array('H')`), which every caller already
    handles.
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as loader:
            return IdChunk(loader["flat"], loader["offsets"])
    return read_pickle(path)


def chunks(seq, n):
    """
    Yield successive n-sized chunks from seq.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def map_pool(array, fun, total=None, desc=None, process_chunksize=100):
    """
    Apply `fun` to every item of `array` across a process pool.

    `process_chunksize` is the batching that matters: `Executor.map` hands each
    worker that many items per task and flattens the results incrementally, so
    batching the input by hand beforehand only bundles twice and serialises the
    work back onto one worker.
    """
    with futures.ProcessPoolExecutor(_default_workers()) as executor:
        try:
            if desc is None:
                return list(executor.map(fun, array, chunksize=process_chunksize))
            return list(
                tqdm(
                    executor.map(fun, array, chunksize=process_chunksize),
                    total=len(array) if total is None else total,
                    desc=desc,
                )
            )
        except futures.process.BrokenProcessPool as e:
            raise BrokenProcessPool(_MISSING_MAIN_GUARD) from e


def delete_folder(pth):
    """
    Recursively delete the directory ``pth`` and everything inside it.

    ``pth`` is a ``pathlib.Path``. Walks the tree, unlinking files and
    recursing into subdirectories, then removes the now-empty directory itself.
    """
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    pth.rmdir()


def to_pickle(ob, fn):
    """
    Pickle object ``ob`` to the file ``fn``, creating parent directories.

    ``fn`` is a ``pathlib.Path``; any missing parent directories are created.
    The write is atomic (see `atomic_path`): a half-written chunk pickle is the
    worst of the cache corruptions, because `read_pickle` on it raises from
    inside a worker process during counting.
    """
    with atomic_path(fn) as tmp, open(tmp, "wb") as outfile:
        pickle.dump(ob, outfile)


def read_pickle(fn):
    """
    Unpickle and return the object stored in the file ``fn``.

    **A bunch directory is a trusted local cache, never something to load from
    an untrusted source.** Unpickling executes code by design, so opening
    somebody else's bunch is equivalent to running their program. This is the
    honest answer rather than a fixable flaw: the alternative (a pickle-free
    on-disk format) was considered in ADR 0002 and deferred, because the chunk
    files are a regenerable derived cache and reworking the bunch format
    piecemeal is the wrong scope.
    """
    with open(fn, "rb") as infile:
        return pickle.load(infile)


def dsum(*dicts):
    """
    sum up numerical values in multiple dictionaries
    """
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)
