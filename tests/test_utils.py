import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from hyperhyper import utils


def raise_to_the_tenth(x):
    return pow(x, 10)


def test_map_pool_preserves_length_and_order():
    some_list = list(range(100))
    results = utils.map_pool(some_list, raise_to_the_tenth)
    assert len(results) == 100
    assert results[50] == pow(50, 10)
    assert results == [pow(x, 10) for x in some_list]


def test_dsum():
    assert utils.dsum({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 5, "c": 4}
    assert utils.dsum() == {}
    assert utils.dsum({"a": 1}) == {"a": 1}
    # a plain dict, so a lookup of a missing key raises instead of yielding 0
    assert type(utils.dsum({"a": 1})) is dict


def test_chunks():
    assert list(utils.chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(utils.chunks([], 2)) == []


def test_matrix_roundtrip(tmp_path):
    m = csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float32))
    path = tmp_path / "m.npz"

    utils.save_matrix(path, m)
    loaded = utils.load_matrix(path)

    assert loaded.shape == m.shape
    assert loaded.dtype == m.dtype
    assert np.array_equal(loaded.toarray(), m.toarray())


def test_matrix_legacy_format_still_loads(tmp_path):
    """
    Users have caches on disk that were written by the hand-rolled predecessor
    of `save_npz`, so the fallback read path has to keep working.
    """
    m = csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float32))
    path = tmp_path / "legacy.npz"
    np.savez(path, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)

    loaded = utils.load_matrix(path)
    assert np.array_equal(loaded.toarray(), m.toarray())


def test_arrays_roundtrip(tmp_path):
    a1 = np.arange(6).reshape(2, 3)
    a2 = np.array([1.5, 2.5])
    path = tmp_path / "arrays"

    utils.save_arrays(path, a1, a2)
    loaded1, loaded2 = utils.load_arrays(path)

    assert np.array_equal(loaded1, a1)
    assert np.array_equal(loaded2, a2)


def test_default_workers():
    assert utils._default_workers() >= 1


# --- atomic writes (ADR 0002) ----------------------------------------------
#
# Every artifact in a bunch directory is a regenerable cache that is written
# once and read back on the next run. The failure these guard against is the
# quiet one: an interrupted write (Ctrl-C, full disk, OOM kill during a long
# SVD) leaving a *truncated file that still looks like a valid cache entry*, so
# the next run finds the path and fails deep inside numpy/scipy instead of
# simply rebuilding.


def test_atomic_path_leaves_no_file_when_the_writer_raises(tmp_path):
    target = tmp_path / "sub" / "artifact.bin"
    with pytest.raises(RuntimeError), utils.atomic_path(target) as tmp:
        tmp.write_bytes(b"half a file")
        raise RuntimeError("interrupted")
    assert not target.exists()
    # and no temporary litter either
    assert list((tmp_path / "sub").iterdir()) == []


def test_atomic_path_does_not_clobber_an_existing_file_on_failure(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"the good previous cache entry")
    with pytest.raises(RuntimeError), utils.atomic_path(target) as tmp:
        tmp.write_bytes(b"garbage")
        raise RuntimeError("interrupted")
    assert target.read_bytes() == b"the good previous cache entry"


def test_atomic_path_replaces_on_success(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"old")
    with utils.atomic_path(target) as tmp:
        tmp.write_bytes(b"new")
    assert target.read_bytes() == b"new"


def test_save_matrix_does_not_leave_a_stray_npz(tmp_path):
    # numpy/scipy append `.npz` to a name that lacks it, which would defeat the
    # rename and leave the temporary file behind under a different name
    m = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    utils.save_matrix(tmp_path / "m", m)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.npz"]
    assert utils.load_matrix(tmp_path / "m").shape == (2, 2)


def test_save_arrays_does_not_leave_a_stray_npz(tmp_path):
    utils.save_arrays(tmp_path / "a", np.arange(4.0), np.arange(2.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.npz"]
    a1, _a2 = utils.load_arrays(tmp_path / "a")
    assert a1.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_to_pickle_is_atomic(tmp_path):
    target = tmp_path / "nested" / "obj.pkl"
    utils.to_pickle({"a": 1}, target)
    assert utils.read_pickle(target) == {"a": 1}
    assert sorted(p.name for p in target.parent.iterdir()) == ["obj.pkl"]


def test_load_arrays_refuses_a_pickled_object_array(tmp_path):
    # `allow_pickle=False` is a security property, not a preference: an `.npz`
    # carrying an object array executes code on load. These caches are plain
    # numeric data, so nothing legitimate needs that path.
    path = tmp_path / "evil.npz"
    np.savez(path, a1=np.array([{"code": "here"}], dtype=object), a2=np.arange(2.0))
    with pytest.raises(ValueError, match="allow_pickle=False"):
        utils.load_arrays(path)


def _kill_own_process(_x):
    """A worker that dies without raising -- what a spawn recursion looks like."""
    os._exit(1)


def test_map_pool_explains_a_dead_worker(tmp_path):
    # The stdlib raises "A process in the process pool was terminated abruptly",
    # which names a symptom and leaves the reader with nothing to act on. In
    # practice the cause is nearly always a script without an
    # `if __name__ == "__main__":` guard, so the message has to say so.
    with pytest.raises(utils.BrokenProcessPool, match=r'__name__ == "__main__"'):
        utils.map_pool([1, 2, 3], _kill_own_process, process_chunksize=1)
