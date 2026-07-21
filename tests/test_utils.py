import numpy as np
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
