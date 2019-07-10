import pytest

import hyperhyper


def foo(li):
    return [pow(x, 10) for x in li]


def test_map_chunks():
    some_list = list(range(100))
    results = hyperhyper.utils.map_pool_chunks(
        some_list, foo, chunk_size=10, combine=True
    )
    assert len(results) == 100
    assert results[50] == pow(50, 10)
    print(results)
