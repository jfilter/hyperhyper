import pytest

import hyperhyper


def test_map_chunks():
    some_list = list(range(100))
    results = hyperhyper.utils.map_chunks(
        some_list, lambda li: [pow(x, 10) for x in li], chunk_size=10
    )
    assert len(results) == 100
    assert results[50] == pow(50, 10)
    print(results)
