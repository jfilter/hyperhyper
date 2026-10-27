"""
`Vocab.add_documents` replaces gensim's for speed, so gensim's is the oracle.

The ids matter beyond lookup: `filter_extremes` breaks document-frequency ties by
id order, so a different id assignment would keep a different set of tokens.
"""

import random

import pytest
from gensim.corpora import Dictionary

from hyperhyper.corpus import Vocab


def _state(d):
    return (
        list(d.token2id.items()),
        list(d.cfs.items()),
        list(d.dfs.items()),
        d.num_docs,
        d.num_pos,
        d.num_nnz,
    )


def _zipf_docs(n, vocab, seed):
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(vocab)]
    weights = [1 / (i + 1) for i in range(vocab)]
    return [rng.choices(words, weights, k=rng.randrange(25)) for _ in range(n)]


def _built(cls, docs, **kwargs):
    d = cls()
    d.add_documents(docs, **kwargs)
    return d


def test_matches_gensim_exactly():
    docs = _zipf_docs(3000, 800, seed=1)
    assert _state(_built(Vocab, docs)) == _state(_built(Dictionary, docs))


def test_matches_gensim_when_added_in_several_batches():
    docs = _zipf_docs(2000, 300, seed=2)
    ours, theirs = Vocab(), Dictionary()
    for batch in (docs[:700], docs[700:]):
        ours.add_documents(batch)
        theirs.add_documents(batch)
    assert _state(ours) == _state(theirs)


def test_matches_gensim_through_pruning():
    # 25k docs crosses two of gensim's 10k-doc pruning checkpoints
    docs = _zipf_docs(25000, 5000, seed=3)
    ours = _built(Vocab, docs, prune_at=400)
    assert _state(ours) == _state(_built(Dictionary, docs, prune_at=400))
    assert len(ours) <= 5000


def test_matches_gensim_on_bytes_and_generator_documents():
    docs = [[b"caf\xc3\xa9", "café", "b"], ["a"], [], ["b", b"a"]]
    ours = _built(Vocab, (iter(d) for d in docs))
    assert _state(ours) == _state(_built(Dictionary, docs))


@pytest.mark.parametrize(
    "docs", [[{"a": 5, "b": 2}, ("b", "c")], [{"a": "x"}], [{"b", "a"}, []]]
)
def test_matches_gensim_on_mapping_tuple_and_set_documents(docs):
    # gensim iterates a mapping's keys; `Counter(mapping)` would read its values
    assert _state(_built(Vocab, docs)) == _state(_built(Dictionary, docs))


def test_matches_gensim_without_pruning():
    docs = _zipf_docs(12000, 3000, seed=5)
    assert _state(_built(Vocab, docs, prune_at=None)) == _state(
        _built(Dictionary, docs, prune_at=None)
    )


def test_constructor_filters_the_same_tokens_as_gensim():
    docs = _zipf_docs(3000, 2000, seed=4)
    ours = Vocab(docs, keep_n=500)
    theirs = Dictionary(docs)
    theirs.filter_extremes(no_below=0, no_above=1, keep_n=500)
    assert _state(ours) == _state(theirs)


def test_rejects_a_string_document_like_gensim():
    with pytest.raises(TypeError):
        Vocab().add_documents(["not tokenized"])
