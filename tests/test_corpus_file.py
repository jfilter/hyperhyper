"""
`corpus.json` must carry everything the pickled `corpus.pkl` carried, so a
reopened bunch counts exactly what the bunch that wrote it counted.
"""

import functools
import json
import shutil

import numpy as np
import pytest

import hyperhyper
from hyperhyper.bunch import CORPUS_FILE, LEGACY_CORPUS_FILE
from hyperhyper.preprocessing import tokenize_texts


def _state(corpus):
    v = corpus.vocab
    return (
        list(v.token2id.items()),
        list(v.cfs.items()),
        list(v.dfs.items()),
        (v.num_docs, v.num_pos, v.num_nnz, v.size),
        corpus.vocab_size,
        corpus.lang,
        corpus.preproc_fun,
        dict(corpus.counts),
        corpus.size,
        [t.name for t in corpus.texts],
    )


def test_reopened_bunch_has_the_same_corpus(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus)
    assert not (bunch_path / LEGACY_CORPUS_FILE).exists()
    reopened = hyperhyper.Bunch(bunch_path).corpus
    assert _state(reopened) == _state(corpus)
    # the id of a token absent from the vocab still counts as 0
    assert reopened.counts[10**9] == 0


@pytest.mark.parametrize("subsample", ["deter", "prob"])
def test_reopened_bunch_counts_the_same_bits(corpus, bunch_path, tmp_path, subsample):
    # "prob" draws from an RNG seeded per chunk, so it also pins the chunk names
    pair_args = {"subsample": subsample, "subsample_factor": 1e-2}
    expected = hyperhyper.Bunch(bunch_path, corpus).pmi_matrix(pair_args=pair_args)
    # moved, with only corpus file and chunks: the matrix is recounted from them
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(bunch_path / CORPUS_FILE, moved)
    shutil.copytree(bunch_path / "texts", moved / "texts")
    shutil.rmtree(bunch_path)
    actual = hyperhyper.Bunch(moved).pmi_matrix(pair_args=pair_args)
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())


def test_chunk_paths_are_relative_to_the_bunch(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus)
    data = json.loads((bunch_path / CORPUS_FILE).read_text())
    assert data["texts"] == [f"texts/texts_{i}.npz" for i in range(len(data["texts"]))]
    assert data["preproc_fun"] == {
        "module": "hyperhyper.preprocessing",
        "qualname": "tokenize_texts",
    }


def test_a_legacy_pickle_bunch_still_opens(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus)
    (bunch_path / CORPUS_FILE).unlink()
    corpus.save(str(bunch_path / LEGACY_CORPUS_FILE))
    reopened = hyperhyper.Bunch(bunch_path)
    assert _state(reopened.corpus) == _state(corpus)
    # opening it leaves the pickle as it was
    assert not (bunch_path / CORPUS_FILE).exists()
    with pytest.raises(ValueError):
        hyperhyper.Bunch(bunch_path, corpus)


@pytest.mark.parametrize(
    "preproc",
    [lambda texts: tokenize_texts(texts), functools.partial(tokenize_texts)],
    ids=["lambda", "partial"],
)
def test_a_preproc_func_without_an_importable_name_is_refused(preproc, bunch_path):
    corpus = hyperhyper.Corpus.from_texts(["a b c. d e."], preproc_func=preproc)
    with pytest.raises(TypeError, match="cannot be stored by name"):
        hyperhyper.Bunch(bunch_path, corpus)


def test_an_unknown_format_is_refused(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus)
    path = bunch_path / CORPUS_FILE
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "format": "corpus-v0"}))
    with pytest.raises(ValueError, match="unsupported corpus format"):
        hyperhyper.Bunch(bunch_path)


def test_a_tokenizer_that_cannot_be_imported_is_named(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus)
    path = bunch_path / CORPUS_FILE
    data = json.loads(path.read_text())
    gone = {"module": "__main__", "qualname": "my_tokenizer"}
    path.write_text(json.dumps({**data, "preproc_fun": gone}))
    with pytest.raises(ImportError, match=r"__main__\.my_tokenizer"):
        hyperhyper.Bunch(bunch_path)


def test_texts_in_memory_are_refused(corpus, tmp_path):
    with pytest.raises(ValueError, match="texts_to_file"):
        corpus.save_json(tmp_path / CORPUS_FILE)


def test_a_corpus_built_from_text_files_reopens(text_files, bunch_path):
    corpus = hyperhyper.Corpus.from_text_files(text_files, preproc_func=tokenize_texts)
    hyperhyper.Bunch(bunch_path, corpus)
    assert _state(hyperhyper.Bunch(bunch_path).corpus) == _state(corpus)
