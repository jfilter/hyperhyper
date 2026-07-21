import pytest

import hyperhyper
from hyperhyper.preprocessing import texts_to_sents, tokenize_texts


def test_corpus(sample_texts):
    sents = []
    for t in sample_texts:
        sents += t.split("\n\n")
    corpus = hyperhyper.Corpus.from_sents(sents, preproc_func=tokenize_texts)
    assert corpus.size == len(sents)
    assert corpus.counts[corpus.vocab.token2id["wikipedia"]] > 0
    assert corpus.vocab.token2id["wikipedia"] == corpus.vocab.tokens.index("wikipedia")

    keys = corpus.vocab.token2id.keys()
    assert len(keys) > 0

    for k in keys:
        i = corpus.vocab.token2id[k]
        assert i < len(keys)


def test_texts(sample_texts):
    corpus = hyperhyper.Corpus.from_texts(sample_texts, preproc_func=tokenize_texts)
    assert corpus.size == len(sample_texts)
    assert "wikipedia" in corpus.vocab.token2id


def test_text_files(text_files):
    corpus = hyperhyper.Corpus.from_text_files(text_files, preproc_func=tokenize_texts)
    assert corpus.size > 2
    assert "wikipedia" in corpus.vocab.token2id


def test_text_files_view_fraction(text_files):
    corpus = hyperhyper.Corpus.from_text_files(
        text_files, preproc_func=tokenize_texts, view_fraction=0.2
    )
    assert corpus.size > 2


def test_text_files_vocab_is_deterministic(tmp_path):
    """
    Regression test: the vocabulary must not depend on worker completion order.

    Every rare word here has a document frequency of exactly 1, and `keep_n`
    cuts the vocabulary in the middle of that tie. `filter_extremes` breaks
    df-ties by insertion order, so merging the per-file vocabularies as the
    futures completed kept a *different set of words* on each run.
    """
    directory = tmp_path / "tied"
    directory.mkdir()
    common = "alpha beta gamma delta epsilon"
    for i in range(10):
        # avoid digits: the preprocessing maps them all to "0"
        rare = " ".join(f"rare{chr(97 + i)}{chr(97 + j)}" for j in range(20))
        (directory / f"f{chr(97 + i)}.txt").write_text(
            f"{common} {rare}\n", encoding="utf-8"
        )

    def vocab_of():
        corpus = hyperhyper.Corpus.from_text_files(
            directory, preproc_func=tokenize_texts, keep_n=60
        )
        return sorted(corpus.vocab.token2id)

    first = vocab_of()
    assert len(first) == 60
    for _ in range(2):
        assert vocab_of() == first


@pytest.mark.slow
def test_sent_split_spacy(spacy_model, sample_texts):
    """
    The default preprocessing splits the texts into sentences with spaCy, so
    the corpus ends up holding more entries than there are texts.
    """
    corpus = hyperhyper.Corpus.from_texts(sample_texts, preproc_func=texts_to_sents)
    assert corpus.size > len(sample_texts)
    assert "wikipedia" in corpus.vocab.token2id


@pytest.mark.slow
def test_text_files_spacy(spacy_model, text_files):
    corpus = hyperhyper.Corpus.from_text_files(text_files, preproc_func=texts_to_sents)
    assert corpus.size > 2
    assert "wikipedia" in corpus.vocab.token2id


def test_one_corpus_feeds_two_bunches(corpus, tmp_path):
    """
    `texts_to_file` used to overwrite `corpus.texts` with the paths it had just
    written, so a second bunch pickled those paths as if they were token lists
    and the workers died on `'PosixPath' object is not iterable`.
    """
    first = hyperhyper.Bunch(tmp_path / "first", corpus)
    second = hyperhyper.Bunch(tmp_path / "second", corpus)

    assert first.pair_counts(window=2).nnz > 0
    assert second.pair_counts(window=2).nnz > 0
    # each bunch keeps its own chunks, and both describe the same corpus
    assert first.pair_counts(window=2).nnz == second.pair_counts(window=2).nnz
