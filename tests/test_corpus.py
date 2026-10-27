from hyperhyper.corpus import Corpus, TransformToIndicesClosure, Vocab


def test_from_file_reads_one_sentence_per_line_up_to_limit(tmp_path):
    f = tmp_path / "sents.txt"
    f.write_text("Ice-cream is cold.\nThe city's café\nnot read\n")
    corpus = Corpus.from_file(f, limit=2)
    assert corpus.size == 2
    assert [corpus.vocab[i] for i in corpus.texts[1]] == ["the", "city's", "café"]


def _corpus_with_vocab(n_tokens):
    corpus = Corpus.__new__(Corpus)
    # unfiltered: `Vocab(texts)` would cap it at its default `keep_n=50000`
    corpus.vocab = Vocab()
    corpus.vocab.add_documents([[f"w{i}" for i in range(n_tokens)]])
    return corpus


def test_ids_fit_in_16_bits_up_to_65535_tokens():
    corpus = _corpus_with_vocab(65535)
    ids = TransformToIndicesClosure(corpus)(["w7", "unknown"])
    assert ids.typecode == "H"
    # <UNK> is the id one past the vocabulary, 65535 itself
    assert list(ids) == [corpus.vocab.token2id["w7"], 65535]


def test_ids_widen_beyond_65535_tokens():
    corpus = _corpus_with_vocab(65536)
    ids = TransformToIndicesClosure(corpus)(["w7", "unknown"])
    assert ids.typecode == "L"
    assert list(ids) == [corpus.vocab.token2id["w7"], 65536]
