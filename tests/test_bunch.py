import tempfile
from pathlib import Path

import pytest

import hyperhyper


@pytest.fixture()
def corpus():
    some_text1 = """
    The English Wikipedia is the English-language edition of the free online encyclopedia Wikipedia. Founded on 15 January 2001, it is the first edition of Wikipedia and, as of April 2019, has the most articles of any of the editions.[2] As of June 2019, 12% of articles in all Wikipedias belong to the English-language edition. This share has gradually declined from more than 50 percent in 2003, due to the growth of Wikipedias in other languages.[3] As of 1 June 2019, there are 5,870,200 articles on the site,[4] having surpassed the 5 million mark on 1 November 2015.[5] In October 2015, the combined text of the English Wikipedia's articles totalled 11.5 gigabytes when compressed.[6]

    The Simple English Wikipedia is a variation in which most of the articles use only basic English vocabulary. There is also the Old English (Ænglisc/Anglo-Saxon) Wikipedia (angwiki). Community-produced news publications include The Signpost.[7]
    """

    some_text2 = """
    The English Wikipedia was the first Wikipedia edition and has remained the largest. It has pioneered many ideas as conventions, policies or features which were later adopted by Wikipedia editions in some of the other languages. These ideas include "featured articles",[8] the neutral-point-of-view policy,[9] navigation templates,[10] the sorting of short "stub" articles into sub-categories,[11] dispute resolution mechanisms such as mediation and arbitration,[12] and weekly collaborations.[13]

    The English Wikipedia has adopted features from Wikipedias in other languages. These features include verified revisions from the German Wikipedia (dewiki) and town population-lookup templates from the Dutch Wikipedia (nlwiki).

    Although the English Wikipedia stores images and audio files, as well as text files, many of the images have been moved to Wikimedia Commons with the same name, as passed-through files. However, the English Wikipedia also has fair-use images and audio/video files (with copyright restrictions), most of which are not allowed on Commons.

    Many of the most active participants in the Wikimedia Foundation, and the developers of the MediaWiki software that powers Wikipedia, are English users.
    """

    texts = [some_text1, some_text2]
    c = hyperhyper.Corpus.from_texts(texts)
    return c


def test_bunch(corpus):
    bunch = hyperhyper.Bunch("test_bunch", corpus, force_overwrite=True)
    pmi_matrix, _ = bunch.pmi()
    bunch.eval_sim(pmi_matrix)

    bunch.eval_analogy(pmi_matrix)

    # testing the evaluation of pmi
    english_idx = corpus.vocab.token2id["english"]
    wikipedia_idx = corpus.vocab.token2id["wikipedia"]
    for sim, token_idx in pmi_matrix.most_similar(english_idx):
        assert pmi_matrix.similarity(english_idx, token_idx) == pmi_matrix.similarity(token_idx, english_idx)
        assert pmi_matrix.similarity(english_idx, token_idx) == sim

    pmi_matrix.most_similar_vectors([english_idx], [wikipedia_idx])

    svd_matrix, _ = bunch.svd(dim=2)

    # testing the evaluation of svd
    english_idx = corpus.vocab.token2id["english"]
    for sim, token_idx in svd_matrix.most_similar(english_idx):
        assert svd_matrix.similarity(english_idx, token_idx) == svd_matrix.similarity(token_idx, english_idx)
        assert svd_matrix.similarity(english_idx, token_idx) == sim

    svd_matrix, _ = bunch.svd(dim=2, keyed_vectors=True)
    svd_matrix = bunch.svd(dim=3, keyed_vectors=True, evaluate=False)

    # `most_similar` comes from gensim's keyedvectors
    svd_matrix.most_similar("english")

    assert pmi_matrix.m.count_nonzero() > 0


def test_db_query(corpus):
    bunch = hyperhyper.Bunch("test_bunch", corpus, force_overwrite=True)
    bunch.svd(dim=2)
    res = bunch.results(query={"dim": 2, "pair_args": {"window": 2}})
    print(res)


def test_bunch_text_files():
    some_text1 = """
    The English Wikipedia is the English-language edition of the free online encyclopedia Wikipedia. Founded on 15 January 2001, it is the first edition of Wikipedia and, as of April 2019, has the most articles of any of the editions.[2] As of June 2019, 12% of articles in all Wikipedias belong to the English-language edition. This share has gradually declined from more than 50 percent in 2003, due to the growth of Wikipedias in other languages.[3] As of 1 June 2019, there are 5,870,200 articles on the site,[4] having surpassed the 5 million mark on 1 November 2015.[5] In October 2015, the combined text of the English Wikipedia's articles totalled 11.5 gigabytes when compressed.[6]

    The Simple English Wikipedia is a variation in which most of the articles use only basic English vocabulary. There is also the Old English (Ænglisc/Anglo-Saxon) Wikipedia (angwiki). Community-produced news publications include The Signpost.[7]
    """

    some_text2 = """
    The English Wikipedia was the first Wikipedia edition and has remained the largest. It has pioneered many ideas as conventions, policies or features which were later adopted by Wikipedia editions in some of the other languages. These ideas include "featured articles",[8] the neutral-point-of-view policy,[9] navigation templates,[10] the sorting of short "stub" articles into sub-categories,[11] dispute resolution mechanisms such as mediation and arbitration,[12] and weekly collaborations.[13]

    The English Wikipedia has adopted features from Wikipedias in other languages. These features include verified revisions from the German Wikipedia (dewiki) and town population-lookup templates from the Dutch Wikipedia (nlwiki).

    Although the English Wikipedia stores images and audio files, as well as text files, many of the images have been moved to Wikimedia Commons with the same name, as passed-through files. However, the English Wikipedia also has fair-use images and audio/video files (with copyright restrictions), most of which are not allowed on Commons.

    Many of the most active participants in the Wikimedia Foundation, and the developers of the MediaWiki software that powers Wikipedia, are English users.
    """

    texts = [some_text1, some_text2]
    # setup
    test_dir = tempfile.mkdtemp()
    for i, t in enumerate(texts):
        Path(test_dir + f"/{i}.txt").write_text(t)
    # test
    corpus = hyperhyper.Corpus.from_text_files(test_dir)
    bunch = hyperhyper.Bunch("test_bunch", corpus, force_overwrite=True)

    pmi_matrix, _ = bunch.pmi()
    bunch.eval_sim(pmi_matrix)
    svd_matrix, _ = bunch.svd(dim=2)
    svd_matrix, _ = bunch.svd(dim=2, keyed_vectors=True)
    svd_matrix = bunch.svd(dim=2, keyed_vectors=True, evaluate=False)

    print(svd_matrix.most_similar("english"))

    assert pmi_matrix.m.count_nonzero() > 0
