import pytest

import hyperhyper

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


def test_corpus():
    corpus = hyperhyper.Corpus.from_sents(texts)
    assert corpus.size == 2
    assert corpus.counts[corpus.vocab.token2id["wikipedia"]] > 0
    assert corpus.vocab.token2id["wikipedia"] == corpus.vocab.tokens.index("wikipedia")

    keys = corpus.vocab.token2id.keys()
    print(len(keys))

    for k in keys:
        i = corpus.vocab.token2id[k]
        assert i < len(keys)


def test_sent_split():
    corpus = hyperhyper.Corpus.from_texts(texts)
    print(corpus.texts)
    assert corpus.size > 2
