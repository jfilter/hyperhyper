"""
simple text preprocessing such as cleaning and tokenization
"""

import os
import re

from gensim.parsing.preprocessing import (preprocess_string,
                                          strip_non_alphanum, strip_tags)
from tqdm import tqdm

from .utils import map_pool

try:
    import spacy
except:
    spacy = None


def simple_preproc(text):
    """
    replace digits with 0 and lowercase text
    """
    return re.sub(r"\d", "0", text.lower())


def tokenize_string(text):
    """
    tokenize based on whitespaces
    """
    CUSTOM_FILTERS = [simple_preproc, strip_tags, strip_non_alphanum]
    return preprocess_string(text, CUSTOM_FILTERS)


def tokenize_texts(texts):
    """
    tokenize multiple texts (list of texts) based on whitespaces
    """
    return [tokenize_string(t) for t in texts]


def tokenize_texts_parallel(texts):
    """
    tokenize multiple texts based on whitespaces in parrallel
    """
    return map_pool(texts, tokenize_string)


def texts_to_sents(texts, model="en_core_web_sm", remove_stop=True, lemmatize=True):
    """
    transform list of texts to list of sents (list of tokens) and apply
    simple text preprocessing
    """
    texts = [strip_tags(t) for t in texts]
    results = []

    assert spacy is not None, 'please install spacy, i.e., "pip install spacy"'

    try:
        nlp = spacy.load(model, disable=["ner"])
    except Exception as e:
        print(e, "\ntrying to download model...")
        os.system("python -m spacy download " + model)
        nlp = spacy.load(model, disable=["ner"])

    for doc in tqdm(nlp.pipe(texts), total=len(texts), desc="texts to sents"):
        for s in doc.sents:
            results.append(
                [
                    simple_preproc(
                        strip_non_alphanum(t.lemma_ if lemmatize else t.text)
                    )
                    for t in s
                    if not any((t.is_punct, t.is_space, remove_stop and t.is_stop))
                ]
            )
    return results
