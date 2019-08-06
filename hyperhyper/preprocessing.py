import os
import re

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_non_alphanum,
    strip_tags,
)
from tqdm import tqdm

from .utils import map_pool

try:
    import spacy
except:
    spacy = None


def simple_preproc(t):
    return re.sub(r"\d", "0", t.lower())


def tokenize_string(text):
    CUSTOM_FILTERS = [simple_preproc, strip_tags, strip_non_alphanum]
    return preprocess_string(text, CUSTOM_FILTERS)


def tokenize_texts(texts):
    # work on multiple texts in parallel
    return [tokenize_string(t) for t in texts]


def tokenize_texts_parallel(texts):
    return map_pool(texts, tokenize_string)


# transform array of texts to arrays of sents (arrays of tokens) with simple preprocessing
def texts_to_sents(texts, model="en_core_web_sm", remove_stop=True, lemmatize=True):
    texts = [strip_tags(t) for t in texts]
    results = []

    assert spacy is not None, 'please install spacy, i.e., "pip install spacy"'

    try:
        nlp = spacy.load(model, disable=["ner"])
    except Exception as e:
        print(e)
        print("trying to download model...")
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
