import re

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_non_alphanum,
    strip_tags,
)
from tqdm import tqdm

try:
    import spacy
except:
    pass


def simple_preproc(t):
    return re.sub(r"\d", "0", t.lower())


CUSTOM_FILTERS = [simple_preproc, strip_tags, strip_non_alphanum]


def simple_tokenizer(text):
    return preprocess_string(text, CUSTOM_FILTERS)


# transform array of texts to arrays of sents (arrays of tokens) with simple preprocessing
def texts_to_sents(texts, model="en_core_web_sm"):
    if type(texts) is str:
        # this is called for evaluation
        return simple_preproc(texts)

    texts = [strip_tags(t) for t in texts]
    results = []
    nlp = spacy.load(model, disable=["ner"])
    for doc in tqdm(nlp.pipe(texts), total=len(texts), desc="texts to sents"):
        for s in doc.sents:
            results.append(
                [
                    simple_preproc(t.text)
                    for t in s
                    if not any((t.is_punct, t.is_space, t.is_stop))
                ]
            )
    return results
