import re

from tqdm import tqdm

import spacy


# transform array of texts to arrays of sents (arrays of tokens) with simple preprocessing
def texts_to_sents(texts, model="en_core_web_sm"):
    results = []
    nlp = spacy.load(model, disable=["ner"])
    for doc in tqdm(nlp.pipe(texts), total=len(texts)):
        for s in doc.sents:
            results.append(
                [
                    re.sub(r"\d", "0", t.text.lower())
                    for t in s
                    if not any((t.is_punct, t.is_space, t.is_stop))
                ]
            )
    return results
