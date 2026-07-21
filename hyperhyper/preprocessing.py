"""
simple text preprocessing such as cleaning and tokenization
"""

import logging
import re
import sys

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_non_alphanum,
    strip_tags,
)
from tqdm import tqdm

from .utils import map_pool

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.cli import download as spacy_download
except Exception as e:
    # deliberately not just `ImportError`: `import spacy` pulls in spacy.cli,
    # which pulls in typer/click, and an incompatible click raises TypeError
    # rather than ImportError. spaCy is optional, so a broken install has to
    # degrade to "spaCy unavailable" instead of breaking `import hyperhyper`.
    logger.debug("spaCy is not usable, disabling it: %r", e)
    spacy = None
    spacy_download = None


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


def texts_to_sents(
    texts, model="en_core_web_sm", remove_stop=True, lemmatize=True, n_process=1
):
    """
    transform list of texts to list of sents (list of tokens) and apply
    simple text preprocessing

    Args:
        n_process (int): Number of processes for `nlp.pipe`, defaults to 1.
            Only raise this if you know the call is *not* already running
            inside a process pool (`Corpus.from_text_files` runs one) and the
            calling module is protected by an `if __name__ == "__main__"`
            guard. spaCy re-imports the parent module in every child, so
            without a guard an unguarded script re-enters this function and
            spawns pools forever.
    """
    texts = [strip_tags(t) for t in texts]
    results = []

    if spacy is None:
        raise ModuleNotFoundError(
            'spaCy is required for texts_to_sents; install with: pip install "hyperhyper[full]"'
        )

    try:
        nlp = spacy.load(model, exclude=["ner"])
    except OSError as e:
        logger.warning("%s, trying to download model %s ...", e, model)
        try:
            spacy_download(model)
        except SystemExit as exc:
            raise RuntimeError(
                f"downloading the spaCy model {model!r} failed; install it "
                f"manually with: {sys.executable} -m spacy download {model}"
            ) from exc
        nlp = spacy.load(model, exclude=["ner"])

    for doc in tqdm(
        nlp.pipe(texts, batch_size=1000, n_process=n_process),
        total=len(texts),
        desc="texts to sents",
    ):
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
