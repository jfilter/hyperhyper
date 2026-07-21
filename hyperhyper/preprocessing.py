"""
simple text preprocessing such as cleaning and tokenization
"""

import logging
import multiprocessing
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

# Tokenizing in a process pool only pays off once there is enough text to
# amortize spawning one interpreter per core, and every child re-imports this
# module. Measured on an M1 Pro (10 workers, Python 3.12), tokenizing real
# corpus lines with `tokenize_string`, pool already warm:
#
#     input                    chars   serial     pool
#     100 sentences              16k    0.001s    2.303s
#     12k sentences              2.0M   0.075s    3.870s
#     250k sentences            40.6M   1.595s    3.621s
#     2.5k documents            40.8M   1.053s    3.545s
#     10k documents            163.2M   4.302s    5.009s
#
# The pool is a net loss across that whole range -- it never came out cheaper
# than serial, and at 1M sentences it died with BrokenProcessPool. The
# threshold below is deliberately far more conservative than those numbers
# would justify: it is set high enough to keep small inputs out of the pool
# while leaving the parallel path in place for genuinely large corpora, rather
# than unilaterally removing it.
#
# The evaluation is the case this matters most for. It preprocesses one column
# of a test set at a time; the biggest bundled column is 144k characters and
# all 19 datasets together are 1.5M, so the whole evaluation now stays serial
# -- it used to spawn 12 pools per `eval_similarity` (~41s) to do ~0.09s of
# tokenization.
PARALLEL_MIN_CHARS = 2_000_000

# spaCy is imported lazily, not at module import time: it costs ~2.2s, it is
# only ever needed by `texts_to_sents`, and every process-pool child that
# imports this module used to pay for it. `_UNSET` distinguishes "not tried
# yet" from the two settled states, so setting `spacy = None` from the outside
# (as the tests do) still means "spaCy is unavailable" and is not overwritten
# by a later import attempt.
_UNSET = object()
spacy = _UNSET
spacy_download = _UNSET


def _import_spacy():
    """
    Import spaCy on first use and cache the result in the module globals.

    Returns the module, or `None` if spaCy is not usable.
    """
    global spacy, spacy_download

    if spacy is _UNSET:
        try:
            import spacy as _spacy
            from spacy.cli import download as _spacy_download
        except Exception as e:
            # deliberately not just `ImportError`: `import spacy` pulls in
            # spacy.cli, which pulls in typer/click, and an incompatible click
            # raises TypeError rather than ImportError. spaCy is optional, so a
            # broken install has to degrade to "spaCy unavailable" instead of
            # breaking `texts_to_sents`'s import.
            logger.debug("spaCy is not usable, disabling it: %r", e)
            _spacy = None
            _spacy_download = None
        spacy, spacy_download = _spacy, _spacy_download

    return spacy


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
    tokenize multiple texts based on whitespaces in parallel

    Falls back to the serial tokenizer for inputs below `PARALLEL_MIN_CHARS`,
    where spawning a process pool costs orders of magnitude more than the
    tokenization itself, and inside a worker process, where a nested pool would
    multiply out (`Corpus.from_text_files(preproc_func=tokenize_texts_parallel)`
    already runs this in a pool of `workers` processes, so nesting would ask for
    `workers**2`). The results are identical either way -- the same
    `tokenize_string` is applied to the same items in the same order -- so both
    are purely scheduling decisions.
    """
    if not hasattr(texts, "__len__"):
        texts = list(texts)

    if multiprocessing.parent_process() is not None:
        return tokenize_texts(texts)

    if sum(len(t) for t in texts) < PARALLEL_MIN_CHARS:
        return tokenize_texts(texts)

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

    if _import_spacy() is None:
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
