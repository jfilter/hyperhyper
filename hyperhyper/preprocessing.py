"""
simple text preprocessing such as cleaning and tokenization
"""

import logging
import multiprocessing
import re
import sys
import time
import unicodedata

from tqdm import tqdm

from .utils import _default_workers, map_pool

logger = logging.getLogger(__name__)

# --- Inlined gensim.parsing.preprocessing helpers -------------------------
#
# `tokenize_string` used to import `preprocess_string`, `strip_tags` and
# `strip_non_alphanum` from `gensim.parsing.preprocessing`. Those three are
# ~6 lines of regex in total; inlining them pins the tokenizer's *semantics*
# in this repo where they are tested and cannot drift under a gensim upgrade.
# gensim stays a hard dependency (via `Vocab`/`KeyedVectors`) -- this is about
# owning the semantics, not dropping the dependency.
#
# These are byte-for-byte equivalent to gensim's helpers for `str` input:
# gensim compiles both patterns with `re.UNICODE`, and its `utils.to_unicode`
# call is a no-op for `str` (it only decodes `bytes`), and every value flowing
# through this module is already `str`.
_RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
_RE_NONALPHA = re.compile(r"\W", re.UNICODE)


def _strip_tags(s):
    """Remove `<...>` tags (gensim.parsing `strip_tags`, inlined)."""
    return _RE_TAGS.sub("", s)


def _strip_non_alphanum(s):
    """Replace every non-word (`\\W`) char with a space (gensim `strip_non_alphanum`)."""
    return _RE_NONALPHA.sub(" ", s)


def _preprocess_string(s, filters):
    """Apply `filters` in order then whitespace-split (gensim `preprocess_string`)."""
    for f in filters:
        s = f(s)
    return s.split()


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

# The threshold above turned out to be far too low, and the table above says why
# without having drawn the conclusion: the pool is a *net loss at every size that
# was measured*, including 163M characters -- yet 2M lets it run, and at exactly
# 2M the same table records serial 0.075s against pool 3.870s. Every corpus
# larger than a couple of megabytes therefore paid ~3s of spawn startup to make
# tokenization slower.
#
# Re-measured with `tokenize_string_v2` on synthetic Zipfian text (M1 Pro, 10
# workers, Python 3.12):
#
#      6.6M chars   serial 0.42s   pool 2.91s
#     26.2M chars   serial 1.80s   pool 3.52s
#     52.5M chars   serial 3.72s   pool 5.07s
#
# The gap narrows but does not close: extrapolating the two slopes puts
# break-even beyond 100M characters, which for a package aimed at *small,
# domain-specific* corpora is never. The cost is not the tokenizing, it is
# shipping every string to a worker and every token list back.
#
# So the decision is no longer a fixed character count. It is measured, the same
# way `pair_counts._pool_is_worth_starting` measures it: tokenize a sample,
# extrapolate, and only start a pool if it beats serial by a margin. That
# self-calibrates on machines and text this table never saw, and it cannot go on
# being quietly wrong.
#
# Scheduling has no effect on the *result*: `map_pool` preserves order and the
# tokenizer is a pure function, so serial and pooled output are identical.

# What a pool costs before it tokenizes anything -- dominated by each spawned
# child importing the package. Same measurement as `pair_counts`.
POOL_STARTUP_SECONDS = 3.0

# How much faster the pool must look before it is worth starting.
POOL_SPEEDUP_MARGIN = 1.3

# Texts tokenized in-process to estimate the per-text cost: enough to be above
# timer noise, few enough to be a rounding error on any corpus where the answer
# is not already obvious.
PROBE_TEXTS = 2000


def _pool_is_worth_starting(texts, tokenizer):
    """
    Whether tokenizing `texts` across a process pool beats doing it here.

    Measured rather than derived from a character count, because the per-character
    cost is not a property of the corpus: it swings with text shape (many short
    lines cost more per character than few long ones) and with the tokenizer.
    """
    workers = _default_workers()
    if len(texts) < 2 or workers < 2:
        return False

    # Sample with a stride rather than taking the first N: a corpus is very
    # often ordered (by document, by date, by source), so its head is not a fair
    # sample of it. Probing `texts[:N]` of a corpus whose short headlines come
    # first would underestimate the work and refuse a pool that was worth
    # starting -- and the reverse for a corpus that opens with long documents.
    step = max(1, len(texts) // PROBE_TEXTS)
    sample = texts[::step][:PROBE_TEXTS]
    start = time.perf_counter()
    for t in sample:
        tokenizer(t)
    elapsed = time.perf_counter() - start
    if elapsed == 0:
        return False

    serial = elapsed / len(sample) * len(texts)
    # `serial / workers` is the *optimistic* bound: it counts the tokenizing but
    # not the cost of shipping every string to a worker and every token list
    # back, which the measurements above show to be the dominant term for this
    # workload. The estimate is therefore biased towards the pool, and
    # `POOL_SPEEDUP_MARGIN` is what keeps that bias from deciding marginal cases.
    parallel = POOL_STARTUP_SECONDS + serial / min(workers, len(texts))
    logger.debug(
        "tokenizing %d texts: serial ~%.2fs, pool ~%.2fs", len(texts), serial, parallel
    )
    return parallel * POOL_SPEEDUP_MARGIN < serial


def _should_pool(texts, tokenizer):
    """
    The single place that decides serial vs pool for tokenization.

    Three gates, cheapest first:

    1. **Inside a worker** -- never. `Corpus.from_text_files` already runs the
       tokenizer in a pool of `workers` processes, so nesting would ask for
       `workers ** 2`.
    2. **Below `PARALLEL_MIN_CHARS`** -- never, without measuring. This keeps
       the common small case (the evaluation preprocesses one dataset column at
       a time) from paying even for the probe.
    3. **Otherwise, measured** -- see `_pool_is_worth_starting`.

    All three are scheduling only: `map_pool` preserves order and the tokenizer
    is pure, so the tokens are identical whichever way this goes.
    """
    if multiprocessing.parent_process() is not None:
        return False
    if sum(len(t) for t in texts) < PARALLEL_MIN_CHARS:
        return False
    return _pool_is_worth_starting(texts, tokenizer)


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

    NB: existing bunches pickle this function by *reference* (module+qualname),
    so its behaviour must never change -- see `tokenize_string_v2` for the
    fixed tokenizer under a new name. The gensim.parsing helpers it used to call
    are now inlined above (byte-for-byte identical for `str` input).
    """
    CUSTOM_FILTERS = [simple_preproc, _strip_tags, _strip_non_alphanum]
    return _preprocess_string(text, CUSTOM_FILTERS)


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

    if not _should_pool(texts, tokenize_string):
        return tokenize_texts(texts)

    return map_pool(texts, tokenize_string)


# --- v2 tokenizer ---------------------------------------------------------
#
# `tokenize_string_v2` is the fixed tokenizer (ADR 0002). It lives under a NEW
# name on purpose: `Corpus(SaveLoad)` pickles `preproc_fun` by reference, so
# editing `tokenize_string` in place would silently change every existing
# bunch's preprocessing on reopen. The old functions stay forever.
#
# What it fixes relative to v1:
#   1. NFC normalization FIRST -- v1 left NFD 'café' with a combining mark
#      that `\W` deleted, so 'cafe' and 'café' became two vocab entries.
#   2. Typographic canonicalization -- curly apostrophe U+2019 -> ASCII "'", and
#      the Unicode hyphens (U+2010, U+2011, U+2012, U+2013, U+2014) -> ASCII "-".
#   3. Extraction, not destruction: `\w+(?:['\-]\w+)*` keeps `city's`,
#      `ice-cream`, `don't` whole and splits on everything else. v1's
#      `\W`-substitution shattered them into `city`/`s`, `ice`/`cream`.
#   4. Digit->"0" is now a PARAMETER (`normalize_digits`), default False: keep
#      digits, since the audience is small domain corpora where numerals carry
#      meaning. `normalize_digits=True` restores the legacy behaviour, which is
#      the Levy-Goldberg-Dagan / hyperwords convention this package reimplements.

# U+2019 RIGHT SINGLE QUOTATION MARK (used as a typographic apostrophe) is
# folded to ASCII "'" so the extraction pattern's clitic joiner covers curly
# apostrophes too. Written as an escape so the ambiguous glyph never appears
# in the source literally.
_CURLY_APOSTROPHE = "\u2019"

# The Unicode hyphen/dash characters that a `-` in the joiner set should treat
# as a hyphen. Written as escapes (the glyphs are visually ambiguous with an
# ASCII hyphen):
#   U+2010 HYPHEN, U+2011 NON-BREAKING HYPHEN, U+2012 FIGURE DASH,
#   U+2013 EN DASH, U+2014 EM DASH.
_UNICODE_HYPHENS = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
)

# Extraction pattern: a run of word chars, optionally continued across a single
# apostrophe or hyphen into more word chars. Keeps `city's`, `ice-cream`,
# `don't` whole; splits on everything else. `.` is deliberately NOT in the
# joiner set -- adding it would glue `word.Next` together in scraped text with
# missing spaces, a worse trade than losing `U.S.A.`.
_WORD_RE = re.compile(r"\w+(?:['\-]\w+)*", re.UNICODE)


def tokenize_string_v2(text, lower=True, normalize_digits=False):
    """
    Fixed whitespace/regex tokenizer (ADR 0002); see the module comment above.

    Args:
        text (str): the text to tokenize.
        lower (bool): lowercase before extraction (default True).
        normalize_digits (bool): replace every digit with "0" (default False --
            digits are kept). True is the Levy-Goldberg-Dagan / hyperwords
            convention this package reimplements.

    Returns:
        list[str]: the extracted tokens.
    """
    # NFC first: composes 'café' (NFD) back into 'café' so the accent
    # is a word char instead of a `\W` combining mark that gets dropped.
    text = unicodedata.normalize("NFC", text)
    # canonicalize typographic variants before extraction so the joiner set
    # ("'" and "-") covers curly apostrophes and Unicode hyphens too
    text = text.replace(_CURLY_APOSTROPHE, "'").translate(_UNICODE_HYPHENS)
    if lower:
        text = text.lower()
    if normalize_digits:
        text = re.sub(r"\d", "0", text)
    return _WORD_RE.findall(text)


def tokenize_texts_v2(texts, lower=True, normalize_digits=False):
    """
    Serial v2 tokenizer over a list of texts (see `tokenize_string_v2`).
    """
    return [
        tokenize_string_v2(t, lower=lower, normalize_digits=normalize_digits)
        for t in texts
    ]


def tokenize_texts_parallel_v2(texts):
    """
    Parallel v2 tokenizer -- the new default `preproc_func` for the constructors.

    Mirrors `tokenize_texts_parallel`'s scheduling exactly (same threshold, same
    "no nested pool inside a worker" guard); only the per-item tokenizer differs.
    It uses `tokenize_string_v2`'s defaults (`lower=True`, `normalize_digits=
    False`). A caller who wants non-default v2 options passes an explicit
    top-level `preproc_func` (e.g. `functools.partial` is *not* picklable-by-
    reference; define a small module-level wrapper instead -- see docs/usage.md).
    """
    if not hasattr(texts, "__len__"):
        texts = list(texts)

    if not _should_pool(texts, tokenize_string_v2):
        return tokenize_texts_v2(texts)

    return map_pool(texts, tokenize_string_v2)


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
    texts = [_strip_tags(t) for t in texts]
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
                        _strip_non_alphanum(t.lemma_ if lemmatize else t.text)
                    )
                    for t in s
                    if not any((t.is_punct, t.is_space, remove_stop and t.is_stop))
                ]
            )
    return results
