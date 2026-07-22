"""
Evaluate the performance of embeddings with word simularities and word analogies.

Can't use the evaluation methods in gensim because the keyed vector structure does not work for PPMI.
So we have to caculate the metrics ourselves.
"""

import csv
import logging
import math
from collections import Counter
from importlib.resources import files
from pathlib import Path

from scipy.stats import spearmanr

from . import evaluation_datasets

logger = logging.getLogger(__name__)

# The datasets ship as strict UTF-8 TSV (`.tsv`); the legacy whitespace `.txt`
# format is still read for users' existing files. A file's format is chosen by
# its extension, never by sniffing the delimiter (ADR 0002).
_DATASET_SUFFIXES = (".tsv", ".txt")

# The required TSV header row, keyed by the number of columns `setup_test_tokens`
# keeps for each dataset kind: a similarity row is `word1 word2 score` (3), an
# analogy row is `a a_prime b b_prime` (4). These names are the on-disk contract.
_TSV_HEADERS = {
    3: ("word1", "word2", "score"),
    4: ("a", "a_prime", "b", "b_prime"),
}

# The domain proxy tasks (ADR 0001, P4) are *variable width*: a synonym
# multiple-choice row carries as many distractors as its file declares. Their
# width therefore cannot be passed in the way `ws`/`analogy` pass a fixed
# `keep_len` -- it is read off the file's own header, which is exactly the
# capability the TSV migration bought (ADR 0002 anticipated this: "fits the
# richer rows the P4 domain-proxy datasets will need").
#
# Keying the header by column count, as the two original kinds do, also stops
# working here: a synonym row with one distractor would be three columns and
# collide with `ws`. So these kinds are keyed by *name* instead, and a caller
# reading one must say which kind it wants.
_SYNONYM_MIN_DISTRACTORS = 2
_CATEGORY_HEADER = ("word", "category")


def _synonym_header(n_columns):
    """The required synonym-MC header for a file of `n_columns` columns."""
    return ("target", "answer", *(f"distractor{i}" for i in range(1, n_columns - 1)))


class MalformedDatasetError(ValueError):
    """
    A `.tsv` dataset is structurally invalid.

    Strict parsing is a main reason for the TSV migration (ADR 0002): a bad
    header, a wrong column count, an empty required field or a non-finite
    similarity score *raises* -- naming the file and 1-based line number -- rather
    than being silently dropped the way the legacy whitespace parser skips it.
    """


class DuplicateDatasetError(ValueError):
    """
    A directory holds the same dataset as both `foo.tsv` and `foo.txt`.

    Scoring both would double-count the dataset (and its format would be
    ambiguous), so discovery refuses rather than guessing which one is canonical.
    """


def _dataset_stem(name):
    """The dataset name without its `.tsv`/`.txt` suffix."""
    for suffix in _DATASET_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _dataset_files(directory):
    """
    The `.tsv`/`.txt` datasets directly inside `directory`, or `[]` if absent.

    `directory` may be an `importlib.resources` `Traversable` (the bundled
    data, possibly inside a zipimport) or a real `pathlib.Path` (a user's
    `data_dir`); both expose `iterdir`/`is_dir`, so the same code walks either.

    A directory that holds the same dataset in both formats (`foo.tsv` and
    `foo.txt`) is rejected with `DuplicateDatasetError` rather than scoring both.
    """
    if not directory.is_dir():
        return []
    found = [p for p in directory.iterdir() if p.name.endswith(_DATASET_SUFFIXES)]
    by_stem = {}
    for p in found:
        by_stem.setdefault(_dataset_stem(p.name), []).append(p.name)
    collisions = {
        stem: sorted(names) for stem, names in by_stem.items() if len(names) > 1
    }
    if collisions:
        detail = "; ".join(
            f"{stem}: {names}" for stem, names in sorted(collisions.items())
        )
        raise DuplicateDatasetError(
            f"duplicate dataset(s) present as both .tsv and .txt in {directory}: "
            f"{detail}. Keep exactly one format per dataset."
        )
    return found


def read_test_data(lang, kind, data_dir=None, include_bundled=True):
    """
    Read the similarity/analogy test datasets for one language and `kind`.

    By default this is the data bundled with the package. Pass `data_dir` to
    also evaluate on your own domain datasets: it is a directory laid out the
    same way as the bundled data -- either ``<data_dir>/<lang>/<kind>/*.tsv``
    (mirroring the bundle exactly) or, for a single-language collection, a flat
    ``<data_dir>/<kind>/*.tsv``; whichever exists is used. The bundled files are
    strict UTF-8 TSV (header row, ``# key: value`` preamble); a legacy
    whitespace ``.txt`` in the same layout is still read via the compatibility
    path (chosen by extension, never by delimiter sniffing).

    `include_bundled=False` evaluates *only* `data_dir`, replacing the bundled
    sets instead of adding to them; the default (`True`) evaluates the user's
    sets *alongside* the bundled ones, so existing behaviour is unchanged when
    `data_dir` is not given.

    Bundled entries are returned as `Traversable`s rather than filesystem
    paths: extracting the directory to a temporary location (which is what
    `as_file` does for a zipimported package) and returning the paths
    afterwards hands back names that no longer exist by the time the caller
    reads them. A `data_dir` on a real filesystem yields ordinary
    `pathlib.Path`s, which the callers' `read_text`/`.name` usage handles the
    same way.
    """
    datasets = []
    if include_bundled:
        bundled = files(evaluation_datasets).joinpath(lang).joinpath(kind)
        datasets.extend(_dataset_files(bundled))
    if data_dir is not None:
        root = Path(data_dir)
        # prefer the bundle-mirroring `<data_dir>/<lang>/<kind>`; fall back to a
        # flat `<data_dir>/<kind>` for a single-language collection
        nested = root.joinpath(lang).joinpath(kind)
        custom = nested if nested.is_dir() else root.joinpath(kind)
        datasets.extend(_dataset_files(custom))
    return sorted(datasets, key=lambda p: p.name)


def data_name(data):
    """
    the file name of a test dataset, without the `.tsv`/`.txt` suffix
    """
    return _dataset_stem(data.name)


def to_item(li):
    """
    Squeeze a preprocessed test-set entry down to the single token it stands for.

    Returns `None` when there is no such token, which happens two ways:

    * the preprocessing dropped the entry entirely (stop word, punctuation), or
    * it produced *several* tokens -- multi-word entries such as `vice
      president`, and every hyphenated form (`ice-cream` -> `['ice', 'cream']`).

    The multi-token case used to return the first token. That silently scored a
    different word than the dataset asked about: `ice-cream` was evaluated as
    `ice`. Such a row cannot be answered, so it has to be dropped and counted
    as out-of-vocabulary instead.
    """
    if isinstance(li, list):
        if len(li) != 1:
            return None
        return to_item(li[0])
    return li


def penalize_oov(score, oov):
    """
    Fold the out-of-vocabulary rate into a score.

    Scaling by `1 - oov` only penalizes non-negative scores. Spearman is
    genuinely negative for an embedding that is worse than chance, and there
    scaling *rewards* missing data: -0.5 at oov=0.0 gives -0.500, but -0.5 at
    oov=0.9 gives -0.050, i.e. an almost-perfect-looking score for an embedding
    that got the sign wrong and could only answer a tenth of the questions.

    Subtracting the magnitude instead applies the same downward penalty
    whatever the sign, so more missing vocabulary always lowers the number.
    For non-negative scores -- analogy accuracy and positive correlations,
    which is every score anyone actually reports -- this is exactly the old
    `score * (1 - oov)`. Only the negative branch changes, and it now ranges
    down to -2 rather than being squeezed towards 0.
    """
    return score - abs(score) * oov


def aggregate(full_results, kind):
    """
    Combine the per-dataset scores into a micro (item-weighted) and a macro
    (dataset-weighted) average.

    Aggregation is de-duplicated across datasets (mechanism (b) of ADR 0001).
    The bundled similarity sets are not disjoint: ``ws353_similarity`` (203
    pairs) and ``ws353_relatedness`` (251) are *complete subsets* of ``ws353``
    (351), and the sim/rel split itself shares 103 pairs; German ``schm280``
    overlaps ``ws353rel`` by 126 pairs; ``en/google`` and ``en/msr`` analogies
    share 106 quadruples. The old micro-average weighted every dataset by its
    in-vocabulary row count and summed those raw counts, so each shared item
    was folded into the pool two or three times.

    To count every item exactly once, each dataset carries a private
    ``_micro_weight``: the number of its scored items (unordered word pairs for
    similarity, ordered quadruples for analogy) that no *earlier* dataset in the
    read order already contributed. Datasets are processed in the sorted read
    order and greedily claim their items; a dataset whose items were all already
    claimed (a redundant parent or subset -- e.g. ``ws353`` once its sim/rel
    split has been counted, or vice versa depending on order) gets weight 0 and
    is dropped from BOTH micro and macro. It is still scored and still appears
    in ``results`` individually, because users and papers report ws353,
    ws353_similarity and ws353_relatedness separately.

    Effect on the reported numbers: micro is now a unique-item-weighted mean, so
    the English word-similarity micro no longer triple-counts the WordSim353
    pairs; macro is a mean over the non-redundant datasets only. Both numbers
    move relative to the old (buggy) aggregation -- see the CHANGELOG. Per-dataset
    ``score``/``oov``/``fullscore`` are unchanged.

    Every dataset can also be skipped for lack of in-vocabulary rows -- the
    normal case for the small, domain-specific corpora this package targets.
    Averaging nothing used to raise `ZeroDivisionError`, so report `nan` and say
    why instead. (A dataset that had in-vocabulary rows always contributes at
    least one item unless everything it holds was already counted, so an empty
    pool means an empty `results`.)
    """
    pooled = [r for r in full_results if r["_micro_weight"] > 0]
    if len(pooled) == 0:
        for r in full_results:
            r.pop("_micro_weight", None)
        logger.warning(
            "no %s dataset had any in-vocabulary row; the vocabulary and the "
            "test data do not overlap, so there is nothing to average",
            kind,
        )
        return {"micro": float("nan"), "macro": float("nan"), "results": full_results}

    weights = [r["_micro_weight"] for r in pooled]
    scores = [r["score"] for r in pooled]
    micro_avg = sum(w * s for w, s in zip(weights, scores, strict=True)) / sum(weights)
    macro_avg = sum(scores) / len(scores)

    # the weight is an internal aggregation detail, not part of the public result
    for r in full_results:
        r.pop("_micro_weight", None)
    return {"micro": micro_avg, "macro": macro_avg, "results": full_results}


def setup_test_tokens(p, keep_len=None, *, kind=None):
    """
    Read a dataset file into its word/score columns.

    The on-disk format is chosen by the file's extension, never by sniffing its
    delimiter (ADR 0002):

    * ``.tsv`` -- the current strict, standard-library-``csv`` TSV format. A
      ``# key: value`` preamble is skipped, the header row is validated against
      `keep_len` (``word1 word2 score`` for 3, ``a a_prime b b_prime`` for 4),
      and every subsequent non-blank record must have exactly `keep_len` columns.
      A bad header, wrong column count, empty required field or non-finite
      similarity score *raises* `MalformedDatasetError` with the file and 1-based
      line number, rather than being silently dropped.
    * ``.txt`` -- the legacy whitespace format, kept for users' existing files.
      Blank and comment lines (leading ``#`` or a ``:`` word2vec section header)
      are skipped; a malformed non-comment row is reported with a
      `logger.warning` (file, line, content) and skipped so one bad row does not
      abort the evaluation.

    The fixed-width kinds are read by passing `keep_len` (3 or 4), which is how
    every existing caller uses this. The variable-width P4 kinds are read by
    passing ``kind="synonym"`` or ``kind="category"`` instead, and their width
    comes from the file's own header -- a synonym file declares how many
    distractors it carries. Legacy ``.txt`` cannot express those kinds (it has no
    header to declare a width), so they are TSV-only.

    Returns the kept columns as `zip(*rows, strict=True)` for both formats; the
    per-field/per-pair scoring downstream is unchanged.
    """
    if kind is not None and keep_len is not None:
        raise ValueError("pass either keep_len or kind, not both")
    if kind is None and keep_len is None:
        raise ValueError("pass keep_len (fixed-width kinds) or kind")
    if p.name.endswith(".tsv"):
        return _read_tsv(p, keep_len, kind)
    if kind is not None:
        raise MalformedDatasetError(
            f"{p.name}: the {kind!r} task is TSV-only -- the legacy whitespace "
            f"format has no header row to declare the row width"
        )
    return _read_txt_whitespace(p, keep_len)


def _resolve_header(p, header, header_idx, keep_len, kind):
    """
    Validate the file's header row and return `(expected, keep_len, score_idx)`.

    For the fixed-width kinds the caller states the width and the header must
    match it exactly. For the variable-width P4 kinds it is the other way round:
    the header *is* the declaration, so it determines the width every data row
    must then have.
    """
    if kind is None:
        expected = _TSV_HEADERS[keep_len]
    elif kind == "category":
        expected, keep_len = _CATEGORY_HEADER, len(_CATEGORY_HEADER)
    elif kind == "synonym":
        keep_len = len(header)
        if keep_len < _SYNONYM_MIN_DISTRACTORS + 2:
            raise MalformedDatasetError(
                f"{p.name}:{header_idx + 1}: a synonym multiple-choice file needs "
                f"a target, an answer and at least {_SYNONYM_MIN_DISTRACTORS} "
                f"distractors ({_SYNONYM_MIN_DISTRACTORS + 2} columns); "
                f"got {keep_len}: {header!r}"
            )
        expected = _synonym_header(keep_len)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    if tuple(header) != expected:
        raise MalformedDatasetError(
            f"{p.name}:{header_idx + 1}: header must be {list(expected)!r}, "
            f"got {header!r}"
        )
    score_idx = expected.index("score") if "score" in expected else None
    return expected, keep_len, score_idx


def _read_tsv(p, keep_len, kind=None):
    """Strict TSV parse (see `setup_test_tokens`); raises on any malformed input."""
    lines = p.read_text(encoding="utf-8").split("\n")

    # the header is the first line that is neither blank nor a `#`-preamble line;
    # metadata/comments are allowed *only* in that preamble, before the header
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header_idx = i
        break
    if header_idx is None:
        raise MalformedDatasetError(
            f"{p.name}: no header row found after the optional '# key: value' preamble"
        )

    header = next(csv.reader([lines[header_idx]], delimiter="\t"))
    _expected, keep_len, score_idx = _resolve_header(
        p, header, header_idx, keep_len, kind
    )

    kept = []
    for lineno, line in enumerate(lines[header_idx + 1 :], header_idx + 2):
        if not line.strip():
            # blank records are allowed and skipped; there is no comment
            # convention after the header -- a stray `#` line is a malformed row
            continue
        fields = next(csv.reader([line], delimiter="\t"))
        if len(fields) != keep_len:
            raise MalformedDatasetError(
                f"{p.name}:{lineno}: expected {keep_len} tab-separated column(s), "
                f"got {len(fields)}: {line!r}"
            )
        if any(field == "" for field in fields):
            raise MalformedDatasetError(
                f"{p.name}:{lineno}: empty required field: {line!r}"
            )
        if score_idx is not None:
            raw = fields[score_idx]
            try:
                value = float(raw)
            except ValueError:
                raise MalformedDatasetError(
                    f"{p.name}:{lineno}: score {raw!r} is not a finite number"
                ) from None
            if not math.isfinite(value):
                raise MalformedDatasetError(
                    f"{p.name}:{lineno}: score {raw!r} is not finite"
                )
        kept.append(fields)
    # every kept row has exactly `keep_len` columns after the checks above
    return zip(*kept, strict=True)


def _read_txt_whitespace(p, keep_len):
    """Legacy whitespace parse (see `setup_test_tokens`); warns-and-skips bad rows."""
    kept = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        stripped = line.strip()
        if not stripped:
            # blank lines are normal padding and skipped silently
            continue
        if stripped[0] in "#:":
            # a real comment / section-header line, never a data row
            continue
        fields = line.split()
        if len(fields) != keep_len:
            logger.warning(
                "%s:%d: skipping malformed line -- %d field(s), expected %d: %r",
                p.name,
                lineno,
                len(fields),
                keep_len,
                line,
            )
            continue
        kept.append(fields)
    # every kept row has exactly `keep_len` fields after the checks above
    return zip(*kept, strict=True)


def eval_similarity(
    vectors, token2id, preproc_fun, lang="en", data_dir=None, include_bundled=True
):
    """
    Evaluate word similarity on several test datasets.

    Pass `data_dir` (with optional `include_bundled`) to evaluate on your own
    domain datasets; see `read_test_data`. The micro/macro averages count every
    unique word pair once even when datasets overlap; see `aggregate`.
    """
    full_results = []
    # unordered word pairs already counted in the micro/macro pool, so a pair
    # shared between datasets (ws353 vs its sim/rel split, etc.) is not weighted
    # more than once
    seen = set()

    for data in read_test_data(
        lang, "ws", data_dir=data_dir, include_bundled=include_bundled
    ):
        results = []
        pair_keys = []

        token1, token2, sims = setup_test_tokens(data, 3)
        # The gold column is text on disk and has to be cast before it reaches
        # `spearmanr`, which column-stacks its two arguments: a float array
        # stacked with a string array promotes *everything* to strings, so both
        # columns end up ranked lexicographically ("10" < "2.5" < "9") and even
        # a perfect embedding scores well below 1.0.
        sims = [float(s) for s in sims]
        # preprocess tokens 'in batch'
        token1, token2 = preproc_fun(token1), preproc_fun(token2)
        # strict=False: preproc_fun is not guaranteed to be length-preserving
        # (texts_to_sents emits one entry per sentence, not per input text)
        lines = list(zip(token1, token2, sims, strict=False))
        for x, y, sim in lines:
            x, y = to_item(x), to_item(y)

            # skip over OOV
            if x in token2id and y in token2id:
                results.append((vectors.similarity(token2id[x], token2id[y]), sim))
                # unordered: "car train" and "train car" are the same pair
                pair_keys.append(frozenset((token2id[x], token2id[y])))

        if len(results) == 0:
            logger.warning("not enough results for this dataset: %s", data.name)
            continue

        actual, expected = zip(*results, strict=True)
        spear_res = spearmanr(actual, expected).statistic
        oov = (len(lines) - len(results)) / len(lines)

        # weight this dataset by the pairs it adds that no earlier dataset held
        micro_weight = len(set(pair_keys) - seen)
        seen.update(pair_keys)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": spear_res,
                "oov": oov,
                # consider the portion of OOV
                "fullscore": penalize_oov(spear_res, oov),
                "_micro_weight": micro_weight,
            }
        )

    return aggregate(full_results, "word similarity")


# analogies
def eval_analogies(
    vectors,
    token2id,
    preproc_fun,
    lang="en",
    objective="add",
    data_dir=None,
    include_bundled=True,
):
    """
    Evaluate word-analogy accuracy on the bundled (and/or user) datasets.

    ``objective`` selects the analogy recovery objective handed to
    ``most_similar_vectors``: ``"add"`` (3CosAdd, the default -- unchanged
    behaviour and the metric the recorded results were computed with) or
    ``"mul"`` (3CosMul, Levy & Goldberg 2014). The exclusion set, OOV handling
    and unanswerable-row skipping are identical for both objectives.

    Pass `data_dir` (with optional `include_bundled`) to evaluate on your own
    datasets; see `read_test_data`. The micro/macro averages count every unique
    quadruple once even when datasets overlap (en/google and en/msr share 106);
    see `aggregate`.
    """
    full_results = []
    # answered quadruples already counted in the pool, so a quadruple shared
    # between datasets is not weighted more than once
    seen = set()

    for data in read_test_data(
        lang, "analogy", data_dir=data_dir, include_bundled=include_bundled
    ):
        results = []
        quad_keys = []

        line_tokens = setup_test_tokens(data, 4)
        line_tokens = [preproc_fun(t) for t in line_tokens]
        # strict=False: see the note in eval_similarity
        lines = list(zip(*line_tokens, strict=False))
        for tokens in lines:
            tokens = [to_item(x) for x in tokens]
            # skip over OOV
            if not all(x in token2id for x in tokens):
                continue

            tokens = [token2id[x] for x in tokens]
            # the dataset columns are `a a_ b b_`, i.e. the relation a -> a_ is
            # mirrored by b -> b_. 3CosAdd thus asks for a_ - a + b.
            a, a_, b, b_ = tokens
            exclusions = {a, a_, b}
            # A guess only counts if it is outside `exclusions`, so a row whose
            # expected answer is itself one of the question words can never
            # score 1. Scoring it 0 puts an unanswerable row in the accuracy
            # denominator and caps the reported number invisibly -- under the
            # default lemmatizing preprocessing `write writes work works`
            # collapses to `write write work work`, which is 31% of
            # en/analogy/google.txt and 80% of en/analogy/msr.txt. Drop the row
            # so it lands in the honest `oov` bucket instead.
            if b_ in exclusions:
                continue
            guesses = vectors.most_similar_vectors(
                [a_, b], [a], topn=len(exclusions) + 1, objective=objective
            )
            guesses = [int(idx) for idx, _ in guesses if int(idx) not in exclusions]
            result = 1 if guesses and guesses[0] == b_ else 0
            results.append(result)
            # ordered: the relation a->a_ :: b->b_ is direction-specific
            quad_keys.append((a, a_, b, b_))

        if len(results) == 0:
            logger.warning("not enough results for this dataset: %s", data.name)
            continue

        accuracy = sum(results) / len(results)
        oov = (len(lines) - len(results)) / len(lines)

        # weight this dataset by the quadruples no earlier dataset already held
        micro_weight = len(set(quad_keys) - seen)
        seen.update(quad_keys)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": accuracy,
                "oov": oov,
                # consider the portion of OOV; accuracy is never negative, so
                # this matches the old `accuracy * (1 - oov)` exactly
                "fullscore": penalize_oov(accuracy, oov),
                "_micro_weight": micro_weight,
            }
        )

    return aggregate(full_results, "analogy")


# ---------------------------------------------------------------------------
# domain proxy tasks (ADR 0001, P4)
#
# Both tasks below exist because the bundled general-language similarity and
# analogy sets do not serve this package's stated audience: on a small,
# domain-specific corpus they are largely out-of-vocabulary, so they measure
# almost nothing. The alternative is not to *rate* domain word pairs -- that
# needs human judgement, which is the very thing ADR 0001 forbids generating --
# but to build tasks whose gold answer is a **membership fact** a domain already
# records. Verification is then a lookup, not an opinion:
#
# * synonym multiple choice: gold comes from a thesaurus/glossary entry;
# * category purity: gold comes from a taxonomy's class assignment.
#
# Neither is bundled, and that is deliberate: the useful version of each is
# built from *the user's own* glossary, so they live behind `data_dir`. See
# `tools/build_domain_tasks/`.
# ---------------------------------------------------------------------------


def eval_synonyms(
    vectors, token2id, preproc_fun, lang="en", data_dir=None, include_bundled=True
):
    """
    Evaluate synonym multiple choice: is the true synonym the nearest candidate?

    Each row is ``target answer distractor1 ... distractorK``. The model scores
    the cosine between the target and every candidate, and the row counts as
    correct when the answer wins outright. This is the TOEFL-style task, and its
    appeal for domain evaluation is that the gold answer is a fact a glossary
    already records rather than a rating someone had to produce.

    Two deliberate strictnesses:

    * **A tie does not count as correct.** The answer must beat every
      distractor, not merely match the best of them. A model that assigns the
      identical score to several candidates -- which happens readily on a sparse
      PPMI matrix where a rare target shares no context with any candidate, so
      every cosine is 0 -- has not chosen anything, and scoring that as a hit
      would report a chance artefact as competence.
    * **A row is scored only if the target and *every* candidate are
      in-vocabulary.** Dropping just the missing distractors would quietly make
      the question easier for exactly those rows the corpus covers worst, so a
      thinner vocabulary would inflate the score. Incomplete rows go to `oov`.

    Note that accuracy here has a chance floor of ``1/(K+1)``, unlike the
    similarity and analogy scores; compare against that floor, not against 0.

    Pass `data_dir` (with optional `include_bundled`) to evaluate on your own
    datasets; see `read_test_data`.
    """
    full_results = []
    # rows already counted in the micro/macro pool, so a question shared between
    # datasets is not weighted more than once
    seen = set()

    for data in read_test_data(
        lang, "synonym", data_dir=data_dir, include_bundled=include_bundled
    ):
        results = []
        row_keys = []

        columns = list(setup_test_tokens(data, kind="synonym"))
        columns = [preproc_fun(col) for col in columns]
        # strict=False: see the note in eval_similarity
        rows = list(zip(*columns, strict=False))
        for row in rows:
            row = [to_item(entry) for entry in row]
            if not all(entry in token2id for entry in row):
                continue
            target, *candidates = [token2id[entry] for entry in row]
            # a candidate identical to the target would win on cosine 1 by
            # construction and says nothing about the embedding
            if target in candidates or len(set(candidates)) != len(candidates):
                continue
            scores = [vectors.similarity(target, c) for c in candidates]
            answer_score, distractor_scores = scores[0], scores[1:]
            results.append(1 if answer_score > max(distractor_scores) else 0)
            row_keys.append((target, tuple(candidates)))

        if len(results) == 0:
            logger.warning("not enough results for this dataset: %s", data.name)
            continue

        accuracy = sum(results) / len(results)
        oov = (len(rows) - len(results)) / len(rows)

        micro_weight = len(set(row_keys) - seen)
        seen.update(row_keys)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": accuracy,
                "oov": oov,
                "fullscore": penalize_oov(accuracy, oov),
                "_micro_weight": micro_weight,
            }
        )

    return aggregate(full_results, "synonym choice")


def eval_categories(
    vectors, token2id, preproc_fun, lang="en", data_dir=None, include_bundled=True
):
    """
    Evaluate category purity: does a word's nearest neighbour share its class?

    Each row is ``word category``. For every in-vocabulary word, the nearest
    other in-vocabulary word *of the dataset* is found by cosine, and the score
    is the fraction whose nearest neighbour carries the same category label.
    Gold is again a membership fact -- a taxonomy assignment -- not a judgement.

    Nearest-neighbour purity is used rather than running a clustering algorithm
    on purpose: k-means would add an initialisation seed, an iteration count and
    a cluster-count choice, all of which move the number without telling you
    anything about the embedding. This metric is deterministic given the vectors.

    The neighbour search is restricted to the dataset's own words rather than the
    whole vocabulary. That keeps the number interpretable -- it asks whether the
    categories are locally separated *from each other*, which is the question a
    domain taxonomy poses -- and it makes the chance floor computable, which is
    the next point.

    **The chance floor is not 0 and it moves with the data**, so the result
    carries a `baseline` alongside `score`: the purity a random neighbour
    assignment would give for these category sizes,
    ``sum(n_c * (n_c - 1)) / (n * (n - 1))``. One dominant category can push that
    floor high, and a score below it means the embedding is worse than chance.
    Reporting purity alone would hide that.

    A dataset needs at least two categories and two scoreable words, otherwise
    purity is trivially 1 and meaningless; such a dataset is skipped with a
    warning.

    Pass `data_dir` (with optional `include_bundled`) to evaluate on your own
    datasets; see `read_test_data`.
    """
    full_results = []
    seen = set()

    for data in read_test_data(
        lang, "category", data_dir=data_dir, include_bundled=include_bundled
    ):
        words, categories = setup_test_tokens(data, kind="category")
        # only the word column is preprocessed; the category is a label, not a
        # word to be looked up
        words = preproc_fun(list(words))
        # strict=False: see the note in eval_similarity
        rows = list(zip(words, categories, strict=False))

        entries = []
        for word, category in rows:
            token = to_item(word)
            if token in token2id:
                entries.append((token2id[token], category))
        # a word appearing twice would be its own nearest neighbour
        entries = list({idx: (idx, cat) for idx, cat in entries}.values())

        n = len(entries)
        distinct = {cat for _idx, cat in entries}
        if n < 2 or len(distinct) < 2:
            logger.warning(
                "skipping %s: needs >=2 in-vocabulary words in >=2 categories, "
                "got %d word(s) in %d categor(y/ies)",
                data.name,
                n,
                len(distinct),
            )
            continue

        hits = 0
        for i, (idx, category) in enumerate(entries):
            best_score, best_category = None, None
            for j, (other, other_category) in enumerate(entries):
                if i == j:
                    continue
                score = vectors.similarity(idx, other)
                if best_score is None or score > best_score:
                    best_score, best_category = score, other_category
            hits += 1 if best_category == category else 0
        purity = hits / n

        sizes = Counter(cat for _idx, cat in entries)
        # probability that a uniformly random *other* word shares the category
        baseline = sum(s * (s - 1) for s in sizes.values()) / (n * (n - 1))

        oov = (len(rows) - n) / len(rows) if rows else 0.0

        # the scored word set, so the same taxonomy in two datasets is not
        # weighted twice
        word_keys = {idx for idx, _cat in entries}
        micro_weight = len(word_keys - seen)
        seen.update(word_keys)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": purity,
                "baseline": baseline,
                "oov": oov,
                "fullscore": penalize_oov(purity, oov),
                "_micro_weight": micro_weight,
            }
        )

    return aggregate(full_results, "category purity")


# Number of leading word columns per row for each dataset kind: a similarity row
# is `word1 word2 score`, an analogy row is `a a_ b b_`, a category row is
# `word category` (only the first column is a word to look up). `None` means
# *every* column is a word -- a synonym row is all target and candidates, and how
# many of those there are is declared by the file's own header.
_WORD_COLUMNS = {"ws": 2, "analogy": 4, "category": 1, "synonym": None}

# the fixed row width of each fixed-width kind; the P4 kinds are read by name
_KEEP_LEN = {"ws": 3, "analogy": 4}


def dataset_coverage(
    token2id, preproc_fun, lang="en", kind="ws", data_dir=None, include_bundled=True
):
    """
    Report, per dataset, the fraction of rows fully in-vocabulary.

    A user can run this *before* training to learn which bundled or custom test
    sets their corpus vocabulary can actually be scored on -- the small,
    domain-specific corpora this package targets often share little vocabulary
    with the general-language benchmarks.

    Coverage is computed under the *same* preprocessing and single-token
    reduction the evaluator uses (`preproc_fun` then `to_item`), so the number
    matches what evaluation will see: a row counts as covered only if every one
    of its words survives preprocessing to a single in-vocabulary token. A
    hyphenated or multi-word entry (which `to_item` drops) therefore counts as
    not-covered, exactly as the evaluator treats it as OOV.

    `kind` is ``"ws"`` (2 word columns), ``"analogy"`` (4), ``"synonym"`` (every
    column: target and all candidates) or ``"category"`` (1; the label is not
    looked up). `data_dir` and `include_bundled` behave as in `read_test_data`.
    Returns a list of dicts, one per dataset: ``name``, ``kind``, ``rows``
    (total), ``covered`` (in-vocabulary rows) and ``coverage`` (the fraction,
    `nan` for an empty dataset).

    Coverage is the *first* thing to run for the P4 domain tasks: a
    glossary-built synonym set whose coverage is near zero cannot be scored at
    all, and knowing that before training is the whole point.
    """
    if kind not in _WORD_COLUMNS:
        raise ValueError(f"kind must be one of {sorted(_WORD_COLUMNS)}, got {kind!r}")
    n_word_cols = _WORD_COLUMNS[kind]
    keep_len = _KEEP_LEN.get(kind)

    report = []
    for data in read_test_data(
        lang, kind, data_dir=data_dir, include_bundled=include_bundled
    ):
        if keep_len is None:
            columns = list(setup_test_tokens(data, kind=kind))
        else:
            columns = list(setup_test_tokens(data, keep_len))
        # keep only the word columns (dropping the similarity score / the
        # category label) and preprocess each in batch, exactly as the
        # evaluators do; `None` means every column is a word
        word_columns = [preproc_fun(col) for col in columns[:n_word_cols]]
        # strict=False: preproc_fun need not be length-preserving (see the note
        # in eval_similarity)
        rows = list(zip(*word_columns, strict=False))

        covered = sum(
            1 for row in rows if all(to_item(token) in token2id for token in row)
        )
        n_rows = len(rows)
        report.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "kind": kind,
                "rows": n_rows,
                "covered": covered,
                "coverage": covered / n_rows if n_rows else float("nan"),
            }
        )
    return report
