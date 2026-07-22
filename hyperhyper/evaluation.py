"""
Evaluate the performance of embeddings with word simularities and word analogies.

Can't use the evaluation methods in gensim because the keyed vector structure does not work for PPMI.
So we have to caculate the metrics ourselves.
"""

import logging
from importlib.resources import files
from pathlib import Path

from scipy.stats import spearmanr

from . import evaluation_datasets

logger = logging.getLogger(__name__)


def _txt_files(directory):
    """
    The `.txt` datasets directly inside `directory`, or `[]` if it is absent.

    `directory` may be an `importlib.resources` `Traversable` (the bundled
    data, possibly inside a zipimport) or a real `pathlib.Path` (a user's
    `data_dir`); both expose `iterdir`/`is_dir`, so the same code walks either.
    """
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.name.endswith(".txt")]


def read_test_data(lang, kind, data_dir=None, include_bundled=True):
    """
    Read the similarity/analogy test datasets for one language and `kind`.

    By default this is the data bundled with the package. Pass `data_dir` to
    also evaluate on your own domain datasets: it is a directory laid out the
    same way as the bundled data -- either ``<data_dir>/<lang>/<kind>/*.txt``
    (mirroring the bundle exactly) or, for a single-language collection, a flat
    ``<data_dir>/<kind>/*.txt``; whichever exists is used. The files use the
    same 3-column (``word1 word2 score``) or 4-column (``a a_ b b_``) format.

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
        datasets.extend(_txt_files(bundled))
    if data_dir is not None:
        root = Path(data_dir)
        # prefer the bundle-mirroring `<data_dir>/<lang>/<kind>`; fall back to a
        # flat `<data_dir>/<kind>` for a single-language collection
        nested = root.joinpath(lang).joinpath(kind)
        custom = nested if nested.is_dir() else root.joinpath(kind)
        datasets.extend(_txt_files(custom))
    return sorted(datasets, key=lambda p: p.name)


def data_name(data):
    """
    the file name of a test dataset, without the `.txt` suffix
    """
    return data.name.removesuffix(".txt")


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


def setup_test_tokens(p, keep_len):
    """
    Read in traning data from files and discard comments (etc.)
    """
    lines = p.read_text(encoding="utf-8").split("\n")
    lines = [line.split() for line in lines]
    lines = [line for line in lines if len(line) == keep_len]
    # every line has exactly `keep_len` fields after the filter above
    return zip(*lines, strict=True)


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


# number of leading word columns per row for each dataset kind: a similarity
# row is `word1 word2 score`, an analogy row is `a a_ b b_`
_WORD_COLUMNS = {"ws": 2, "analogy": 4}


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

    `kind` is ``"ws"`` (2 word columns) or ``"analogy"`` (4). `data_dir` and
    `include_bundled` behave as in `read_test_data`. Returns a list of dicts,
    one per dataset: ``name``, ``kind``, ``rows`` (total), ``covered``
    (in-vocabulary rows) and ``coverage`` (the fraction, `nan` for an empty
    dataset).
    """
    if kind not in _WORD_COLUMNS:
        raise ValueError(f"kind must be one of {sorted(_WORD_COLUMNS)}, got {kind!r}")
    n_word_cols = _WORD_COLUMNS[kind]
    keep_len = 3 if kind == "ws" else 4

    report = []
    for data in read_test_data(
        lang, kind, data_dir=data_dir, include_bundled=include_bundled
    ):
        columns = list(setup_test_tokens(data, keep_len))
        # keep only the word columns (drop the trailing similarity score column)
        # and preprocess each in batch, exactly as the evaluators do
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
