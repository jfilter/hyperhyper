"""
Evaluate the performance of embeddings with word simularities and word analogies.

Can't use the evaluation methods in gensim because the keyed vector structure does not work for PPMI.
So we have to caculate the metrics ourselves.
"""

import logging
from importlib.resources import files

from scipy.stats import spearmanr

from . import evaluation_datasets

logger = logging.getLogger(__name__)


def read_test_data(lang, kind):
    """
    read test data that is stored within the module

    Returns `Traversable`s rather than filesystem paths: extracting the
    directory to a temporary location (which is what `as_file` does for a
    zipimported package) and returning the paths afterwards hands back names
    that no longer exist by the time the caller reads them.
    """
    directory = files(evaluation_datasets).joinpath(lang).joinpath(kind)
    return sorted(
        (p for p in directory.iterdir() if p.name.endswith(".txt")),
        key=lambda p: p.name,
    )


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


def aggregate(line_counts, scores, full_results, kind):
    """
    Combine the per-dataset scores into a micro (line-weighted) and a macro
    (dataset-weighted) average.

    Every dataset can be skipped for lack of in-vocabulary rows -- the normal
    case for the small, domain-specific corpora this package targets. Averaging
    nothing used to raise `ZeroDivisionError` from inside the evaluation, so
    report `nan` and say why instead.
    """
    if len(full_results) == 0:
        logger.warning(
            "no %s dataset had any in-vocabulary row; the vocabulary and the "
            "test data do not overlap, so there is nothing to average",
            kind,
        )
        return {"micro": float("nan"), "macro": float("nan"), "results": []}

    micro_avg = sum(x * y for x, y in zip(line_counts, scores, strict=True)) / sum(
        line_counts
    )
    macro_avg = sum(scores) / len(scores)
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


def eval_similarity(vectors, token2id, preproc_fun, lang="en"):
    """
    evaluate word similarity on several test datasets
    """
    line_counts, spear_results, full_results = [], [], []

    for data in read_test_data(lang, "ws"):
        results = []

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

        if len(results) == 0:
            logger.warning("not enough results for this dataset: %s", data.name)
            continue

        actual, expected = zip(*results, strict=True)
        spear_res = spearmanr(actual, expected).statistic
        spear_results.append(spear_res)
        line_counts.append(len(results))
        oov = (len(lines) - len(results)) / len(lines)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": spear_res,
                "oov": oov,
                # consider the portion of OOV
                "fullscore": penalize_oov(spear_res, oov),
            }
        )

    return aggregate(line_counts, spear_results, full_results, "word similarity")


# analogies
def eval_analogies(vectors, token2id, preproc_fun, lang="en", objective="add"):
    """
    Evaluate word-analogy accuracy on the bundled datasets.

    ``objective`` selects the analogy recovery objective handed to
    ``most_similar_vectors``: ``"add"`` (3CosAdd, the default -- unchanged
    behaviour and the metric the recorded results were computed with) or
    ``"mul"`` (3CosMul, Levy & Goldberg 2014). The exclusion set, OOV handling
    and unanswerable-row skipping are identical for both objectives.
    """
    line_counts, full_results = [], []

    for data in read_test_data(lang, "analogy"):
        results = []

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

        if len(results) == 0:
            logger.warning("not enough results for this dataset: %s", data.name)
            continue

        accuracy = sum(results) / len(results)
        line_counts.append(len(results))
        oov = (len(lines) - len(results)) / len(lines)

        full_results.append(
            {
                "name": f"{lang}_{data_name(data)}",
                "score": accuracy,
                "oov": oov,
                # consider the portion of OOV; accuracy is never negative, so
                # this matches the old `accuracy * (1 - oov)` exactly
                "fullscore": penalize_oov(accuracy, oov),
            }
        )

    scores = [x["score"] for x in full_results]
    return aggregate(line_counts, scores, full_results, "analogy")
