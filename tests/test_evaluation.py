"""
Tests for the similarity and analogy evaluation.

These build a tiny hand-made embedding and point the evaluation at a toy
dataset file, so the expected micro/macro numbers are known exactly. A test
that only checks "no exception was raised" is useless here: the analogy
evaluation used to return a constant 0.0 and such a test passed happily.
"""

import hashlib
from concurrent import futures

import numpy as np
import pytest
from scipy.stats import rankdata

from hyperhyper import evaluation, preprocessing
from hyperhyper.preprocessing import tokenize_texts, tokenize_texts_parallel
from hyperhyper.svd import SVDEmbedding

# A toy vector space, built so that the correct 3CosAdd answer is obvious.
#
# The "relation" is the third dimension:
#   athens  -> greece  adds e3
#   baghdad -> iraq    adds e3
# `distractor` is deliberately placed at `baghdad - e3`, i.e. exactly where the
# *inverted* arithmetic (a - a_ + b instead of a_ - a + b) would land.
TOKEN2ID = {
    "athens": 0,
    "greece": 1,
    "baghdad": 2,
    "iraq": 3,
    "distractor": 4,
}

TOY_VECTORS = np.array(
    [
        [1.0, 0.0, 0.0],  # athens     = e1
        [1.0, 0.0, 1.0],  # greece     = e1 + e3
        [0.0, 1.0, 0.0],  # baghdad    = e2
        [0.0, 1.0, 1.0],  # iraq       = e2 + e3
        [0.0, 1.0, -1.0],  # distractor = e2 - e3
    ]
)


@pytest.fixture()
def toy_embedding():
    """
    A real `SVDEmbedding` over the toy vectors (eig=0 keeps `ut` as-is).
    """
    return SVDEmbedding(TOY_VECTORS, np.ones(3), eig=0.0)


def write_dataset(tmp_path, name, lines):
    path = tmp_path / f"{name}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def use_dataset(monkeypatch):
    """
    Point `read_test_data` at a single file we control.
    """

    def _use(path):
        # accept the `data_dir`/`include_bundled` kwargs the real
        # `read_test_data` now takes, so the callers can still pass them
        monkeypatch.setattr(
            evaluation, "read_test_data", lambda lang, kind, **kwargs: [path]
        )

    return _use


# `to_item`


def test_to_item():
    # the single-element case is the one that matters: the preprocessing
    # returns one list of tokens per input word
    assert evaluation.to_item(["word"]) == "word"
    assert evaluation.to_item([]) is None
    assert evaluation.to_item("word") == "word"
    assert evaluation.to_item(42) == 42
    assert evaluation.to_item([["word"]]) == "word"


def test_to_item_drops_multi_token_entries():
    """
    Regression test for the silent word substitution.

    `to_item` used to return the *first* token of a multi-token entry, so a
    dataset row asking about `vice president` was quietly scored as `vice` and
    `ice-cream` (which tokenizes to `['ice', 'cream']`) as `ice`. A different
    word entered the correlation and the row was never reported as OOV.
    """
    assert evaluation.to_item(["vice", "president"]) is None
    assert evaluation.to_item([["first"], ["second"]]) is None
    assert evaluation.to_item(tokenize_texts(["ice-cream"])[0]) is None


# `read_test_data`


def test_read_test_data():
    ws = evaluation.read_test_data("en", "ws")
    assert len(ws) > 0
    assert all(p.suffix == ".txt" for p in ws)
    # the paths have to still be readable after the function returned
    assert all(p.read_text(encoding="utf-8") for p in ws)

    analogies = evaluation.read_test_data("en", "analogy")
    assert len(analogies) > 0


def test_eval_returns_nan_when_nothing_is_in_vocabulary(
    tmp_path, toy_embedding, use_dataset
):
    """
    Small, domain-specific corpora routinely share no vocabulary at all with
    the bundled English test sets. Every dataset is then skipped and there is
    nothing to average -- which used to raise `ZeroDivisionError` from inside
    `Bunch.svd()` / `Bunch.pmi()`, both of which evaluate by default.
    """
    use_dataset(write_dataset(tmp_path, "oov", ["atlantis lemuria mu hyperborea"]))

    analogies = evaluation.eval_analogies(toy_embedding, TOKEN2ID, tokenize_texts)
    assert analogies["results"] == []
    assert np.isnan(analogies["micro"]) and np.isnan(analogies["macro"])

    use_dataset(write_dataset(tmp_path, "oov_ws", ["atlantis lemuria 7"]))
    sims = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)
    assert sims["results"] == []
    assert np.isnan(sims["micro"]) and np.isnan(sims["macro"])


def test_setup_test_tokens(tmp_path, caplog):
    path = write_dataset(
        tmp_path,
        "toy",
        ["# comment", "athens greece baghdad iraq", "too few columns"],
    )
    with caplog.at_level("WARNING"):
        columns = list(evaluation.setup_test_tokens(path, 4))
    # the comment is skipped and the one four-column line survives
    assert columns == [("athens",), ("greece",), ("baghdad",), ("iraq",)]
    # the malformed data line is no longer dropped silently: it warns, naming
    # the file, the 1-based line number and the offending content
    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "toy.txt:3" in record.getMessage()
    assert "too few columns" in record.getMessage()


def test_setup_test_tokens_ignores_comment_that_splits_into_field_count(tmp_path):
    """
    ADR 0002 roadmap 1: a `#`-comment is skipped by its leading `#` *before* the
    field-count filter, so a comment like `# a b` -- which splits into exactly
    three whitespace fields and used to leak into a 3-field similarity file as a
    data row -- is now safely ignored.
    """
    path = write_dataset(
        tmp_path,
        "leaky",
        ["# a b", "athens greece 9", "greece iraq 5"],
    )
    columns = list(evaluation.setup_test_tokens(path, 3))
    assert columns == [("athens", "greece"), ("greece", "iraq"), ("9", "5")]
    # the same for a 4-field analogy file with a comment that splits into four
    path = write_dataset(
        tmp_path,
        "leaky4",
        ["# a b c", "athens greece baghdad iraq"],
    )
    columns = list(evaluation.setup_test_tokens(path, 4))
    assert columns == [("athens",), ("greece",), ("baghdad",), ("iraq",)]


def test_setup_test_tokens_skips_colon_section_headers(tmp_path):
    """
    A word2vec-style analogy section header (`: capital-common-countries`) is a
    comment too: it must not warn as a malformed row and must not be scored.
    """
    path = write_dataset(
        tmp_path,
        "sections",
        [": capital-common-countries", "athens greece baghdad iraq"],
    )
    columns = list(evaluation.setup_test_tokens(path, 4))
    assert columns == [("athens",), ("greece",), ("baghdad",), ("iraq",)]


def test_setup_test_tokens_warns_and_skips_malformed_row(tmp_path, caplog):
    """
    A genuinely malformed data row (wrong column count, not a comment, not
    blank) is skipped so one bad row does not abort an evaluation -- but it is
    reported with `logger.warning(file:line + content)`, not dropped silently.
    """
    path = write_dataset(
        tmp_path,
        "malformed",
        [
            "athens greece 9",
            "greece iraq",  # a column short -- a typo'd row
            "athens iraq 1",
        ],
    )
    with caplog.at_level("WARNING"):
        columns = list(evaluation.setup_test_tokens(path, 3))

    # the two well-formed rows are recovered; the bad one is dropped
    assert columns == [("athens", "athens"), ("greece", "iraq"), ("9", "1")]

    (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
    message = record.getMessage()
    assert "malformed.txt:2" in message
    assert "greece iraq" in message


def test_setup_test_tokens_clean_file_warns_nothing(tmp_path, caplog):
    """A normal, well-formed file parses unchanged and emits no warning."""
    path = write_dataset(
        tmp_path,
        "clean",
        ["athens greece 9", "greece iraq 5", "athens iraq 1"],
    )
    with caplog.at_level("WARNING"):
        columns = list(evaluation.setup_test_tokens(path, 3))
    assert columns == [
        ("athens", "greece", "athens"),
        ("greece", "iraq", "iraq"),
        ("9", "5", "1"),
    ]
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []


# analogies


def test_eval_analogy_svd(tmp_path, toy_embedding, use_dataset):
    """
    Assert on the actual numbers, not merely that nothing blew up.

    Two of the three lines are in-vocabulary, one of those two is answered
    correctly, so the accuracy is 0.5 and a third of the lines are OOV.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "toy_analogy",
            [
                # correct answer: greece - athens + baghdad == iraq
                "athens greece baghdad iraq",
                # same question, but the expected answer is the wrong word
                "athens greece baghdad distractor",
                # entirely out of vocabulary, has to be skipped
                "atlantis lemuria mu hyperborea",
            ],
        )
    )

    res = evaluation.eval_analogies(toy_embedding, TOKEN2ID, tokenize_texts)

    assert res["micro"] == pytest.approx(0.5)
    assert res["macro"] == pytest.approx(0.5)
    assert len(res["results"]) == 1

    (result,) = res["results"]
    assert result["name"] == "en_toy_analogy"
    assert result["score"] == pytest.approx(0.5)
    assert result["oov"] == pytest.approx(1 / 3)
    assert result["fullscore"] == pytest.approx(0.5 * (2 / 3))


def test_eval_analogy_uses_3cosadd_in_the_right_direction(
    tmp_path, toy_embedding, use_dataset
):
    """
    Regression test for the inverted analogy arithmetic.

    The dataset columns are `a a_ b b_`, i.e. the relation a -> a_ is mirrored
    by b -> b_, so 3CosAdd asks for `a_ - a + b`. The old code computed
    `a - a_ + b` instead, which lands on `distractor` here, and additionally
    compared an int against a list of tuples, so *every* dataset scored a
    constant 0.0.
    """
    use_dataset(write_dataset(tmp_path, "direction", ["athens greece baghdad iraq"]))

    res = evaluation.eval_analogies(toy_embedding, TOKEN2ID, tokenize_texts)
    assert res["micro"] == pytest.approx(1.0)

    # spell out what the two directions actually retrieve, so a future
    # regression points straight at the cause
    a, a_, b = TOKEN2ID["athens"], TOKEN2ID["greece"], TOKEN2ID["baghdad"]
    exclusions = {a, a_, b}

    def first_candidate(positives, negatives):
        guesses = toy_embedding.most_similar_vectors(
            positives, negatives, topn=len(exclusions) + 1
        )
        return next(int(i) for i, _ in guesses if int(i) not in exclusions)

    assert first_candidate([a_, b], [a]) == TOKEN2ID["iraq"]
    assert first_candidate([a, b], [a_]) == TOKEN2ID["distractor"]


def test_eval_analogy_skips_out_of_vocabulary_lines(
    tmp_path, toy_embedding, use_dataset
):
    use_dataset(
        write_dataset(
            tmp_path,
            "mixed",
            ["athens greece baghdad iraq", "athens greece baghdad atlantis"],
        )
    )

    res = evaluation.eval_analogies(toy_embedding, TOKEN2ID, tokenize_texts)
    (result,) = res["results"]
    assert result["oov"] == pytest.approx(0.5)
    assert result["score"] == pytest.approx(1.0)


# 3CosMul objective (FEATURE 1)

# A second toy space, laid out so 3CosAdd and 3CosMul return *different*
# answers for the analogy `athens:greece :: baghdad:?` (columns `a a_ b b_`):
#   greece (a_) = e1, baghdad (b) = e2, athens (a) = e3, mutually orthogonal.
#   cairo (d1) leans on one positive (cos=0.95, 0.10); iraq (d2) is balanced
#   (0.50, 0.50). 3CosAdd sums (1.05 > 1.00) and answers `cairo` (wrong);
#   3CosMul multiplies the [0,1]-mapped cosines and answers `iraq` (correct).
MUL_TOKEN2ID = {"athens": 0, "greece": 1, "baghdad": 2, "cairo": 3, "iraq": 4}
MUL_VECTORS = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],  # athens  (a)
        [1.0, 0.0, 0.0, 0.0],  # greece  (a_)
        [0.0, 1.0, 0.0, 0.0],  # baghdad (b)
        [0.95, 0.10, 0.0, 0.29580399],  # cairo (d1)
        [0.50, 0.50, 0.0, 0.70710678],  # iraq  (d2)
    ]
)


@pytest.fixture()
def cosmul_embedding():
    return SVDEmbedding(MUL_VECTORS, np.ones(4), eig=0.0)


def test_eval_analogy_objective_mul_differs_from_add(
    tmp_path, cosmul_embedding, use_dataset
):
    use_dataset(write_dataset(tmp_path, "cosmul", ["athens greece baghdad iraq"]))

    add = evaluation.eval_analogies(
        cosmul_embedding, MUL_TOKEN2ID, tokenize_texts, objective="add"
    )
    mul = evaluation.eval_analogies(
        cosmul_embedding, MUL_TOKEN2ID, tokenize_texts, objective="mul"
    )

    # 3CosAdd gets this analogy wrong, 3CosMul gets it right
    assert add["micro"] == pytest.approx(0.0)
    assert mul["micro"] == pytest.approx(1.0)

    # the default objective is 3CosAdd, so omitting it is unchanged behaviour
    default = evaluation.eval_analogies(cosmul_embedding, MUL_TOKEN2ID, tokenize_texts)
    assert default["micro"] == pytest.approx(add["micro"])


# word similarity


def test_eval_similarity(tmp_path, toy_embedding, use_dataset):
    """
    The three in-vocabulary pairs are ordered by cosine similarity exactly the
    way the gold scores order them, so Spearman is 1.0.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "toy_ws",
            [
                "athens greece 9",  # cos = 0.707
                "greece iraq 5",  # cos = 0.5
                "athens iraq 1",  # cos = 0.0
                "atlantis lemuria 7",  # out of vocabulary
            ],
        )
    )

    res = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)

    assert res["micro"] == pytest.approx(1.0)
    assert res["macro"] == pytest.approx(1.0)

    (result,) = res["results"]
    assert result["name"] == "en_toy_ws"
    assert result["score"] == pytest.approx(1.0)
    assert result["oov"] == pytest.approx(0.25)
    assert result["fullscore"] == pytest.approx(0.75)


def test_gold_scores_are_ranked_numerically_not_lexicographically(
    tmp_path, toy_embedding, use_dataset
):
    """
    Regression test for the gold column being ranked as text.

    The three pairs are listed in descending cosine order (0.707, 0.5, 0.0) and
    the gold scores descend with them (10 > 9 > 2.5), so Spearman is 1.0.

    `spearmanr` column-stacks its two arguments. Leaving the gold column as the
    `str`s that were read off disk promotes the stacked array to `<U…`, and
    both columns are then ranked lexicographically: "10" < "2.5" < "9" inverts
    the gold ordering and the dataset scores -0.5 instead of 1.0.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "lexicographic",
            [
                "athens greece 10",  # cos = 0.707, highest gold
                "greece iraq 9",  # cos = 0.5
                "athens iraq 2.5",  # cos = 0.0,   lowest gold
            ],
        )
    )

    res = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)
    (result,) = res["results"]
    assert result["score"] == pytest.approx(1.0)


def test_similarity_skips_multi_token_entries_as_oov(
    tmp_path, toy_embedding, use_dataset
):
    """
    Regression test for the silent substitution described in `to_item`.

    `athens-baghdad` tokenizes to `['athens', 'baghdad']`. It used to be scored
    as plain `athens`, so the row silently measured a pair the dataset never
    asked about *and* counted towards the score instead of towards `oov`.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "multi_token",
            [
                "athens greece 9",
                "greece iraq 5",
                "athens iraq 1",
                # scored as `athens greece 3` before the fix
                "athens-baghdad greece 3",
            ],
        )
    )

    res = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)
    (result,) = res["results"]
    assert result["oov"] == pytest.approx(0.25)
    assert result["score"] == pytest.approx(1.0)


def test_fullscore_never_rewards_missing_vocabulary(
    tmp_path, toy_embedding, use_dataset
):
    """
    Regression test for `fullscore = score * (1 - oov)` inverting below zero.

    The gold order here is the exact reverse of the cosine order, so Spearman
    is -1.0 -- what a genuinely bad embedding looks like. Scaling by `1 - oov`
    then *improves* the reported number as more of the dataset goes missing
    (-1.0 becomes -0.75 at 25% OOV), so an embedding could look better simply
    by knowing fewer words. Penalizing the magnitude keeps `fullscore` at or
    below `score` whichever side of zero it is on.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "negative",
            [
                "athens greece 1",  # cos = 0.707, lowest gold
                "greece iraq 5",  # cos = 0.5
                "athens iraq 9",  # cos = 0.0,   highest gold
                "atlantis lemuria 7",  # out of vocabulary
            ],
        )
    )

    res = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)
    (result,) = res["results"]
    assert result["score"] == pytest.approx(-1.0)
    assert result["oov"] == pytest.approx(0.25)
    # the property that matters: OOV is a penalty, never a bonus
    assert result["fullscore"] < result["score"]
    assert result["fullscore"] == pytest.approx(-1.25)


def test_penalize_oov():
    # non-negative scores behave exactly as the old `score * (1 - oov)` did
    assert evaluation.penalize_oov(0.5, 0.0) == pytest.approx(0.5)
    assert evaluation.penalize_oov(0.5, 0.9) == pytest.approx(0.05)
    assert evaluation.penalize_oov(0.0, 0.5) == pytest.approx(0.0)
    # negative scores get worse rather than better
    assert evaluation.penalize_oov(-0.5, 0.0) == pytest.approx(-0.5)
    assert evaluation.penalize_oov(-0.5, 0.9) == pytest.approx(-0.95)


def test_analogy_skips_rows_whose_answer_is_one_of_the_question_words(
    tmp_path, toy_embedding, use_dataset
):
    """
    Regression test for unanswerable analogy rows being scored as wrong.

    A guess only counts if it lies outside `{a, a_, b}`, so a row whose
    expected answer `b_` is itself one of those three can never score 1. Such a
    row used to stay in the accuracy denominator, silently capping the reported
    number; it now lands in `oov` instead.

    This is not an edge case: the default preprocessing lemmatizes, so
    `write writes work works` collapses to `write write work work`, which is
    31% of en/analogy/google.txt and 80% of en/analogy/msr.txt.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "unanswerable",
            [
                "athens greece baghdad iraq",
                # b_ == b: no guess outside {athens, greece, baghdad} can ever
                # be `baghdad`, so this row is unanswerable by construction
                "athens greece baghdad baghdad",
            ],
        )
    )

    res = evaluation.eval_analogies(toy_embedding, TOKEN2ID, tokenize_texts)
    (result,) = res["results"]
    assert result["score"] == pytest.approx(1.0)
    assert result["oov"] == pytest.approx(0.5)
    assert result["fullscore"] == pytest.approx(0.5)


# a perfect embedding, over the real bundled datasets


class PerfectSimilarity:
    """
    A stand-in embedding whose cosine similarity is a strictly increasing
    function of the gold score, mapped onto the real cosine range [-1, 1].

    It reproduces the gold *ranking* without reproducing the gold *values*,
    which is what makes it a usable oracle: Spearman has to come out at exactly
    1.0, and it stays 1.0 under any monotone rescaling of either column.
    """

    def __init__(self, cosines):
        self.cosines = cosines

    def similarity(self, w_idx_1, w_idx_2):
        return self.cosines[(w_idx_1, w_idx_2)]


def build_perfect_similarity(path, preproc_fun):
    """
    Read a similarity dataset and build (dataset, embedding, token2id) such
    that the embedding ranks every surviving row exactly the way the gold
    scores do.

    Rows whose word pair already occurred are dropped, and the de-duplicated
    dataset is written back out. Several of the bundled files list the same
    pair twice with *different* gold scores, and no embedding can rank one word
    pair two ways at once -- keeping them would cap the achievable score below
    1.0 for reasons that have nothing to do with the code under test.
    """
    token1, token2, sims = evaluation.setup_test_tokens(path, 3)
    token1, token2 = preproc_fun(token1), preproc_fun(token2)

    token2id, rows, seen = {}, [], set()
    for x, y, sim in zip(token1, token2, sims, strict=True):
        x, y = evaluation.to_item(x), evaluation.to_item(y)
        if x is None or y is None:
            continue
        for word in (x, y):
            token2id.setdefault(word, len(token2id))
        pair = (token2id[x], token2id[y])
        if pair in seen:
            continue
        seen.add(pair)
        rows.append((x, y, pair, float(sim)))

    # map the gold ranks onto [-1, 1]; tied gold scores keep a tied cosine
    ranks = rankdata([sim for *_, sim in rows])
    lo, hi = ranks.min(), ranks.max()
    cosines = {
        pair: -1.0 + 2.0 * (rank - lo) / (hi - lo)
        for (*_, pair, _), rank in zip(rows, ranks, strict=True)
    }
    return rows, PerfectSimilarity(cosines), token2id


REAL_WS_DATASETS = [
    (lang, path)
    for lang in ("en", "de")
    for path in evaluation.read_test_data(lang, "ws")
]


def test_every_bundled_language_has_similarity_datasets():
    # guards the parametrization below against silently degrading to nothing
    assert len(REAL_WS_DATASETS) == 13


@pytest.mark.parametrize(
    ("lang", "path"),
    REAL_WS_DATASETS,
    ids=[f"{lang}_{evaluation.data_name(p)}" for lang, p in REAL_WS_DATASETS],
)
def test_perfect_embedding_scores_one_on_every_real_dataset(
    lang, path, tmp_path, use_dataset
):
    """
    The strongest available regression test for the gold column being ranked
    as text: an embedding whose cosines reproduce the gold ranking exactly
    *must* score 1.0, on every dataset, by definition of Spearman.

    Before the fix `spearmanr` promoted the stacked columns to strings and
    ranked both lexicographically, so every one of these files scored between
    0.66 and 0.76 instead. `en/bruni_men.txt` was worst (0.659) because its
    gold scores run from 0.000000 to 50.000000 and the varying integer width
    is exactly what lexicographic ordering gets wrong -- and it is half of all
    English similarity rows, so it dominated the micro average.
    """
    rows, vectors, token2id = build_perfect_similarity(path, tokenize_texts)

    deduplicated = write_dataset(
        tmp_path,
        evaluation.data_name(path),
        [f"{x} {y} {sim}" for x, y, _, sim in rows],
    )
    use_dataset(deduplicated)

    res = evaluation.eval_similarity(vectors, token2id, tokenize_texts, lang)
    (result,) = res["results"]

    assert result["oov"] == 0.0
    assert result["score"] == pytest.approx(1.0)
    assert result["fullscore"] == pytest.approx(1.0)


# process pools


class SerialExecutorStub:
    """
    A stand-in for `ProcessPoolExecutor` that does the work in this process.

    Used so the regression tests below stay fast *and* still fail on an
    assertion (with a count) rather than on a timeout, if the evaluation ever
    starts spawning pools again.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fun, iterable, **kwargs):
        return map(fun, iterable)


@pytest.fixture()
def count_pools(monkeypatch):
    """
    Count how often a `ProcessPoolExecutor` gets built, and neuter it.

    `hyperhyper.utils` does `from concurrent import futures` and then reaches
    for `futures.ProcessPoolExecutor`, so patching the attribute on the
    `concurrent.futures` module is what the code under test actually sees.
    """
    calls = []

    class Counting(SerialExecutorStub):
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(futures, "ProcessPoolExecutor", Counting)
    return calls


def test_eval_similarity_spawns_no_process_pools(toy_embedding, count_pools):
    """
    The evaluation preprocesses a few thousand very short strings. Spawning a
    process pool for that costs ~3s per call and the tokenization itself costs
    ~0.08s for *all* of it, so a pool is pure overhead here.

    This used to spawn one pool per dataset per column -- 12 for the six
    bundled English similarity sets, which was 28% of a full `bunch.svd()`.
    """
    evaluation.eval_similarity(
        toy_embedding, TOKEN2ID, tokenize_texts_parallel, lang="en"
    )

    assert count_pools == []


def test_eval_analogies_spawns_no_process_pools(toy_embedding, count_pools):
    evaluation.eval_analogies(
        toy_embedding, TOKEN2ID, tokenize_texts_parallel, lang="en"
    )

    assert count_pools == []


def test_parallel_preprocessing_does_not_change_any_score(
    tmp_path, toy_embedding, use_dataset, count_pools, monkeypatch
):
    """
    The threshold that keeps the evaluation out of the process pool is a
    scheduling decision, not a semantic one: the same tokenizer is applied to
    the same items in the same order either way, so every score has to come out
    bit-identical.

    `count_pools` swaps in a serial executor, so `PARALLEL_MIN_CHARS = 0` here
    reproduces the old, unconditional `map_pool` code path without paying for
    real subprocesses.
    """
    use_dataset(
        write_dataset(
            tmp_path,
            "toy",
            ["athens greece 9.5", "baghdad iraq 8.0", "athens iraq 1.0"],
        )
    )

    serial = evaluation.eval_similarity(
        toy_embedding, TOKEN2ID, tokenize_texts, lang="en"
    )
    thresholded = evaluation.eval_similarity(
        toy_embedding, TOKEN2ID, tokenize_texts_parallel, lang="en"
    )
    assert count_pools == []

    monkeypatch.setattr(preprocessing, "PARALLEL_MIN_CHARS", 0)
    pooled = evaluation.eval_similarity(
        toy_embedding, TOKEN2ID, tokenize_texts_parallel, lang="en"
    )

    assert count_pools != []  # the forced-pool run really did take that path
    # a real score, not the nan that an all-out-of-vocabulary run returns
    assert serial["results"][0]["oov"] == 0.0
    assert serial["micro"] == pytest.approx(0.8660254037844387)
    assert thresholded == serial
    assert thresholded == pooled


# de-duplicated aggregation (TASK 1: the micro double-counting bug)


@pytest.fixture()
def use_datasets(monkeypatch):
    """
    Point `read_test_data` at several files we control, in a fixed order.

    Unlike `use_dataset`, this is what exercises the cross-dataset
    de-duplication: the aggregation only counts a word pair once even when a
    later dataset repeats it.
    """

    def _use(paths):
        monkeypatch.setattr(
            evaluation, "read_test_data", lambda lang, kind, **kwargs: list(paths)
        )

    return _use


def test_micro_counts_a_shared_pair_once(tmp_path, toy_embedding, use_datasets):
    """
    Regression test for the WordSim353 double-counting bug (ADR 0001, TASK 1).

    `ws353_similarity`/`ws353_relatedness` are complete subsets of `ws353`, so
    the old line-weighted micro folded every shared pair in two or three times.
    Here `subset`'s two pairs are a strict subset of `parent`'s three, and the
    two datasets are deliberately scored differently (Spearman -1.0 vs +1.0).

    Old (buggy) micro = (3*-1.0 + 2*+1.0) / 5 = -0.2, macro = 0.0.
    Fixed: `parent` sorts first and claims all three pairs, so `subset` adds no
    new pair, gets weight 0 and is dropped from micro AND macro -- while still
    being reported. Micro and macro are therefore `parent`'s score, -1.0.
    """
    parent = write_dataset(
        tmp_path,
        "aaa_parent",
        # gold ascends 1<2<3 while the cosines descend (0.707>0.5>0.0): Spearman -1
        ["athens greece 1", "greece iraq 2", "athens iraq 3"],
    )
    subset = write_dataset(
        tmp_path,
        "aaa_subset",
        # the same two pairs, gold now agreeing with the cosines: Spearman +1
        ["athens greece 9", "greece iraq 5"],
    )
    use_datasets([parent, subset])

    res = evaluation.eval_similarity(toy_embedding, TOKEN2ID, tokenize_texts)

    # every dataset is still reported individually
    assert {r["name"] for r in res["results"]} == {"en_aaa_parent", "en_aaa_subset"}
    scores = {r["name"]: r["score"] for r in res["results"]}
    assert scores["en_aaa_parent"] == pytest.approx(-1.0)
    assert scores["en_aaa_subset"] == pytest.approx(1.0)

    # the shared pairs are counted once: the redundant subset drops out of the
    # pool, so micro/macro are the parent's score alone -- not the buggy -0.2/0.0
    assert res["micro"] == pytest.approx(-1.0)
    assert res["macro"] == pytest.approx(-1.0)


class RandomSimilarity:
    """
    A deterministic pseudo-random cosine for every unordered id pair.

    Enough to give each real WordSim353 dataset a *different* Spearman score, so
    a test can tell whether the redundant subsets were folded into the pool.
    """

    def similarity(self, i, j):
        a, b = sorted((int(i), int(j)))
        digest = hashlib.blake2b(f"{a}-{b}".encode(), digest_size=4).hexdigest()
        return int(digest, 16) / 0xFFFFFFFF * 2 - 1


def _in_vocab_pairs(path, token2id):
    """
    The unordered word pairs of a similarity file that are fully in-vocabulary,
    in the read order and *keeping* within-file repeats -- so `len(...)` is the
    row count the old line-weighted micro used and `set(...)` is the unique-pair
    count the fixed micro uses.
    """
    t1, t2, _ = evaluation.setup_test_tokens(path, 3)
    t1, t2 = tokenize_texts(t1), tokenize_texts(t2)
    pairs = []
    for x, y in zip(t1, t2, strict=True):
        x, y = evaluation.to_item(x), evaluation.to_item(y)
        if x in token2id and y in token2id:
            pairs.append(frozenset((token2id[x], token2id[y])))
    return pairs


def test_real_overlapping_ws353_split_is_deduplicated(use_datasets):
    """
    On real data with a real overlap: `ws353_similarity` and
    `ws353_relatedness` are the Agirre split and share ~100 pairs. The fixed
    micro must weight the second file only by the pairs the first did not
    already contribute, and that must differ from the old line-weighted micro
    (which summed both raw row counts, folding the shared pairs in twice).

    This does not assume an exact subset relationship, so it stays valid as the
    bundled files are curated -- it only needs the two files to overlap and to
    be scored differently, which `RandomSimilarity` guarantees.
    """
    ws = {evaluation.data_name(p): p for p in evaluation.read_test_data("en", "ws")}
    sim, rel = ws["ws353_similarity"], ws["ws353_relatedness"]

    # a token2id covering every word the two files mention, so oov is minimal
    # and both datasets are scored
    token2id = {}
    for path in (sim, rel):
        t1, t2, _ = evaluation.setup_test_tokens(path, 3)
        for col in (tokenize_texts(t1), tokenize_texts(t2)):
            for token in col:
                token = evaluation.to_item(token)
                if token is not None:
                    token2id.setdefault(token, len(token2id))

    vectors = RandomSimilarity()

    use_datasets([sim, rel])
    res = evaluation.eval_similarity(vectors, token2id, tokenize_texts)

    # both datasets are still reported, with genuinely different scores
    by_name = {r["name"]: r["score"] for r in res["results"]}
    assert set(by_name) == {"en_ws353_similarity", "en_ws353_relatedness"}
    assert by_name["en_ws353_similarity"] != pytest.approx(
        by_name["en_ws353_relatedness"]
    )

    sim_pairs, rel_pairs = (
        _in_vocab_pairs(sim, token2id),
        _in_vocab_pairs(rel, token2id),
    )
    overlap = len(set(sim_pairs) & set(rel_pairs))
    assert overlap > 0  # the whole point: these two files really do share pairs

    # the fixed micro weights `rel` only by its *new* pairs
    seen = set(sim_pairs)
    w_sim = len(set(sim_pairs))
    w_rel = len(set(rel_pairs) - seen)
    expected = (
        w_sim * by_name["en_ws353_similarity"] + w_rel * by_name["en_ws353_relatedness"]
    ) / (w_sim + w_rel)
    assert res["micro"] == pytest.approx(expected)

    # the old, buggy line-weighted micro counted every shared pair twice
    old_buggy = (
        len(sim_pairs) * by_name["en_ws353_similarity"]
        + len(rel_pairs) * by_name["en_ws353_relatedness"]
    ) / (len(sim_pairs) + len(rel_pairs))
    assert res["micro"] != pytest.approx(old_buggy)


def test_identical_dataset_repeated_is_counted_once(use_datasets):
    """
    The dedup extreme, on the real `ws353` file: evaluating it twice must give
    the same micro/macro as evaluating it once, because the repeat adds no new
    pair. Robust to curation -- it only reads one file and compares it to a copy
    of itself.
    """
    ws = {evaluation.data_name(p): p for p in evaluation.read_test_data("en", "ws")}
    parent = ws["ws353"]

    token2id = {}
    t1, t2, _ = evaluation.setup_test_tokens(parent, 3)
    for col in (tokenize_texts(t1), tokenize_texts(t2)):
        for token in col:
            token = evaluation.to_item(token)
            if token is not None:
                token2id.setdefault(token, len(token2id))

    vectors = RandomSimilarity()

    use_datasets([parent])
    once = evaluation.eval_similarity(vectors, token2id, tokenize_texts)
    use_datasets([parent, parent])
    twice = evaluation.eval_similarity(vectors, token2id, tokenize_texts)

    assert len(twice["results"]) == 2  # still reported twice
    assert twice["micro"] == pytest.approx(once["micro"])
    assert twice["macro"] == pytest.approx(once["macro"])


# user-supplied data (TASK 2: data_dir)


def write_kind_dataset(root, lang, kind, name, lines, *, nested):
    """
    Write a dataset into a `data_dir`, either the bundle-mirroring
    `<root>/<lang>/<kind>/` layout or the flat `<root>/<kind>/` one.
    """
    directory = (root / lang / kind) if nested else (root / kind)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("nested", [True, False], ids=["lang-nested", "flat"])
def test_data_dir_evaluates_user_supplied_dataset(tmp_path, toy_embedding, nested):
    """
    TASK 2: a user points `data_dir` at their own `ws/` files and they are
    evaluated. `include_bundled=False` restricts scoring to just those files.
    """
    write_kind_dataset(
        tmp_path,
        "en",
        "ws",
        "domain",
        ["athens greece 9", "greece iraq 5", "athens iraq 1"],
        nested=nested,
    )

    res = evaluation.eval_similarity(
        toy_embedding,
        TOKEN2ID,
        tokenize_texts,
        data_dir=tmp_path,
        include_bundled=False,
    )

    (result,) = res["results"]
    assert result["name"] == "en_domain"
    assert result["score"] == pytest.approx(1.0)
    assert res["micro"] == pytest.approx(1.0)


def test_data_dir_evaluates_alongside_bundled_by_default(tmp_path, toy_embedding):
    """
    The default is `include_bundled=True`: the user's set is evaluated
    *alongside* the bundled ones. The toy vocabulary shares no word with the
    real English sets, so those are skipped as out-of-vocabulary and only the
    custom set is scored -- but it is the read_test_data merge, not a
    replacement, that put it there (the bundled files were attempted).
    """
    write_kind_dataset(
        tmp_path,
        "en",
        "ws",
        "domain",
        ["athens greece 9", "greece iraq 5"],
        nested=True,
    )

    merged = evaluation.read_test_data("en", "ws", data_dir=tmp_path)
    bundled = evaluation.read_test_data("en", "ws")
    assert len(merged) == len(bundled) + 1
    assert "domain.txt" in {p.name for p in merged}

    res = evaluation.eval_similarity(
        toy_embedding, TOKEN2ID, tokenize_texts, data_dir=tmp_path
    )
    assert "en_domain" in {r["name"] for r in res["results"]}


def test_data_dir_analogies(tmp_path, toy_embedding):
    """TASK 2 for the analogy side of the evaluator."""
    write_kind_dataset(
        tmp_path,
        "en",
        "analogy",
        "domain",
        ["athens greece baghdad iraq"],
        nested=True,
    )
    res = evaluation.eval_analogies(
        toy_embedding,
        TOKEN2ID,
        tokenize_texts,
        data_dir=tmp_path,
        include_bundled=False,
    )
    (result,) = res["results"]
    assert result["name"] == "en_domain"
    assert result["score"] == pytest.approx(1.0)


# coverage report (TASK 3)


def test_dataset_coverage_reports_in_vocabulary_fraction(tmp_path, toy_embedding):
    """
    TASK 3: `dataset_coverage` reports, per dataset, the fraction of rows whose
    every word survives preprocessing to a single in-vocabulary token -- the
    same rows the evaluator can actually score.
    """
    write_kind_dataset(
        tmp_path,
        "en",
        "ws",
        "domain",
        [
            "athens greece 9",  # both in vocab
            "greece iraq 5",  # both in vocab
            "athens atlantis 3",  # atlantis is OOV
            "lemuria mu 1",  # both OOV
        ],
        nested=True,
    )

    report = evaluation.dataset_coverage(
        TOKEN2ID, tokenize_texts, kind="ws", data_dir=tmp_path, include_bundled=False
    )

    (entry,) = report
    assert entry["name"] == "en_domain"
    assert entry["kind"] == "ws"
    assert entry["rows"] == 4
    assert entry["covered"] == 2
    assert entry["coverage"] == pytest.approx(0.5)


def test_dataset_coverage_matches_the_evaluator(tmp_path, toy_embedding):
    """
    The coverage fraction has to equal `1 - oov` the evaluator reports, or the
    number would mislead the user it is meant to help. A hyphenated entry
    (dropped by `to_item`) must count as not-covered on both sides.
    """
    lines = [
        "athens greece 9",
        "greece iraq 5",
        "athens iraq 1",
        "athens-baghdad greece 3",  # multi-token -> OOV for the evaluator
    ]
    write_kind_dataset(tmp_path, "en", "ws", "domain", lines, nested=True)

    (entry,) = evaluation.dataset_coverage(
        TOKEN2ID, tokenize_texts, kind="ws", data_dir=tmp_path, include_bundled=False
    )
    res = evaluation.eval_similarity(
        toy_embedding,
        TOKEN2ID,
        tokenize_texts,
        data_dir=tmp_path,
        include_bundled=False,
    )
    (result,) = res["results"]

    assert entry["coverage"] == pytest.approx(3 / 4)
    assert entry["coverage"] == pytest.approx(1 - result["oov"])


def test_dataset_coverage_analogy_kind(tmp_path):
    """Coverage also works for the 4-column analogy datasets."""
    write_kind_dataset(
        tmp_path,
        "en",
        "analogy",
        "domain",
        [
            "athens greece baghdad iraq",  # all four in vocab
            "athens greece baghdad atlantis",  # atlantis OOV
        ],
        nested=True,
    )
    (entry,) = evaluation.dataset_coverage(
        TOKEN2ID,
        tokenize_texts,
        kind="analogy",
        data_dir=tmp_path,
        include_bundled=False,
    )
    assert entry["rows"] == 2
    assert entry["covered"] == 1
    assert entry["coverage"] == pytest.approx(0.5)


def test_dataset_coverage_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        evaluation.dataset_coverage(TOKEN2ID, tokenize_texts, kind="bogus")
