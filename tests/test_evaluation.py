"""
Tests for the similarity and analogy evaluation.

These build a tiny hand-made embedding and point the evaluation at a toy
dataset file, so the expected micro/macro numbers are known exactly. A test
that only checks "no exception was raised" is useless here: the analogy
evaluation used to return a constant 0.0 and such a test passed happily.
"""

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
        monkeypatch.setattr(evaluation, "read_test_data", lambda lang, kind: [path])

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


def test_setup_test_tokens(tmp_path):
    path = write_dataset(
        tmp_path,
        "toy",
        ["# comment", "athens greece baghdad iraq", "too few columns"],
    )
    columns = list(evaluation.setup_test_tokens(path, 4))
    # only the one line with exactly four columns survives
    assert columns == [("athens",), ("greece",), ("baghdad",), ("iraq",)]


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
