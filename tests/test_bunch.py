import subprocess
import sys

import numpy as np
import pytest

import hyperhyper
from hyperhyper.bunch import _canonical
from hyperhyper.preprocessing import texts_to_sents, tokenize_texts


def test_bunch_pmi(corpus, bunch_path):
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    pmi_matrix, eval_results = bunch.pmi()

    assert pmi_matrix.m.count_nonzero() > 0
    assert set(eval_results) == {"micro", "macro", "results"}

    english_idx = corpus.vocab.token2id["english"]
    wikipedia_idx = corpus.vocab.token2id["wikipedia"]

    most_similar = pmi_matrix.most_similar(english_idx)
    assert len(most_similar) > 0
    for token_idx, sim in most_similar:
        assert pmi_matrix.similarity(english_idx, token_idx) == pytest.approx(
            pmi_matrix.similarity(token_idx, english_idx), rel=1e-5
        )
        assert pmi_matrix.similarity(english_idx, token_idx) == pytest.approx(
            sim, rel=1e-5
        )

    guesses = pmi_matrix.most_similar_vectors([english_idx], [wikipedia_idx], topn=3)
    assert len(guesses) == 3
    assert all(0 <= int(idx) < pmi_matrix.m.shape[0] for idx, _ in guesses)


def test_bunch_svd(corpus, bunch_path):
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    svd_matrix, eval_results = bunch.svd(dim=2)

    assert svd_matrix.m.shape == (corpus.vocab.size + 1, 2)
    assert set(eval_results) == {"micro", "macro", "results"}

    english_idx = corpus.vocab.token2id["english"]
    for token_idx, sim in svd_matrix.most_similar(english_idx):
        assert svd_matrix.similarity(english_idx, token_idx) == pytest.approx(
            svd_matrix.similarity(token_idx, english_idx)
        )
        assert svd_matrix.similarity(english_idx, token_idx) == pytest.approx(sim)


def test_svd_add_context_participates_in_cache_key(corpus, bunch_path):
    """
    FEATURE 2: `add_context` must move the svd cache path, so the `w+c` and the
    word-only factorization never share a cache entry -- and the default
    (`add_context=False`) must stay a distinct, stable key.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    base = {
        "impl": "scipy",
        "impl_args": {},
        "neg": 1,
        "cds": 0.75,
        "dim": 2,
        "pair_args": bunch._effective_pair_args(),
    }
    off = bunch.dict_to_path("svd", {**base, "add_context": False})
    on = bunch.dict_to_path("svd", {**base, "add_context": True})
    assert off != on


def test_svd_wplusc_end_to_end_and_composes_with_cosmul(corpus, bunch_path):
    """
    FEATURE 2 through the public API, composing with FEATURE 1.

    `add_context=True` builds a genuinely different representation than the
    word-only default, lands in its own cache file, and 3CosMul evaluates on it.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)

    word_only, _ = bunch.svd(dim=4)
    wplusc, _ = bunch.svd(dim=4, add_context=True)

    # two distinct cache entries on disk
    svd_files = list((bunch_path / "svd").iterdir())
    assert len(svd_files) == 2

    # same shape, but the context vectors really were folded in
    assert wplusc.m.shape == word_only.m.shape
    assert not np.allclose(wplusc.m, word_only.m)

    # 3CosMul (FEATURE 1) composes with the w+c representation (FEATURE 2)
    english_idx = corpus.vocab.token2id["english"]
    wikipedia_idx = corpus.vocab.token2id["wikipedia"]
    guesses = wplusc.most_similar_vectors(
        [english_idx], [wikipedia_idx], topn=3, objective="mul"
    )
    assert len(guesses) == 3
    assert all(0 <= int(idx) < wplusc.m.shape[0] for idx, _ in guesses)


def test_svd_default_path_unchanged_by_add_context_feature(corpus, bunch_path):
    """
    Guard that the word-only svd is byte-for-byte the previous behaviour: the
    cached arrays are still the two-array `a1`/`a2` layout, with no context
    vectors written.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    _ut, _s, vt = bunch.svd_matrix(impl="scipy", dim=2)
    assert vt is None

    (svd_file,) = list((bunch_path / "svd").iterdir())
    loaded = np.load(str(svd_file))
    assert sorted(loaded.files) == ["a1", "a2"]  # no a3 (context) array


def test_bunch_svd_invalid_impl(corpus, bunch_path):
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    with pytest.raises(ValueError):
        bunch.svd(dim=2, impl="sparsesvd")


def test_bunch_keyed_vectors(corpus, bunch_path):
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)

    keyed_vectors, _ = bunch.svd(dim=2, keyed_vectors=True)
    assert keyed_vectors.vector_size == 2
    assert "english" in keyed_vectors

    keyed_vectors = bunch.svd(dim=3, keyed_vectors=True, evaluate=False)
    assert keyed_vectors.vector_size == 3

    # `most_similar` here comes from gensim's keyed vectors
    similar = keyed_vectors.most_similar("english")
    assert len(similar) > 0
    assert all(isinstance(token, str) for token, _ in similar)


def test_bunch_refuses_to_overwrite(corpus, bunch_path):
    hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    with pytest.raises(ValueError):
        hyperhyper.Bunch(bunch_path, corpus)


def test_recorded_method_names(corpus, bunch_path):
    """
    The `record` decorator must not swallow the identity of the two headline
    public methods.
    """
    assert hyperhyper.Bunch.pmi.__name__ == "pmi"
    assert hyperhyper.Bunch.svd.__name__ == "svd"
    assert hyperhyper.Bunch.svd.__doc__ is not None


def test_db_query(corpus, bunch_path):
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(dim=2)

    res = bunch.results(query={"dim": 2, "pair_args": {"window": 2}})
    assert len(res) == 1
    assert res[0]["dim"] == 2
    assert res[0]["method"] == "svd"
    assert res[0]["pair_args__window"] == 2

    # a query that matches nothing comes back empty
    assert bunch.results(query={"dim": 99}) == []


def test_bunch_text_files(text_files, bunch_path):
    corpus = hyperhyper.Corpus.from_text_files(text_files, preproc_func=tokenize_texts)
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)

    pmi_matrix, _ = bunch.pmi()
    assert pmi_matrix.m.count_nonzero() > 0

    svd_matrix, _ = bunch.svd(dim=2)
    assert svd_matrix.m.shape == (corpus.vocab.size + 1, 2)


def test_pmi_cache_key_covers_kwargs(corpus_factory, tmp_path):
    """
    Regression test for the cache key ignoring `**kwargs`.

    `pmi_matrix` built its path from `{"cds": cds, **pair_args}` while passing
    `**kwargs` on to `count_pairs`, so `window` was invisible to the key: a
    sweep served the *first* window's matrix for every later one, and the answer
    depended on the order the sweep happened to run in.
    """

    def sweep(path, windows):
        bunch = hyperhyper.Bunch(path, corpus_factory(), force_overwrite=True)
        return {w: bunch.pmi_matrix(window=w).count_nonzero() for w in windows}

    forwards = sweep(tmp_path / "forwards", [2, 10])
    backwards = sweep(tmp_path / "backwards", [10, 2])

    # a sweep must produce genuinely different matrices ...
    assert forwards[2] != forwards[10]
    # ... and must not depend on the order they were built in
    assert forwards == backwards


def test_pmi_public_api_sweep_is_order_independent(corpus_factory, tmp_path):
    """
    The same hole, reached through the public `Bunch.pmi(window=...)`.
    """

    def sweep(path, windows):
        bunch = hyperhyper.Bunch(path, corpus_factory(), force_overwrite=True)
        out = {}
        for w in windows:
            embedding, _ = bunch.pmi(window=w)
            out[w] = embedding.m.count_nonzero()
        return out

    forwards = sweep(tmp_path / "fwd", [2, 10])
    backwards = sweep(tmp_path / "bwd", [10, 2])

    assert forwards[2] != forwards[10]
    assert forwards == backwards


def test_matrix_cache_paths_separate_every_pair_count_argument(corpus, bunch_path):
    """
    Every argument that reaches `count_pairs` has to move the cache path -- for
    the `pmi` and the `svd` cache alike.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)

    def pmi_path(**kwargs):
        return bunch.dict_to_path(
            "pmi", {"cds": 0.75, "pair_args": bunch._effective_pair_args(**kwargs)}
        )

    def svd_path(**kwargs):
        return bunch.dict_to_path(
            "svd",
            {
                "impl": "scipy",
                "impl_args": {},
                "neg": 1,
                "cds": 0.75,
                "dim": 2,
                "pair_args": bunch._effective_pair_args(**kwargs),
            },
        )

    for kwargs in ({"window": 10}, {"subsample": "off"}, {"min_count": 3}, {"seed": 7}):
        assert pmi_path(**kwargs) != pmi_path()
        assert svd_path(**kwargs) != svd_path()

    # spelling the very same run two different ways stays one entry
    assert pmi_path(window=2) == pmi_path()
    assert svd_path(pair_args={"window": 2}) == svd_path(window=2)


def test_pair_counts_cache_key_is_the_effective_parameter_set(corpus, bunch_path):
    """
    `pair_counts()` and `pair_counts(window=2)` are the same computation, so
    they must share one cache entry rather than writing `default.npz` next to
    `window_2.npz` -- a name whose meaning silently moves when a default does.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.pair_counts()
    bunch.pair_counts(window=2)

    files = list((bunch_path / "pair_counts").iterdir())
    assert len(files) == 1
    # a format tag, so a changed default cannot resurrect a stale entry
    assert "_v2-" in files[0].name or files[0].name.startswith("v2-")


def test_dict_to_path_does_not_collide(corpus, bunch_path):
    """
    The old `"_".join(f"{k}_{v}".lower())` scheme neither reserved its separator
    nor preserved case, so plainly different parameter sets shared a file.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    collide = [
        ({"impl": "Scipy"}, {"impl": "scipy"}),
        ({"decay": None}, {"decay": "None"}),
        ({"dyn": 1, "win": 5}, {"dyn": "1_win_5"}),
        ({"a": "b_c", "d": 1}, {"a_b": "c", "d": 1}),
        # the realistic vector: `impl_args` and `pair_args` sharing a key name
        (
            {"impl_args": {"n": 1}, "pair_args": {}},
            {"impl_args": {}, "pair_args": {"n": 1}},
        ),
    ]
    for left, right in collide:
        assert bunch.dict_to_path("svd", left) != bunch.dict_to_path("svd", right)

    # the deliberate integral-float cast must survive: `dim=500.0` is `dim=500`
    assert bunch.dict_to_path("svd", {"dim": 500.0}) == bunch.dict_to_path(
        "svd", {"dim": 500}
    )


# a set whose `repr` order is hash-randomised across `PYTHONHASHSEED` values
_DIGEST_SCRIPT = (
    "from hyperhyper.bunch import Bunch;"
    "import types;"
    "b = Bunch.__new__(Bunch);"
    "b.path = __import__('pathlib').Path('/x');"
    "print(b.dict_to_path('svd', {'tags': {'a', 'b', 'c', 'd', 'e', 'f'}}).name)"
)


def test_canonical_sorts_sets_deterministically():
    """
    Regression test for BUG 4: a `set` was serialised via `repr`, whose order is
    hash-randomised, so the cache digest changed run to run. `_canonical` now
    turns a set into a sorted list.
    """
    assert _canonical({"c", "a", "b"}) == ["a", "b", "c"]
    assert _canonical(frozenset({3, 1, 2})) == [1, 2, 3]
    # nested inside the containers the cache key is actually built from
    assert _canonical({"k": {"y", "x"}}) == {"k": ["x", "y"]}


def test_canonical_rejects_unserialisable_values():
    """
    BUG 4: rather than falling back to `repr` (unstable for many types),
    `_canonical` refuses anything `json.dumps` cannot handle natively.
    """
    with pytest.raises(TypeError):
        _canonical(object())


def test_scalar_params_do_not_collide():
    """
    BUG 4 must not regress the property that `True`, `None`, `1` and `"1"` map to
    distinct cache keys (they serialise to `true`, `null`, `1` and `"1"`).
    """
    import json

    encoded = {json.dumps(_canonical(v)) for v in (True, False, None, 1, 0, "1", "0")}
    assert len(encoded) == 7


def test_cache_digest_is_stable_across_hash_seeds():
    """
    BUG 4, end to end: the same set-valued parameter must hash to the same cache
    file name in two processes started with different `PYTHONHASHSEED`, or a
    sweep silently misses its own cache from one run to the next.
    """

    def digest(seed):
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST_SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": str(seed), "PATH": __import__("os").environ["PATH"]},
        )
        return out.stdout.strip()

    seeds = [digest(s) for s in (0, 1, 2, 12345)]
    assert len(set(seeds)) == 1, seeds


def test_svd_dim_larger_than_vocab_does_not_crash(corpus, bunch_path):
    """
    BUG 2, through the public API: a `dim` at or above the vocabulary size used
    to crash the default `scipy` backend with an opaque ARPACK `ValueError`. It
    must now clamp and succeed, returning an embedding no wider than the rank.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    vocab = corpus.vocab.size + 1
    embedding, _ = bunch.svd(dim=vocab + 50)
    assert 0 < embedding.m.shape[1] < vocab
    assert embedding.m.shape[0] == vocab


def test_record_binds_positional_and_default_arguments(corpus, bunch_path):
    """
    `record` inspected only `kwargs`: positional arguments vanished and unspoken
    defaults were never written, so `b.svd(5)` and `b.svd(10)` produced rows
    that were identical in every parameter column.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(2)
    bunch.svd(3)

    rows = bunch.results()
    assert sorted(r["dim"] for r in rows) == [2, 3]
    # a default that was never spelled out is still recorded and still findable
    assert all(r["cds"] == 0.75 for r in rows)
    assert len(bunch.results(query={"cds": 0.75})) == 2


def test_record_captures_the_tokenizer_identity(corpus, bunch_path):
    """
    ADR 0002 item 5: the corpus tokenizer's qualname joins the recorded
    parameters, so scores computed under v1 and v2 are attributable and never
    collide in results.db. The `corpus` fixture uses `tokenize_texts`.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(dim=2)

    row = bunch.results()[0]
    assert row["tokenizer"] == "tokenize_texts"
    # and it is a real, queryable parameter column
    assert len(bunch.results(query={"tokenizer": "tokenize_texts"})) == 1
    assert bunch.results(query={"tokenizer": "tokenize_texts_parallel_v2"}) == []


def test_record_tokenizer_identity_separates_v1_and_v2(tmp_path):
    """
    Two corpora that differ *only* in their tokenizer must not overwrite each
    other's row: the same run under v1 and v2 has different vocab and different
    numbers, so the recorded `tokenizer` column keeps them as distinct rows.
    """
    from hyperhyper.preprocessing import tokenize_texts, tokenize_texts_v2

    texts = ["the english wikipedia 2001", "english wikipedia founded"] * 20
    v1 = hyperhyper.Corpus.from_sents(texts, preproc_func=tokenize_texts)
    v2 = hyperhyper.Corpus.from_sents(texts, preproc_func=tokenize_texts_v2)

    hyperhyper.Bunch(tmp_path / "v1", v1, force_overwrite=True).svd(dim=2)
    with hyperhyper.Bunch(tmp_path / "v1") as b1:
        assert b1.results()[0]["tokenizer"] == "tokenize_texts"

    hyperhyper.Bunch(tmp_path / "v2", v2, force_overwrite=True).svd(dim=2)
    with hyperhyper.Bunch(tmp_path / "v2") as b2:
        assert b2.results()[0]["tokenizer"] == "tokenize_texts_v2"


def test_record_attributes_kwargs_to_pair_args(corpus, bunch_path):
    """
    A loose `window=` is forwarded to `count_pairs`, so it must be recorded as
    `pair_args__window` -- not as a sibling column contradicting it.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(dim=2, window=10)

    row = bunch.results()[0]
    assert row["pair_args__window"] == 10
    assert "window" not in row or row["window"] is None


def test_results_filter_on_string_columns(corpus, bunch_path):
    """
    Values were interpolated unquoted, so any string filter asked SQLite for a
    column of that name: `{"impl": "scipy"}` raised `no such column: scipy`.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(dim=2)

    assert len(bunch.results(query={"impl": "scipy"})) == 1
    assert len(bunch.results(query={"method": "svd"})) == 1
    assert bunch.results(query={"impl": "gensim"}) == []
    assert len(bunch.results(query={"pair_args": {"dynamic_window": "deter"}})) == 1


def test_results_rejects_sql_injection(corpus, bunch_path):
    """
    `order`, `limit` and the query keys were all interpolated into the SQL.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.svd(dim=2)

    with pytest.raises(ValueError):
        bunch.results(order="micro_results desc; drop table experiments")
    with pytest.raises(ValueError):
        bunch.results(limit="1; drop table experiments")
    with pytest.raises(ValueError):
        bunch.results(query={"dim=2 or 1": 1})

    # a value that looks like SQL is data, not syntax
    assert bunch.results(query={"impl": "scipy'; drop table experiments; --"}) == []
    assert len(bunch.results()) == 1


def test_force_overwrite_without_corpus_keeps_the_data(corpus, bunch_path):
    """
    `delete_folder` ran before the `corpus is None` branch decided to *load*
    `corpus.pkl`, so reopening a bunch this way wiped every cached matrix and
    the results database, and only then raised `FileNotFoundError`.
    """
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)
    bunch.pmi_matrix()
    before = sorted(str(p.relative_to(bunch_path)) for p in bunch_path.rglob("*"))
    assert before

    with pytest.raises(ValueError):
        hyperhyper.Bunch(bunch_path, force_overwrite=True)

    after = sorted(str(p.relative_to(bunch_path)) for p in bunch_path.rglob("*"))
    assert after == before

    # and the bunch is still usable afterwards
    assert hyperhyper.Bunch(bunch_path).corpus.vocab.size == corpus.vocab.size


def test_bunch_closes_its_database(corpus, bunch_path):
    """
    `get_db` memoised a connection that nothing ever disposed of.
    """
    with hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True) as bunch:
        bunch.get_db()
        assert bunch.db is not None
    assert bunch.db is None

    # closing twice is not an error, and the bunch reconnects on demand
    bunch.close()
    assert bunch.get_db() is not None
    bunch.close()


@pytest.mark.slow
def test_bunch_text_files_spacy(spacy_model, text_files, bunch_path):
    corpus = hyperhyper.Corpus.from_text_files(text_files, preproc_func=texts_to_sents)
    bunch = hyperhyper.Bunch(bunch_path, corpus, force_overwrite=True)

    pmi_matrix, _ = bunch.pmi()
    assert pmi_matrix.m.count_nonzero() > 0

    keyed_vectors = bunch.svd(dim=2, keyed_vectors=True, evaluate=False)
    assert len(keyed_vectors.most_similar("english")) > 0
