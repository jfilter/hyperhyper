import pytest

import hyperhyper
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
