"""
Tests for the domain proxy tasks -- synonym multiple choice and category
purity (ADR 0001, P4).

These build a tiny hand-made vector space where the right answer is arithmetic
rather than a matter of opinion, so every expected number below is exact. The
point of the tasks is that their gold is a *membership fact*; the point of these
tests is that the scoring of that fact is not merely "runs without raising" --
the analogy evaluation once returned a constant 0.0 and passed such a test.
"""

import numpy as np
import pytest

from hyperhyper import evaluation
from hyperhyper.evaluation import MalformedDatasetError
from hyperhyper.svd import SVDEmbedding
from tools.build_domain_tasks import build_domain_tasks as builder

# A toy space over two well-separated clusters. Within a cluster the vectors are
# close; across clusters they are orthogonal, so every nearest neighbour is a
# cluster-mate and cosine ordering is obvious by inspection.
TOKEN2ID = {
    "aspirin": 0,
    "ibuprofen": 1,
    "paracetamol": 2,
    "liver": 3,
    "kidney": 4,
    "spleen": 5,
    "isolated": 6,
}

TOY_VECTORS = np.array(
    [
        [1.0, 0.00, 0.0, 0.0],  # aspirin
        [1.0, 0.10, 0.0, 0.0],  # ibuprofen   -- nearest to aspirin
        [1.0, 0.25, 0.0, 0.0],  # paracetamol
        [0.0, 0.00, 1.0, 0.00],  # liver
        [0.0, 0.00, 1.0, 0.10],  # kidney
        [0.0, 0.00, 1.0, 0.25],  # spleen
        [0.0, 1.00, 0.0, 0.0],  # isolated -- orthogonal to both clusters
    ]
)


@pytest.fixture()
def toy_embedding():
    return SVDEmbedding(TOY_VECTORS, np.ones(4), eig=0.0)


def identity_preproc(tokens):
    """A `preproc_func` that tokenizes nothing -- the toy words are already tokens."""
    return list(tokens)


def write(tmp_path, kind, name, header, rows, lang="en"):
    path = tmp_path / lang / kind / f"{name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join("\t".join(row) for row in rows)
    path.write_text(f"# hyperhyper-eval: 1\n{header}\n{body}\n", encoding="utf-8")
    return path


SYN_HEADER = "target\tanswer\tdistractor1\tdistractor2"
CAT_HEADER = "word\tcategory"


# ---------------------------------------------------------------------------
# parsing: the width comes from the file's own header
# ---------------------------------------------------------------------------


def test_synonym_width_is_read_from_the_header(tmp_path):
    # three distractors here, two in the fixture above: the reader must take the
    # count from the file rather than from a hardcoded column number
    path = write(
        tmp_path,
        "synonym",
        "wide",
        "target\tanswer\tdistractor1\tdistractor2\tdistractor3",
        [["aspirin", "ibuprofen", "liver", "kidney", "spleen"]],
    )
    columns = list(evaluation.setup_test_tokens(path, kind="synonym"))
    assert len(columns) == 5
    assert columns[0] == ("aspirin",)


def test_synonym_needs_at_least_two_distractors(tmp_path):
    # a single distractor makes the question a coin flip and, worse, gives a
    # three-column row indistinguishable from a `ws` row
    path = write(
        tmp_path,
        "synonym",
        "narrow",
        "target\tanswer\tdistractor1",
        [["aspirin", "ibuprofen", "liver"]],
    )
    with pytest.raises(MalformedDatasetError, match="at least 2"):
        list(evaluation.setup_test_tokens(path, kind="synonym"))


def test_synonym_row_wider_than_its_header_raises(tmp_path):
    path = write(
        tmp_path,
        "synonym",
        "ragged",
        SYN_HEADER,
        [["aspirin", "ibuprofen", "liver", "kidney"], ["a", "b", "c"]],
    )
    with pytest.raises(MalformedDatasetError, match=r"ragged\.tsv:4"):
        list(evaluation.setup_test_tokens(path, kind="synonym"))


def test_category_header_is_fixed(tmp_path):
    path = write(tmp_path, "category", "bad", "word\tclass", [["liver", "organ"]])
    with pytest.raises(MalformedDatasetError, match="header must be"):
        list(evaluation.setup_test_tokens(path, kind="category"))


def test_p4_kinds_are_tsv_only(tmp_path):
    legacy = tmp_path / "en" / "synonym" / "old.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("aspirin ibuprofen liver kidney\n", encoding="utf-8")
    # the whitespace format has no header, so it cannot declare a row width
    with pytest.raises(MalformedDatasetError, match="TSV-only"):
        list(evaluation.setup_test_tokens(legacy, kind="synonym"))


def test_setup_test_tokens_rejects_ambiguous_arguments(tmp_path):
    path = write(tmp_path, "category", "c", CAT_HEADER, [["liver", "organ"]])
    with pytest.raises(ValueError, match="not both"):
        evaluation.setup_test_tokens(path, 3, kind="category")
    with pytest.raises(ValueError, match="keep_len"):
        evaluation.setup_test_tokens(path)


# ---------------------------------------------------------------------------
# synonym multiple choice
# ---------------------------------------------------------------------------


def evaluate_synonyms(tmp_path, embedding, **kwargs):
    return evaluation.eval_synonyms(
        embedding,
        TOKEN2ID,
        identity_preproc,
        data_dir=tmp_path,
        include_bundled=False,
        **kwargs,
    )


def test_synonym_scores_a_correct_choice(tmp_path, toy_embedding):
    # ibuprofen is the nearest neighbour of aspirin by construction
    write(
        tmp_path,
        "synonym",
        "easy",
        SYN_HEADER,
        [["aspirin", "ibuprofen", "liver", "kidney"]],
    )
    result = evaluate_synonyms(tmp_path, toy_embedding)
    assert result["results"][0]["score"] == 1.0
    assert result["results"][0]["oov"] == 0.0


def test_synonym_scores_a_wrong_choice(tmp_path, toy_embedding):
    # the "answer" here is in the other cluster, so a distractor wins
    write(
        tmp_path,
        "synonym",
        "wrong",
        SYN_HEADER,
        [["aspirin", "liver", "ibuprofen", "kidney"]],
    )
    assert evaluate_synonyms(tmp_path, toy_embedding)["results"][0]["score"] == 0.0


def test_synonym_tie_does_not_count_as_correct(tmp_path, toy_embedding):
    # `isolated` is orthogonal to both clusters, so its cosine to the answer and
    # to the distractors is identically 0 -- the model has chosen nothing, and
    # scoring that as a hit would report a chance artefact as competence
    write(
        tmp_path,
        "synonym",
        "tie",
        SYN_HEADER,
        [["isolated", "aspirin", "liver", "kidney"]],
    )
    assert evaluate_synonyms(tmp_path, toy_embedding)["results"][0]["score"] == 0.0


def test_synonym_row_with_any_oov_candidate_is_not_scored(tmp_path, toy_embedding):
    # dropping only the missing distractor would make the question easier for
    # exactly the rows the corpus covers worst
    write(
        tmp_path,
        "synonym",
        "partial",
        SYN_HEADER,
        [
            ["aspirin", "ibuprofen", "liver", "kidney"],
            ["aspirin", "ibuprofen", "liver", "not_in_vocab"],
        ],
    )
    result = evaluate_synonyms(tmp_path, toy_embedding)["results"][0]
    assert result["oov"] == 0.5
    assert result["score"] == 1.0


def test_synonym_answer_equal_to_target_reports_nan(tmp_path, toy_embedding):
    # a candidate identical to the target wins on cosine 1 by construction and
    # says nothing about the embedding, so the row is skipped -- leaving nothing
    # to average, which `aggregate` reports as nan rather than raising
    write(
        tmp_path,
        "synonym",
        "self",
        SYN_HEADER,
        [["aspirin", "aspirin", "liver", "kidney"]],
    )
    result = evaluate_synonyms(tmp_path, toy_embedding)
    assert result["results"] == []
    assert np.isnan(result["micro"])


def test_synonym_duplicate_candidates_are_skipped(tmp_path, toy_embedding):
    # the same word offered twice means the question has two "right" cells
    write(
        tmp_path,
        "synonym",
        "dupe",
        SYN_HEADER,
        [["aspirin", "ibuprofen", "liver", "liver"]],
    )
    assert evaluate_synonyms(tmp_path, toy_embedding)["results"] == []


def test_synonym_micro_average_counts_a_shared_question_once(tmp_path, toy_embedding):
    row_right = ["aspirin", "ibuprofen", "liver", "kidney"]
    row_wrong = ["aspirin", "liver", "ibuprofen", "kidney"]
    write(tmp_path, "synonym", "a_first", SYN_HEADER, [row_right])
    write(tmp_path, "synonym", "b_second", SYN_HEADER, [row_right, row_wrong])
    result = evaluate_synonyms(tmp_path, toy_embedding)
    # de-duplication changes the *weights*, never the per-dataset scores: each
    # dataset still reports its own accuracy over its own rows
    scores = {r["name"]: r["score"] for r in result["results"]}
    assert scores == {"en_a_first": 1.0, "en_b_second": 0.5}
    # `b_second` re-uses the question `a_first` already contributed, so it adds
    # weight 1 (its second question), not 2 -- micro is (1*1.0 + 1*0.5)/2
    assert result["micro"] == pytest.approx(0.75)


def test_synonym_fully_redundant_dataset_is_dropped_from_the_averages(
    tmp_path, toy_embedding
):
    # every question of `b_second` was already counted, so it contributes nothing
    # to micro or macro -- but it is still reported individually, because a user
    # asked for that dataset's number
    row = ["aspirin", "ibuprofen", "liver", "kidney"]
    write(tmp_path, "synonym", "a_first", SYN_HEADER, [row])
    write(tmp_path, "synonym", "b_second", SYN_HEADER, [row])
    result = evaluate_synonyms(tmp_path, toy_embedding)
    assert len(result["results"]) == 2
    assert result["micro"] == pytest.approx(1.0)
    assert result["macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# category purity
# ---------------------------------------------------------------------------


def evaluate_categories(tmp_path, embedding):
    return evaluation.eval_categories(
        embedding,
        TOKEN2ID,
        identity_preproc,
        data_dir=tmp_path,
        include_bundled=False,
    )


def test_category_purity_is_one_for_separated_clusters(tmp_path, toy_embedding):
    write(
        tmp_path,
        "category",
        "clean",
        CAT_HEADER,
        [
            ["aspirin", "drug"],
            ["ibuprofen", "drug"],
            ["paracetamol", "drug"],
            ["liver", "organ"],
            ["kidney", "organ"],
            ["spleen", "organ"],
        ],
    )
    result = evaluate_categories(tmp_path, toy_embedding)["results"][0]
    assert result["score"] == 1.0
    # chance floor for two categories of three: 2 * 3*2 / (6*5)
    assert result["baseline"] == pytest.approx(0.4)


def test_category_purity_falls_when_labels_cross_the_clusters(tmp_path, toy_embedding):
    # same vectors, labels shuffled across the clusters: every nearest neighbour
    # is still a cluster-mate, but now carries the other label
    write(
        tmp_path,
        "category",
        "crossed",
        CAT_HEADER,
        [
            ["aspirin", "a"],
            ["ibuprofen", "b"],
            ["liver", "a"],
            ["kidney", "b"],
        ],
    )
    result = evaluate_categories(tmp_path, toy_embedding)["results"][0]
    assert result["score"] == 0.0
    assert result["baseline"] == pytest.approx(4 / 12)


def test_category_needs_two_categories(tmp_path, toy_embedding):
    # purity over a single category is trivially 1 and measures nothing
    write(
        tmp_path,
        "category",
        "single",
        CAT_HEADER,
        [["aspirin", "drug"], ["ibuprofen", "drug"]],
    )
    assert evaluate_categories(tmp_path, toy_embedding)["results"] == []


def test_category_oov_words_are_reported(tmp_path, toy_embedding):
    write(
        tmp_path,
        "category",
        "partial",
        CAT_HEADER,
        [
            ["aspirin", "drug"],
            ["ibuprofen", "drug"],
            ["liver", "organ"],
            ["missing", "organ"],
        ],
    )
    result = evaluate_categories(tmp_path, toy_embedding)["results"][0]
    assert result["oov"] == 0.25


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_dataset_coverage_handles_the_new_kinds(tmp_path):
    write(
        tmp_path,
        "synonym",
        "s",
        SYN_HEADER,
        [
            ["aspirin", "ibuprofen", "liver", "kidney"],
            ["aspirin", "ibuprofen", "liver", "missing"],
        ],
    )
    write(
        tmp_path, "category", "c", CAT_HEADER, [["liver", "organ"], ["nope", "organ"]]
    )
    syn = evaluation.dataset_coverage(
        TOKEN2ID,
        identity_preproc,
        kind="synonym",
        data_dir=tmp_path,
        include_bundled=False,
    )
    cat = evaluation.dataset_coverage(
        TOKEN2ID,
        identity_preproc,
        kind="category",
        data_dir=tmp_path,
        include_bundled=False,
    )
    assert syn[0]["coverage"] == 0.5
    # only the word column is looked up; the label is not a vocabulary entry
    assert cat[0]["coverage"] == 0.5


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------

GLOSSARY = [
    ("hypertension", "high-blood-pressure"),
    ("tachycardia", "palpitations"),
    ("dyspnea", "breathlessness"),
    ("edema", "swelling"),
    ("syncope", "fainting"),
]


def test_builder_never_offers_a_synonym_as_a_distractor():
    # the failure this guards against is silent: a "wrong" option that is in fact
    # right caps the achievable score below what the embedding deserves
    questions, _counts = builder.build_synonym(GLOSSARY, n_distractors=3, seed=0)
    synonym_pairs = {frozenset((a.casefold(), b.casefold())) for a, b in GLOSSARY}
    for target, _answer, *distractors in questions:
        for distractor in distractors:
            assert (
                frozenset((target.casefold(), distractor.casefold()))
                not in synonym_pairs
            )
            assert distractor.casefold() != target.casefold()


def test_builder_is_deterministic():
    first, _ = builder.build_synonym(GLOSSARY, n_distractors=3, seed=0)
    second, _ = builder.build_synonym(GLOSSARY, n_distractors=3, seed=0)
    assert first == second


def test_builder_seed_changes_the_draw():
    first, _ = builder.build_synonym(GLOSSARY, n_distractors=3, seed=0)
    other, _ = builder.build_synonym(GLOSSARY, n_distractors=3, seed=7)
    assert first != other


def test_builder_emits_a_file_the_evaluator_can_read(tmp_path):
    glossary = tmp_path / "gloss.tsv"
    glossary.write_text(
        "\n".join(f"{a}\t{b}" for a, b in GLOSSARY) + "\n", encoding="utf-8"
    )
    out = tmp_path / "data"
    builder.main(
        [
            "synonym",
            "--glossary",
            str(glossary),
            "--out",
            str(out),
            "--name",
            "gloss",
            "--distractors",
            "3",
        ]
    )
    path = out / "en" / "synonym" / "gloss.tsv"
    columns = list(evaluation.setup_test_tokens(path, kind="synonym"))
    assert len(columns) == 5
    # `high-blood-pressure` is one token under the v2 tokenizer, which is why it
    # survives at all -- v1 shattered it into three
    assert "high-blood-pressure" in columns[1]


def test_builder_drops_a_word_with_conflicting_categories():
    rows = [("liver", "organ"), ("liver", "drug"), ("aspirin", "drug")]
    kept, dropped = builder.build_category(rows)
    assert [row[0] for row in kept] == ["aspirin"]
    assert dropped["conflicting_category"] == 2


def test_builder_skips_multi_token_entries():
    rows = [("vice president", "role"), ("aspirin", "drug")]
    kept, dropped = builder.build_category(rows)
    assert [row[0] for row in kept] == ["aspirin"]
    assert dropped["multitoken"] == 1
