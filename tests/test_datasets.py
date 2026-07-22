"""
The dataset linter -- the acceptance gate for every bundled evaluation set,
now and in the future (ADR 0001, decision 3; retargeted at TSV by ADR 0002).

The bundled sets ship as strict UTF-8 TSV (`.tsv`): a `# key: value` provenance
preamble, a required header row (`word1 word2 score` for similarity,
`a a_prime b b_prime` for analogy), then tab-delimited data. Each file is
parametrized and checked for the structural invariants a usable gold set must
hold:

* a correct header row, with the preamble (and every `#` comment) appearing only
  *before* it -- after the header there is no comment convention, so a stray `#`
  line would be a malformed data row;
* the right number of tab-separated columns per data row (3 similarity, 4 analogy);
* similarity scores that parse as finite floats;
* no duplicate word pair within a file -- case-insensitive, and, for the
  symmetric similarity task, order-insensitive too (cosine is symmetric, so
  `a b` and `b a` are the same pair);
* no self-pairs (`w w`);
* every entry collapsing to exactly one token under the package's *current
  default* preprocessing (`tokenize_string_v2`, ADR 0002), because the scorer can
  only ever look up a single token -- a genuinely multi-word entry is unscoreable
  (TSV can now *represent* it, but the unigram evaluator still treats it as OOV).
  Linting against v2 rather than v1 is load-bearing: v1 shattered
  `narrow-mindedness` and `city's`, which is why curated-v2 had to drop ~1030
  otherwise-good rows; v2 keeps them whole and curated-v3 restored them, so a v1
  check here would reject correct data;
* for analogy rows: the four tokens are distinct, and the answer `b_prime` is not
  one of the query words `a`, `a_prime`, `b` (such a row is unanswerable because
  the scorer excludes the query words), checked on the surface form before
  lemmatization;
* no exact-duplicate analogy rows.

It also pins the format-migration guarantees of ADR 0002: no dataset ships in
both `.tsv` and `.txt` (that raises `DuplicateDatasetError`); a malformed `.tsv`
row *raises* `MalformedDatasetError` with file:line rather than being silently
dropped; and a legacy whitespace `.txt` still reads through the compatibility
path so users' existing files keep working.
"""

import csv
from importlib.resources import files

import pytest

from hyperhyper import evaluation, evaluation_datasets
from hyperhyper.evaluation import DuplicateDatasetError, MalformedDatasetError
from hyperhyper.preprocessing import tokenize_string_v2

# kind -> number of tab-separated columns `setup_test_tokens` keeps for that kind
KIND_FIELDS = {"ws": 3, "analogy": 4}
# kind -> the required TSV header row
KIND_HEADER = {
    "ws": ["word1", "word2", "score"],
    "analogy": ["a", "a_prime", "b", "b_prime"],
}


def _discover():
    """Every bundled `.tsv` dataset, as `(lang, kind, n_fields, name)` tuples."""
    root = files(evaluation_datasets)
    found = []
    # `fr` ships an analogy-only pack (ADR 0001, P2); a language need not have
    # every kind, so skip a `<lang>/<kind>` directory that is not present.
    for lang in ("en", "de", "fr"):
        for kind, n_fields in KIND_FIELDS.items():
            directory = root.joinpath(lang).joinpath(kind)
            if not directory.is_dir():
                continue
            for p in directory.iterdir():
                if p.name.endswith(".tsv"):
                    found.append((lang, kind, n_fields, p.name))
    return sorted(found)


DATASETS = _discover()
DATASET_IDS = [f"{lang}/{kind}/{name}" for lang, kind, _n, name in DATASETS]


def _path(lang, kind, name):
    return files(evaluation_datasets).joinpath(lang).joinpath(kind).joinpath(name)


def _find_header(lines):
    """Index of the header row: the first line that is neither blank nor `#`."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return i
    return None


def _data_rows(lang, kind, name):
    """
    Data rows as `(lineno, fields)`, parsed straight from the file with the
    standard-library TSV reader -- independent of the evaluation parser, so the
    linter validates structure on its own.
    """
    lines = _path(lang, kind, name).read_text(encoding="utf-8").split("\n")
    header_idx = _find_header(lines)
    rows = []
    for lineno, line in enumerate(lines[header_idx + 1 :], header_idx + 2):
        if not line.strip():
            continue
        rows.append((lineno, next(csv.reader([line], delimiter="\t"))))
    return rows


def _single_token(entry):
    toks = tokenize_string_v2(entry)
    return len(toks) == 1


# marks a param-tuple as expanded into the four fixture arguments below
_PARAMS = pytest.mark.parametrize("lang,kind,n_fields,name", DATASETS, ids=DATASET_IDS)


def test_datasets_were_discovered():
    # a broken glob that finds nothing would make every parametrized test below
    # vacuously pass; assert we are actually linting a realistic number of files
    assert len(DATASETS) >= 19


@_PARAMS
def test_header_row_is_correct(lang, kind, n_fields, name):
    lines = _path(lang, kind, name).read_text(encoding="utf-8").split("\n")
    header_idx = _find_header(lines)
    assert header_idx is not None, f"{name}: no header row found"
    header = next(csv.reader([lines[header_idx]], delimiter="\t"))
    assert header == KIND_HEADER[kind], (
        f"{name}: header {header!r} != required {KIND_HEADER[kind]!r}"
    )


@_PARAMS
def test_preamble_and_comments_only_precede_the_header(lang, kind, n_fields, name):
    # metadata/comments are allowed only in the preamble, before the header; a
    # `#` line after the header would be parsed as a malformed data row
    lines = _path(lang, kind, name).read_text(encoding="utf-8").split("\n")
    header_idx = _find_header(lines)
    leaked = [
        (i + 1, line)
        for i, line in enumerate(lines[header_idx + 1 :], header_idx + 1)
        if line.strip().startswith("#")
    ]
    assert not leaked, f"{name}: comment line(s) after the header row: {leaked[:5]}"


@_PARAMS
def test_field_count(lang, kind, n_fields, name):
    bad = [(ln, f) for ln, f in _data_rows(lang, kind, name) if len(f) != n_fields]
    assert not bad, f"{name}: {len(bad)} row(s) without {n_fields} columns: {bad[:5]}"


@_PARAMS
def test_parser_recovers_exactly_the_data_rows(lang, kind, n_fields, name):
    # the strict TSV reader must recover exactly the file's own data rows: the
    # preamble is skipped, the header is consumed, and every non-blank record is
    # kept (a clean bundled file has no malformed row for it to raise on)
    columns = list(evaluation.setup_test_tokens(_path(lang, kind, name), n_fields))
    parsed = len(columns[0]) if columns else 0
    assert parsed == len(_data_rows(lang, kind, name))


@_PARAMS
def test_single_token_entries(lang, kind, n_fields, name):
    offenders = []
    for ln, f in _data_rows(lang, kind, name):
        for entry in f[: 2 if kind == "ws" else 4]:
            if not _single_token(entry):
                offenders.append((ln, entry, tokenize_string_v2(entry)))
    assert not offenders, f"{name}: multi-token entries: {offenders[:5]}"


@_PARAMS
def test_no_self_pairs(lang, kind, n_fields, name):
    if kind == "ws":
        bad = [
            (ln, f)
            for ln, f in _data_rows(lang, kind, name)
            if f[0].lower() == f[1].lower()
        ]
        assert not bad, f"{name}: self-pairs (w w): {bad[:5]}"


def test_ws_scores_parse_as_float():
    for lang, kind, _n, name in DATASETS:
        if kind != "ws":
            continue
        for ln, f in _data_rows(lang, kind, name):
            try:
                value = float(f[2])
            except ValueError:
                pytest.fail(f"{name}:{ln}: score {f[2]!r} does not parse as float")
            assert value == value and value not in (float("inf"), float("-inf")), (
                f"{name}:{ln}: score {f[2]!r} is not finite"
            )


@_PARAMS
def test_no_duplicate_pairs(lang, kind, n_fields, name):
    seen = {}
    dups = []
    for ln, f in _data_rows(lang, kind, name):
        if kind == "ws":
            # similarity is symmetric: treat the pair as unordered
            key = frozenset((f[0].lower(), f[1].lower()))
        else:
            # analogy rows are ordered quadruples; an exact repeat is a dup
            key = tuple(w.lower() for w in f)
        if key in seen:
            dups.append((seen[key], ln, f))
        else:
            seen[key] = ln
    assert not dups, f"{name}: duplicate pairs/rows: {dups[:5]}"


@_PARAMS
def test_analogy_row_shape(lang, kind, n_fields, name):
    if kind != "analogy":
        return
    not_distinct = []
    answer_collapses = []
    for ln, f in _data_rows(lang, kind, name):
        low = [w.lower() for w in f]
        if len(set(low)) != 4:
            not_distinct.append((ln, f))
        # columns are `a a_prime b b_prime`; the scorer excludes {a, a_prime, b},
        # so a row whose answer is one of them can never be scored correct
        if low[3] in (low[0], low[1], low[2]):
            answer_collapses.append((ln, f))
    assert not not_distinct, (
        f"{name}: rows without 4 distinct tokens: {not_distinct[:5]}"
    )
    assert not answer_collapses, (
        f"{name}: answer collapses onto a query word: {answer_collapses[:5]}"
    )


# ADR 0002 format-migration guarantees


def test_no_dataset_ships_in_both_formats():
    # discovery raises rather than scoring a dataset present as both foo.tsv and
    # foo.txt; assert every bundled `<lang>/<kind>` directory is clean
    root = files(evaluation_datasets)
    for lang in ("en", "de", "fr"):
        for kind in KIND_FIELDS:
            directory = root.joinpath(lang).joinpath(kind)
            if directory.is_dir():
                # would raise DuplicateDatasetError on a .tsv/.txt collision
                evaluation._dataset_files(directory)


def test_duplicate_tsv_and_txt_raises(tmp_path):
    (tmp_path / "foo.tsv").write_text(
        "word1\tword2\tscore\nathens\tgreece\t9\n", encoding="utf-8"
    )
    (tmp_path / "foo.txt").write_text("athens greece 9\n", encoding="utf-8")
    with pytest.raises(DuplicateDatasetError, match="foo"):
        evaluation._dataset_files(tmp_path)


@_PARAMS
def test_malformed_tsv_row_raises_with_file_and_line(
    lang, kind, n_fields, name, tmp_path
):
    # strict parsing is a main reason for the migration: append a row with the
    # wrong column count to a real bundled file and confirm it raises, naming the
    # file and 1-based line number, instead of being silently dropped
    original = _path(lang, kind, name).read_text(encoding="utf-8")
    broken = "\t".join(["x"] * (n_fields - 1))  # one column short
    spiked = tmp_path / name
    spiked.write_text(original + broken + "\n", encoding="utf-8")
    with pytest.raises(MalformedDatasetError) as exc:
        list(evaluation.setup_test_tokens(spiked, n_fields))
    assert spiked.name in str(exc.value)


def test_legacy_txt_still_reads_via_compat_path(tmp_path):
    # a user's existing whitespace `.txt` must keep working (chosen by extension):
    # blank/comment lines skipped, malformed rows warned-and-skipped, the rest
    # recovered exactly
    legacy = tmp_path / "legacy.txt"
    legacy.write_text(
        "# a provenance header\nathens greece 9\ngreece iraq 5\n",
        encoding="utf-8",
    )
    columns = list(evaluation.setup_test_tokens(legacy, 3))
    assert columns == [("athens", "greece"), ("greece", "iraq"), ("9", "5")]
