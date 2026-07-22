"""
The dataset linter -- the acceptance gate for every bundled evaluation set,
now and in the future (ADR 0001, decision 3).

Each bundled `ws/` and `analogy/` file is parametrized and checked for the
structural invariants a usable gold set must hold:

* the right number of whitespace-separated fields per line (3 for similarity,
  4 for analogy), ignoring `#`-comment provenance lines;
* similarity scores that actually parse as floats;
* no duplicate word pair within a file -- case-insensitive, and, for the
  symmetric similarity task, order-insensitive too (cosine is symmetric, so
  `a b` and `b a` are the same pair);
* no self-pairs (`w w`);
* every entry collapsing to exactly one token under the package's own
  preprocessing (`tokenize_string`), because the scorer can only ever look up a
  single token -- a hyphenated or multi-word entry is silently unscoreable;
* for analogy rows: the four tokens are distinct, and the answer `b_` is not one
  of the query words `a`, `a_`, `b` (such a row is unanswerable because the
  scorer excludes the query words), checked on the surface form before
  lemmatization;
* no exact-duplicate analogy rows.

It also pins the parser-safety property the provenance headers rely on: a
`#`-comment line must never be picked up by `setup_test_tokens` as a data row.
Since ADR 0002 (roadmap 1) `setup_test_tokens` has real `#`-comment support --
it skips comment lines by their leading `#` *before* the field-count filter --
so this holds even for a comment that splits into exactly the kept field count.
The linter asserts that the parser recovers exactly the intended data rows (no
leaked comment, no dropped malformed line) and, separately, that a comment which
does split into the field count is ignored rather than leaked.
"""

from importlib.resources import files

import pytest

from hyperhyper import evaluation, evaluation_datasets
from hyperhyper.preprocessing import tokenize_string

# kind -> number of whitespace fields `setup_test_tokens` keeps for that kind
KIND_FIELDS = {"ws": 3, "analogy": 4}


def _discover():
    """Every bundled `.txt` dataset, as `(lang, kind, n_fields, name)` tuples."""
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
                if p.name.endswith(".txt"):
                    found.append((lang, kind, n_fields, p.name))
    return sorted(found)


DATASETS = _discover()
DATASET_IDS = [f"{lang}/{kind}/{name}" for lang, kind, _n, name in DATASETS]


def _path(lang, kind, name):
    return files(evaluation_datasets).joinpath(lang).joinpath(kind).joinpath(name)


def _rows(lang, kind, name):
    """Non-comment data lines as `(lineno, fields)`; comment lines are those
    whose first non-space character is `#`."""
    text = _path(lang, kind, name).read_text(encoding="utf-8")
    rows = []
    for i, line in enumerate(text.split("\n"), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append((i, line.split()))
    return rows


def _single_token(entry):
    toks = tokenize_string(entry)
    return len(toks) == 1


# marks a param-tuple as expanded into the four fixture arguments below
_PARAMS = pytest.mark.parametrize("lang,kind,n_fields,name", DATASETS, ids=DATASET_IDS)


def test_datasets_were_discovered():
    # a broken glob that finds nothing would make every parametrized test below
    # vacuously pass; assert we are actually linting a realistic number of files
    assert len(DATASETS) >= 19


@_PARAMS
def test_field_count(lang, kind, n_fields, name):
    bad = [(ln, f) for ln, f in _rows(lang, kind, name) if len(f) != n_fields]
    assert not bad, f"{name}: {len(bad)} line(s) without {n_fields} fields: {bad[:5]}"


@_PARAMS
def test_parser_recovers_exactly_the_data_rows(lang, kind, n_fields, name):
    # Since ADR 0002 `setup_test_tokens` skips `#`-comment lines by their leading
    # `#` before the field-count filter, and warns-and-skips a malformed row.
    # For a clean bundled file the parser must therefore recover exactly the
    # non-comment, correctly-shaped data rows -- no more (a leaked comment) and
    # no fewer (a dropped data line).
    columns = list(evaluation.setup_test_tokens(_path(lang, kind, name), n_fields))
    parsed = len(columns[0]) if columns else 0
    assert parsed == len(_rows(lang, kind, name))


@_PARAMS
def test_comment_with_field_count_is_ignored(lang, kind, n_fields, name, tmp_path):
    # The old fragile invariant was "no comment line may split into exactly
    # `n_fields` fields", because the parser had no notion of comments. ADR 0002
    # retired it with real `#`-comment support. Prove the danger it guarded is
    # gone: prepend to a real bundled file a `#`-comment that DOES split into
    # exactly `n_fields` fields (`# a b` -> 3, `# a b c` -> 4). It must be
    # ignored, so the parser still recovers exactly the file's own data rows and
    # never leaks the comment as a `('#', 'a', ...)` row.
    original = _path(lang, kind, name).read_text(encoding="utf-8")
    leaky_comment = "# " + " ".join("abcdefg"[: n_fields - 1])
    assert len(leaky_comment.split()) == n_fields  # it really is a would-be leak
    spiked = tmp_path / name
    spiked.write_text(leaky_comment + "\n" + original, encoding="utf-8")

    columns = list(evaluation.setup_test_tokens(spiked, n_fields))
    parsed = len(columns[0]) if columns else 0
    assert parsed == len(_rows(lang, kind, name))
    # the comment's leading `#` never entered the data columns
    assert "#" not in columns[0]


@_PARAMS
def test_single_token_entries(lang, kind, n_fields, name):
    offenders = []
    for ln, f in _rows(lang, kind, name):
        for entry in f[: 2 if kind == "ws" else 4]:
            if not _single_token(entry):
                offenders.append((ln, entry, tokenize_string(entry)))
    assert not offenders, f"{name}: multi-token entries: {offenders[:5]}"


@_PARAMS
def test_no_self_pairs(lang, kind, n_fields, name):
    if kind == "ws":
        bad = [
            (ln, f) for ln, f in _rows(lang, kind, name) if f[0].lower() == f[1].lower()
        ]
        assert not bad, f"{name}: self-pairs (w w): {bad[:5]}"


def test_ws_scores_parse_as_float():
    for lang, kind, _n, name in DATASETS:
        if kind != "ws":
            continue
        for ln, f in _rows(lang, kind, name):
            try:
                float(f[2])
            except ValueError:
                pytest.fail(f"{name}:{ln}: score {f[2]!r} does not parse as float")


@_PARAMS
def test_no_duplicate_pairs(lang, kind, n_fields, name):
    seen = {}
    dups = []
    for ln, f in _rows(lang, kind, name):
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
    for ln, f in _rows(lang, kind, name):
        low = [w.lower() for w in f]
        if len(set(low)) != 4:
            not_distinct.append((ln, f))
        # columns are `a a_ b b_`; the scorer excludes {a, a_, b}, so a row whose
        # answer is one of them can never be scored correct
        if low[3] in (low[0], low[1], low[2]):
            answer_collapses.append((ln, f))
    assert not not_distinct, (
        f"{name}: rows without 4 distinct tokens: {not_distinct[:5]}"
    )
    assert not answer_collapses, (
        f"{name}: answer collapses onto a query word: {answer_collapses[:5]}"
    )
