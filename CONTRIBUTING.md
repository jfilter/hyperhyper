# Contributing to `hyperhyper`

Thanks for your interest in improving `hyperhyper`. This document explains how to
set the project up, how to run the test suite, and the one rule that is specific
to this codebase: changes to the counting / PMI / evaluation math can move the
numbers, and that has to be reasoned about rather than merely made green.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
# `--extra full` pulls in spaCy + scikit-learn; `--group dev` adds pytest and ruff
uv sync --extra full --group dev
```

The slow tests additionally need a spaCy language model. Download it once:

```bash
uv run python -m spacy download en_core_web_sm
```

## Running the tests

Tests are split with the `slow` marker (declared in `pyproject.toml`). Slow tests
are the ones that need the spaCy model or exercise the full parameter grid.

```bash
uv run pytest -m "not slow"   # fast suite, no spaCy model required
uv run pytest -m slow         # slow suite, needs en_core_web_sm
uv run pytest                 # everything
```

CI mirrors this split (`.github/workflows/ci.yml`): the Python matrix runs
`pytest -m "not slow"` without the model, and a separate job downloads the model
and runs `pytest -m slow`. There is also a `floor` job that installs the *oldest*
dependency versions allowed by `pyproject.toml` (`--resolution lowest-direct`), so
if you raise or add a dependency bound, make sure the real lower bound works.

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format --check .   # drop `--check` to apply formatting
```

CI runs both `ruff check .` and `ruff format --check .`; a formatting diff fails
the build.

## The equivalence gate for `pair_counts`

`hyperhyper/pair_counts.py` (the pair counting) has a dedicated correctness gate,
because it is the part of the package most likely to be rewritten for speed and
most likely to silently change results when it is.

- `bench/reference.py` holds a **frozen snapshot** of the counting code at a
  recorded git SHA. Do not tidy, modernize, or "fix" it — bugs are part of the
  snapshot, and its whole job is to detect *changes* against a fixed baseline. It
  must not import counting logic from `hyperhyper.pair_counts`, or it would be
  checking code against itself.
- `tests/test_pair_counts_equivalence.py` runs the live `count_pairs` and the
  frozen reference over the same corpus and arguments and compares the matrices.

The gate holds two configurations to two different standards:

- **Deterministic configs must stay bit-identical.** For
  `dynamic_window in (None, "deter", "decay")` and `subsample in (None, "deter")`
  no random number is drawn, so the output is a pure function of the corpus and
  the arguments. These are asserted with `np.testing.assert_array_equal` (not
  `assert_allclose`) — the point is to catch quiet numerical drift such as a
  different summation order or a float64 intermediate rounding differently. Note
  that float32 addition is not associative, so the merge order of the per-chunk
  partial matrices is part of the answer; it is pinned to `sorted(paths)` in
  `merge_order` precisely so the result does not depend on core count or worker
  scheduling.
- **Randomized configs (`"prob"`) only owe statistical equivalence.** A different
  RNG or draw order gives different numbers at the same seed, so these are checked
  by comparing first moments across many seeds against a tolerance derived from
  the measured Monte-Carlo noise. The one exactness they still owe is
  reproducibility: the same seed twice must give the identical matrix.

If a change to the counting is *meant* to alter output, that is a deliberate,
reviewed decision: update the snapshot in `bench/reference.py` together with it
and bump the SHA recorded in its header. Do not widen a tolerance to make a
bit-identity failure pass.

## Changes that move the numbers

Counting, PMI (`hyperhyper/pmi.py`), SVD (`hyperhyper/svd.py`) and evaluation
(`hyperhyper/evaluation.py`) all feed the scores this package reports. A change
in any of them can shift results — sometimes correctly (a bug fix), sometimes
subtly (a precision or ordering change). When you touch these:

- Understand *why* a result changed before accepting it. A test going from red to
  green is not evidence that the new number is the right one.
- Remember that results are cached per-parameter-set on disk (`Bunch` writes
  `.npz` files keyed by a digest of the effective arguments) and recorded in a
  SQLite `results.db`. If the *meaning* of a cache file changes, bump
  `CACHE_FORMAT` in `hyperhyper/bunch.py` so stale entries stop being served.
- User-visible changes belong in `CHANGELOG.md`, especially anything that changes
  numeric results or breaks an API.

## Releasing

Releases are published by pushing a tag; `.github/workflows/publish.yml` does
the rest. There is **no PyPI token** in this repository or its secrets — the
workflow authenticates with Trusted Publishing (OIDC), so GitHub mints a
short-lived credential scoped to this repository, this workflow file and the
`pypi` environment. Nothing long-lived exists to leak.

1. Move the `Unreleased` CHANGELOG entries under a new `## X.Y.Z - YYYY-MM-DD`
   heading, and say at the top of it if reported **numbers** move — that is the
   part users cannot infer from a diff.
2. Bump `version` in `pyproject.toml`.
3. Commit, then `git tag -a vX.Y.Z -m "..."` and `git push --follow-tags`.

The workflow builds once and publishes that exact artifact, refusing to go on if
the tag disagrees with the built version (a `v0.2.0` tag on a `0.1.1` build
would publish `0.1.1` and burn that number — PyPI never allows re-uploading a
version), if `twine check --strict` rejects the metadata, or if the installed
wheel cannot import and find its bundled evaluation data.

Give the `pypi` environment required reviewers in the repository settings. Then
a tag pushed by accident waits for an approval click instead of shipping.

## Questions and bugs

For questions, bug reports and feature proposals, use the
[issues page](https://github.com/jfilter/hyperhyper/issues). Pull requests are
especially welcome when they fix bugs or improve code quality.
