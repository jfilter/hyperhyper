# Working on `hyperhyper`

Count-based word embeddings (PPMI + SVD), a reimplementation of Levy, Goldberg &
Dagan (2015). `CONTRIBUTING.md` covers setup, test layout and releases. Everything
runs under `uv` (`uv run pytest`, `uv run ruff check .`).

**`uv.lock` records the project's own version.** After bumping `version` in
`pyproject.toml`, run `uv lock` and commit it. Locally invisible (`uv run`
re-locks silently), but CI uses `--locked` and every job fails — this has
broken a release tag once.

## Determinism is a contract

Same arguments must produce the **same bits** on any machine at any core count —
users key cached artifacts and published results on this.

- `pair_counts.merge_order()` pins the float32 summation order; it must never
  depend on completion order, core count or chunk count.
- Accumulate in float64, cast to float32 once (`to_count_matrix`). Accumulating
  in float32 rounds at every addition and changes the low bits.
- Emit pairs centre-major, context position ascending — `np.add.at` applies
  additions in index order, so this is observable.
- **Never swap `random.Random` for a numpy generator** in the counting path; it
  would change every randomized result at the same seed. The vectorized counter
  *reproduces* the exact draw stream (see `CountPairsClosure._subsample_draws`).
- Per-chunk RNG is seeded from `Path(...).stem`, not the full filename — the
  full name once made `.pkl` vs `.npz` chunks draw different numbers.

**The gate:** `tests/test_pair_counts_equivalence.py` compares the live counter
bit-for-bit (`assert_array_equal`, deliberately not `allclose`) against the
frozen snapshot `bench/reference.py`. Run it before and after touching
`hyperhyper/pair_counts.py`:

    uv run pytest tests/test_pair_counts_equivalence.py -m slow

`bench/reference.py` may only change for non-counting concerns (record why in
its header). If a change cannot hit bit-identity, that is a finding to raise,
not a tolerance to widen.

## Measure before you optimize

Several documented performance beliefs here were wrong when measured (e.g. the
tokenization pool was 5.6x *slower* than serial). Use `bench/bench_pair_counts.py`
and `bench/bench_svd.py`. **This machine is ±35% noisy**: alternate A/B ordering
and compare medians — back-to-back same-order runs manufacture regressions.
Quote `best` and report the noise level.

## Conventions and boundaries

- Superseded claims are marked, not deleted — the wrong reasoning is kept on
  purpose. Comments say *why*, not *what*; tests document how their tolerances
  were derived.
- `tools/` and `bench/` ship in the sdist but not the wheel; maintainer scripts
  belong there, never in `hyperhyper/`.
- Bundled evaluation data needs per-artifact licence evidence (`docs/adr/0001`).
  Never infer a dataset's licence from the article describing it.
- `allow_pickle=False` on every `np.load`; the one pickle path (`read_pickle`,
  legacy `.pkl` chunks) is trusted-input-only. Format is chosen by extension,
  never sniffed.
- Fast suite by default; `-m slow` adds full grids. spaCy tests skip without
  `en_core_web_sm` — expected locally.
