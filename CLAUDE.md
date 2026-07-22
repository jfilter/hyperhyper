# Working on `hyperhyper`

Count-based word embeddings (PPMI + SVD), a reimplementation of Levy, Goldberg &
Dagan (2015). `CONTRIBUTING.md` covers setup, the test layout and the release
procedure — read it first. This file covers the things that are **not** obvious
from the code and that have already been got wrong at least once.

Everything runs under `uv`: `uv run pytest`, `uv run ruff check .`.

## 1. Determinism is a contract, not a nice property

The single most important rule in this repository. Two runs with the same
arguments must produce the **same bits**, on any machine, at any core count.
Users key cached artifacts and published results on this.

What that means in practice:

- **`pair_counts.merge_order()` pins the float32 summation order.** float
  addition is not associative, so the order partial matrices are summed in is
  part of the answer. Do not make it depend on completion order, core count or
  chunk count.
- **Accumulate in float64, narrow once.** The Python loop accumulates into
  `defaultdict(int)` with Python floats (= float64) and only `to_count_matrix`
  casts to float32. Any rewrite that accumulates in float32 rounds at every
  addition instead of once, and the matrix differs in the low bits.
- **Emit pairs in the loop's order**: centre-major, context position ascending.
  `np.add.at` applies additions in index order, so this is observable.
- **Never swap `random.Random` for a numpy generator** in the counting path.
  This is the trap this codebase fell into twice *in reasoning* and never in
  code. A numpy RNG would be faster to draw from and would change every
  randomized result at the same seed, permanently. The vectorized counter
  instead **reproduces** the draw stream: one `randint(1, window)` per token in
  token order, one `random()` per subsample-eligible token, interleaved per
  sentence. See `CountPairsClosure._subsample_draws`.
- **Storage format must not reach the numbers.** The per-chunk RNG is seeded
  from the chunk's `Path(...).stem`. It used to use the full filename, so
  `texts_0.pkl` and `texts_0.npz` drew different numbers — migrating a corpus
  silently changed results without a token moving.

### The gate

`tests/test_pair_counts_equivalence.py` runs the live counter against a **frozen
snapshot** of the pre-vectorization code (`bench/reference.py`) and requires
`assert_array_equal` — deliberately not `assert_allclose`, which would wave
through exactly the drift the gate exists to catch.

Run it before and after any change to `hyperhyper/pair_counts.py`:

    uv run pytest tests/test_pair_counts_equivalence.py -m slow

`bench/reference.py` is frozen. Change it only for things that are *not
counting logic* (how a chunk is read off disk, for instance), and say so in its
header, which records every such change and why.

If a change cannot hit bit-identity, **that is a finding to raise, not a
tolerance to widen.**

## 2. Measure before you optimize, and distrust the measurement

Several long-standing beliefs in this repository turned out to be wrong when
finally measured, and each had a plausible story attached:

- the tokenization process pool was **5.6x slower** than serial, with a table
  proving it sitting directly above the constant that assumed otherwise;
- `subsample="prob"`/`"dirty"` were documented as "already the fastest
  configurations, not worth vectorizing". They were fastest *per surviving pair*
  and the **slowest per call** — the number a user actually waits on;
- the README claimed counting was ~8.6% of runtime; it was 28.2%.

So:

- Use the benchmarks: `bench/bench_pair_counts.py` (counting core, deliberately
  single-process — through the pool, spawn startup would hide everything),
  `bench/bench_svd.py` (backends, speed *and* fidelity).
- **This machine is noisy — ±35% on repeated identical runs.** A single
  before/after pair proves nothing, and back-to-back A-then-B runs drift upward
  and will manufacture a regression that is not there. Alternate the order and
  compare medians. A "15% regression" reproduced three times in one ordering
  disappeared entirely when the ordering was reversed.
- Quote `best`, not `mean`; report the noise level alongside the number.

## 3. How this repository writes things down

- **Superseded claims are marked, not deleted.** When a measurement overturns a
  documented claim, the old claim stays with a note saying it was wrong and why
  the reasoning was tempting. Several docstrings and CHANGELOG entries are
  structured this way on purpose — the wrong reasoning is the useful part.
- **Comments say *why*, not *what*.** The density is high and deliberate;
  match it. A comment that restates the line below it is noise here.
- **Tests carry their own justification.** Tolerances are derived from measured
  noise, not from magic constants, and the test docstrings explain the
  derivation. Vacuity guards exist (`test_delete_oov_actually_changes_the_matrix`)
  because a test that compares a matrix to itself passes happily.

## 4. Boundaries that are easy to cross by accident

- **`tools/` and `bench/` ship in the sdist but not in the wheel.** Maintainer
  scripts (dataset importers, task builders) belong there, never in
  `hyperhyper/`.
- **Bundled evaluation data is licence-checked per artifact.** `docs/adr/0001`
  records what was verified and where it was read. Do not add a dataset without
  that evidence, and do not infer a dataset's licence from the article that
  describes it — that specific mistake has been made here.
- **`allow_pickle=False` on every `np.load`.** Bunch directories are a *local
  cache*; the `.npz` files are data, not code. The one remaining pickle path
  (`read_pickle`, for pre-`.npz` chunks) is documented as trusted-input-only.
- **Corpus chunks: `.npz` is current, `.pkl` is legacy and still read.** The
  format is chosen by extension, never by sniffing.

## 5. Test suite shape

- Default run is the fast suite; `-m slow` adds the full parameter grids.
- CI runs lint, the fast suite on 3.10–3.13, `test-slow`, and a `floor` job
  against minimum dependency versions.
- The spaCy-dependent tests skip unless `en_core_web_sm` is installed. That is
  expected locally, not a failure.
