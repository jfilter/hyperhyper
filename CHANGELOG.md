# Changelog

## 0.3.0 - 2026-07-22

A performance release: pair counting is now fully vectorized, corpus chunks are
data rather than pickles, and the SVD backend choice is measured instead of
guessed.

> **Reported numbers move in exactly one place.** Randomized configurations —
> `dynamic_window="prob"`, `subsample="prob"`, `subsample="dirty"` — produce
> different (equally valid) matrices than 0.2.0 at the same seed, because the
> per-chunk RNG seed no longer includes the file extension. **The defaults are
> bit-identical**: `dynamic_window="deter"` and `subsample="deter"` draw no
> random numbers at all. Everything else in this release, the vectorization
> included, is bit-identical by construction and by test — including the
> randomized paths, which reproduce their draw streams rather than replacing
> them.

### Changed

-   **Corpus chunks are stored as `.npz`, not pickles.** A chunk is now a flat id
    array plus sentence offsets (`utils.IdChunk`) instead of a pickled list of
    `array('H')`. Three reasons: it loads **6x faster** (10.7ms to 1.7ms for a
    400k-token chunk, ~19% off the counting stage end to end), it is slightly
    smaller, and it is **data rather than code** — 40 of the 41 files in a bunch
    directory no longer execute anything on load. It is also the exact layout the
    vectorized counter builds for itself, so it now skips flattening entirely.

    Bunches written before this keep working: the loader dispatches on the file
    extension, never by sniffing, and pickled chunks still read. This is tested by
    rebuilding a legacy bunch and requiring an identical matrix.

-   **The per-chunk RNG seed no longer includes the file extension.** It was
    derived from the chunk's full filename, so `texts_0.pkl` and `texts_0.npz`
    drew *different numbers* — the on-disk **format was part of the answer**, and
    migrating a corpus would have silently changed every randomized result
    without a single token moving. It now uses the stem.

    > **This moves results for randomized configurations once:**
    > `dynamic_window="prob"`, `subsample="prob"` and `subsample="dirty"` produce
    > different (equally valid) matrices than in 0.2.0 for the same seed. **The
    > defaults are unaffected** — `dynamic_window="deter"` and
    > `subsample="deter"` draw no random numbers at all, so every default result
    > is bit-identical. Same-version results remain reproducible as before.

### Added

-   **`dynamic_window="prob"` is vectorized too, and is now bit-identical.**
    2.6x faster (0.910s to 0.355s on 200k tokens at `window=5`) — and, unlike a
    numpy-RNG rewrite would have been, it reproduces the *exact* matrices this
    configuration produced before, so results recorded with it still hold.

    The equivalence gate previously stated that bit-identity was "not achievable"
    for any randomized configuration. That assumed the vectorization would draw
    from a numpy generator up front; it does not have to. `iterate_tokens` draws
    one `randint(1, window)` per token in token order, and a comprehension over
    the flattened chunk draws the same numbers in the same order. The gate now
    holds this configuration to `assert_array_equal` across every window and
    seed, and the docstring records why the old claim was wrong.

-   **`subsample="prob"`/`"dirty"` are vectorized too, and are also
    bit-identical.** ~3x faster (1.19s to 0.39s for clean, 1.39s to 0.42s for
    dirty, on 720k tokens at `window=5`); with a randomized window on top of the
    subsampling, 2.1x. Every configuration now takes the vectorized path.

    This entry supersedes the paragraph above, which said these two "deliberately
    stay on the Python loop" because they were already the fastest configurations
    and vectorizing them "would optimize the cheap case". That was measured, and
    it was wrong in the way benchmarks usually are: they were fastest *per
    surviving pair* and the slowest *per call*, which is the number a user waits
    on. At `window=5` they were the two slowest rows in the table.

    The real obstacle was the one the equivalence gate named: the two draw
    streams **interleave**. Within a sentence, `iterate_tokens` draws one
    `random()` per subsample-eligible token, and only then, knowing which tokens
    survived, one `randint(1, window)` per survivor — so neither stream can be
    drawn in a single pass over the chunk. `_subsample_draws` walks that
    interleaving sentence by sentence, in Python, and leaves only the *emission*
    to numpy. That is the right split: the draws were the cheap half all along.

    Because the streams are reproduced rather than replaced, `subsample="prob"`
    is now held to `assert_array_equal` against the frozen f68cc74 reference —
    across all 4 windows, all 4 `dynamic_window` modes and both seeds, both knobs
    randomized at once included. The equivalence grid grew from 48 exact cells to
    96, and the statistical comparison for `"prob"` was **removed**: bit-identity
    implies equality of every moment, so keeping it would have cost 10 runs per
    cell to prove less. `"dirty"` keeps its statistical check (the frozen
    reference predates the mode and cannot produce it) and gains an exact one
    against this package's own loop.

### Fixed

-   **An oversized chunk no longer changes randomized results.** A chunk above
    `MAX_VECTORIZED_EVENTS` falls back to the Python loop — but the vectorized
    path had already drawn from the RNG before discovering it was too large, so
    the loop resumed from an advanced generator. A corpus would have counted
    differently depending only on how it happened to be chunked, which is exactly
    what that memory cap must never do. `count_texts` now snapshots and restores
    the RNG state around the attempt. Present since `dynamic_window="prob"` was
    vectorized (unreleased), and found by extending the fallback test to the
    randomized modes.

-   **`bench/bench_svd.py` — which SVD backend to use, measured.** The package
    offered three (`scipy` exact, `gensim` and `scikit` randomized) and never said
    which to pick. It reports speed *and* fidelity in the terms the package
    actually consumes: agreement of the resulting cosine similarities and of the
    top-10 nearest neighbours with the exact backend, rather than raw singular
    vectors (which are only defined up to sign and rotation, so comparing them
    would be misleading).

    The finding: the randomized backends are **not a free speedup**. On a
    5001x5001 PPMI matrix, `scikit` runs 2.4-3.7x faster but shares only 0.69-0.86
    of the exact backend's top-10 neighbours; `gensim` is 1.2-1.7x faster at
    0.59-0.82. `scipy` therefore stays the default, and `gensim` is *dominated* by
    `scikit`, which is both faster and more accurate at every dimension tested.
    `docs/usage.md` now says so under `impl`.

-   **...and a second table for the accuracy knobs, which is the one that
    settles it.** The comparison above measures the backends at their *default*
    settings, which is a point on a curve, not the trade itself. Randomized SVD
    buys fidelity with power iterations and oversampling; both were already
    reachable through `impl_args` and neither had ever been measured. At
    `dim=300`, `scikit` walks from 3.4x speed / 0.76 neighbour overlap
    (`n_iter=4`) through 1.4x / 0.94 (`n_iter=10, n_oversamples=100`) to
    0.5x / 1.00 (`n_iter=20, n_oversamples=200`).

    Read the ends together: a randomized backend *can* give a near-exact answer,
    but the setting that gets there **costs more than computing the exact one**.
    The whole usable range is the narrow band in between — roughly a 10%
    end-to-end saving for a result that is close but not equal. `scipy` stays the
    default on that basis rather than on a hunch, and `gensim` turns out to be
    dominated across the entire curve rather than only at its defaults: matched
    for fidelity it is 3-4x slower than `scikit` and 3x slower than the exact
    backend it approximates.

## 0.2.0 - 2026-07-22

Modernization of the package for current Python and dependency versions, plus
the bug fixes and evaluation-data work that came out of it. Everything below is
user-visible.

> **Reported numbers move in this release, and that is the point.** Several of
> the fixes below were wrong *answers*, not slow ones: word-analogy accuracy was
> structurally 0.0, word-similarity gold scores were ranked as strings, the
> cache key ignored `**kwargs` so results depended on call order, and
> `subsample="prob"` used word2vec's *discard* probability as a *keep*
> probability. The evaluation data was also cleaned and extended. **Do not
> compare a score from 0.1.x with a score from 0.2.0** -- they are not measuring
> the same thing. Scores computed with the same version remain reproducible, and
> the tokenizer identity is now recorded with every result so old and new
> numbers cannot silently collide.

### Added

-   **Pair counting is vectorized on the deterministic paths** — 3.6-4.4x on the
    counting core, and **bit-identical** to the previous matrices. Configurations
    that draw no random numbers (`dynamic_window` of `None`/`"deter"`/`"decay"`
    with `subsample` of `None`/`"off"`/`"deter"`) now emit their (word, context,
    weight) events for a whole chunk with numpy instead of a per-token Python
    loop. On a 1.5M-token corpus, counting drops from 3.60s to 1.41s and a full
    run from 10.04s to 8.94s — 12.41s to 8.94s (**28%**) against where this round
    started.

    Bit-identity is not an accident and rests on two things: the accumulation
    stays in **float64** and narrows to float32 exactly once at the end (the
    Python loop accumulated into Python floats, which *are* float64), and the
    events are emitted in the loop's own order, because float addition is not
    associative and the order of additions into a cell is part of the answer.
    Decay weights come from the same scalar `decay()` via a lookup table rather
    than `np.exp`, which may differ in the last bit.

    The randomized modes (`"prob"`, `"dirty"`, `dynamic_window="prob"`) stay on
    the Python loop: their per-token draw order is a contract, and they emit far
    fewer pairs anyway. A chunk whose event arrays would not fit
    (`MAX_VECTORIZED_EVENTS`) also falls back — a memory decision that is
    invisible in the result, and tested to be.

-   **Tokenization no longer starts a process pool that makes it slower.** The
    pool was gated on a fixed `PARALLEL_MIN_CHARS = 2_000_000`, but the
    measurements recorded next to that constant show the pool losing at *every*
    size tested, up to 163M characters -- so every corpus above a couple of
    megabytes paid ~3s of spawn startup to tokenize more slowly. Re-measured with
    the v2 tokenizer: 6.6M chars 0.42s serial vs 2.91s pooled; 52.5M chars 3.72s
    vs 5.07s. The decision is now *measured* (sample, extrapolate, require a
    margin) the same way `count_pairs` already decides, so it self-calibrates
    instead of being quietly wrong. On a 1.5M-token corpus this cuts corpus
    construction from 4.05s to 1.73s and a full corpus->count->PMI->SVD->evaluate
    run from 12.41s to 10.04s (**19% end to end**).

    Scheduling never affected the result -- `map_pool` preserves order and the
    tokenizer is pure -- so no score or matrix changes.

-   **A dead worker now explains itself.** A script without an
    `if __name__ == "__main__":` guard makes every spawned worker re-run the
    script from the top — including the `hyperhyper` call that started the pool —
    so the pool recurses and collapses. The stdlib reports this as "A process in
    the process pool was terminated abruptly", which names a symptom and gives
    the reader nothing to act on. Both pools (tokenization and pair counting) now
    raise `hyperhyper.utils.BrokenProcessPool` with the cause, the fix, and a
    note that a notebook or REPL is unaffected. The original exception is
    chained.

-   **The pair-counting benchmark works again.** `bench/bench_pair_counts.py`
    kept a hand-written copy of `count_pairs`' argument translation, and the copy
    drifted: when the `dirty` subsampling variant landed, the benchmark was not
    updated and every run died with `AttributeError: ... has no attribute
    'subsampler_dirty'`. Both now call one shared
    `pair_counts.make_count_closure`, and `CountPairsClosure` names its fields
    explicitly instead of swallowing `**kwargs`, so an out-of-date construction
    site fails at construction with the missing argument named rather than in the
    middle of counting. A benchmark that cannot run is worse than none — it looks
    like a safety net.

-   **Cache files are written atomically.** Every artifact in a bunch directory —
    the count and PMI matrices, the SVD arrays, the corpus and text-chunk pickles —
    is now written to a sibling temporary file and renamed into place. An
    interrupted run (Ctrl-C, a full disk, an OOM kill partway through a long SVD)
    previously left a **truncated file that still looked like a valid cache
    entry**, so the next run found it and failed inside numpy/scipy instead of
    rebuilding. The destination is now either absent or complete, and a failed
    write leaves no temporary litter.

-   **`allow_pickle=False` is explicit on every numpy load**, and the bunch
    directory is documented as a **trusted local cache — never open one from an
    untrusted source**. `corpus.pkl` and the text chunks are pickles, so
    unpickling executes code by design; the `.npz` matrices now cannot. This is
    the honest statement of a format property, not a patched vulnerability
    (ADR 0002, roadmap item 7).

-   **Domain proxy evaluation tasks** (ADR 0001, P4) — the answer for corpora the
    general-language benchmarks cannot score. Two new dataset kinds, both with gold
    that is a *membership fact* rather than a rating, so they can be built without
    anyone judging anything:

    -   `bunch.eval_synonym(embd, data_dir=...)` — synonym multiple choice
        (`target answer distractor1 … distractorK`), gold from a glossary.
    -   `bunch.eval_category(embd, data_dir=...)` — category purity
        (`word category`), gold from a taxonomy, scored by nearest-neighbour purity.

    `dataset_coverage(kind=...)` covers both. Synonym files declare their own width
    through their header row, which the TSV format (ADR 0002) made possible.

    **No dataset of either kind is bundled**, deliberately: a general-language
    synonym set would recreate the exact problem these solve. Build them from your
    own glossary with `tools/build_domain_tasks/`.

    Read the scores against their **chance floor, not 0**: synonym accuracy floors
    at `1/(K+1)`, and every `eval_category` result carries a `baseline` next to its
    `score` because purity's floor depends on the category sizes.

-   **Swedish and Danish word-similarity datasets** (ADR 0001, P3):
    `sv/ws/supersim_similarity` and `sv/ws/supersim_relatedness` (1280 pairs each,
    from SuperSim, CC BY 4.0) and `da/ws/ws353` (316 pairs, Danish WordSim-353,
    CC-BY). `lang="sv"` and `lang="da"` now have bundled evaluation data. Each file
    names its source URL, source SHA-256, license, *where the license statement was
    read*, and citation in its own `#` preamble, and counts every dropped row.
    Scores are the published aggregates — nothing was recomputed or rescaled.

    Two caveats, stated rather than buried: the Danish scores are the **original
    English** human ratings carried over with translated word pairs, not ratings
    re-elicited from Danish speakers; and coverage stops here because of
    **licensing**, not effort — most published similarity sets state no license at
    all, and only sets whose permissive license is evidenced on the artifact
    itself are bundled. Use `data_dir` for anything else. See the P3 addendum in
    ADR 0001, and `tools/import_eval_data/` for the importer.

-   **curated-v3 restores 1030 evaluation rows** the old tokenizer had made
    unscoreable (`de/ws/gur350` +10, `en/ws/luong_rare` +20, `en/analogy/msr`
    +1000). Those rows were dropped in curated-v2 not because the data was bad but
    because v1 shattered `narrow-mindedness` and `city's` into two tokens; v2 keeps
    them whole, so `gur350` and `msr` are back at their original upstream sizes.
    No gold score or answer changed — the rows are verbatim from upstream.
    curated-v2's genuine data-quality drops (conflicting duplicates, self-pairs,
    exact duplicates, answer-collapse rows) remain dropped. This *increases*
    evaluation coverage and therefore moves the reported numbers again.

-   **A modern default tokenizer, `tokenize_string_v2`** (ADR 0002). NFC-normalizes
    first (so a decomposed `café` no longer loses its accent and splits into a
    second vocab entry), canonicalizes curly apostrophes and Unicode hyphens, and
    *extracts* words instead of destroying punctuation — `city's`, `ice-cream`,
    `don't` stay whole where the old tokenizer shattered them. Digits are kept by
    default (`normalize_digits=True` restores the Levy-Goldberg-Dagan digit→`0`
    convention). It lands under a new name; the old `tokenize_string` is unchanged,
    so existing bunches (which pickle their tokenizer by reference) are bit-for-bit
    unaffected.
-   **`preproc_func` is documented as the tokenizer plug-in point** — a picklable
    top-level `Callable[[list[str]], list[list[str]]]`, applied consistently to the
    corpus and the evaluation test words; the tokenizer's identity is now recorded
    with each result so v1 and v2 numbers are attributable and never collide.
-   Evaluation files gained a **real `#`-comment convention** and word2vec `:`
    section-header support; a malformed data row now emits a warning naming the
    file and line instead of vanishing silently.

-   **French analogy dataset** (`evaluation_datasets/fr/analogy/capitals.txt`) —
    the first new language, and the first generated dataset (ADR 0001 phase P2).
    315 capital→country analogies from 63 base pairs, each fact independently
    verified against Wikipedia's national-capitals list; LLM-proposed then
    human-web-verified, never an LLM dump. Generated by the offline pipeline in
    `bench/datagen/` (stdlib only, no runtime dependency); rebuilding is
    byte-identical. `eval_analogy(lang="fr")` now works — no registration needed,
    a language is just a data directory.

Evaluation data and tooling (ADR 0001, phases P0/P1):

-   **Evaluate on your own datasets.** `eval_similarity` / `eval_analogies` (and
    `Bunch.eval_sim` / `Bunch.eval_analogy`) take an optional `data_dir` pointing
    at your own `ws/` / `analogy/` files in the same format, evaluated alongside
    the bundled sets (`include_bundled=False` to use only yours). The biggest
    lever for the small-domain corpora this package targets.
-   **Coverage report.** `Bunch.dataset_coverage(kind=...)` /
    `evaluation.dataset_coverage(...)` report, per dataset, the fraction of rows
    fully in-vocabulary under the evaluator's own preprocessing — so you can tell
    before training which sets are usable for your corpus.
-   **A dataset linter** (`tests/test_datasets.py`) that gates every bundled file
    (and any future one) on field count, parseable scores, no duplicate/self
    pairs, single-token entries, and answerable analogy rows.

Three hyperparameters from Levy, Goldberg & Dagan (2015) that were missing on
the count side. All default to the previous behaviour, so existing results and
caches for the old settings are unchanged.

-   **3CosMul analogy objective.** `bunch.eval_analogy(embd, objective="mul")`
    and `most_similar_vectors(..., objective="mul")` on both `PPMIEmbedding` and
    `SVDEmbedding` implement the multiplicative objective from Levy & Goldberg
    (2014), `∏ cos(d, positive) / (∏ cos(d, negative) + 0.01)`, matching the
    `hyperwords` reference (including its dense-cosine remap to `[0, 1]`, which
    PPMI's already-non-negative cosines skip). Default stays `objective="add"`.
-   **`w+c` representation for SVD.** `bunch.svd(add_context=True)` builds the
    embedding as `U·Σ^eig + V·Σ^eig` instead of `U·Σ^eig` alone — the paper's
    context-vector-addition post-processing. `calc_svd` now retains the right
    singular vectors (it discarded them before). Default `add_context=False`;
    not offered for PPMI, which the paper excludes.
-   **Dirty subsampling.** `subsample="dirty"` removes subsampled tokens *before*
    building windows, so the window closes up and reaches further — the variant
    the paper actually reports. The existing `subsample="prob"` is the *clean*
    variant (the slot is kept) and is unchanged.

### Breaking

-   **Python `>=3.10,<3.14` is now required.** Python 3.6-3.9 are dropped. 3.14 is
    explicitly unsupported: no cp314 gensim wheel exists at any version, and the
    sdist does not compile against CPython 3.14 (gensim's vendored Cython
    dereferences `PyDictObject->ma_version_tag`, which was removed).
-   **`gensim>=4,<5` is now required.** `to_keyed_vectors` returns a `KeyedVectors`
    rather than a `WordEmbeddingsKeyedVectors`. Downstream code using gensim-3
    APIs (`.vocab`, `.add`) breaks.
-   **KeyedVectors are float32, not float64.** gensim 4's `KeyedVectors` defaults to
    float32, so similarity scores shift in the last digits.
-   **`spacy>=3,<4` is now required** (with an explicit `click>=8.1` in the `full`
    extra, without which `import spacy` fails on current typer releases).
-   **The spaCy pipeline is loaded with `exclude=["ner"]` instead of `disable=`.**
    The NER component is no longer present in the pipeline at all.
-   **`impl="sparsesvd"` removed** from `Bunch.svd()`, `Bunch.svd_matrix()` and
    `svd.calc_svd()`. It was already unusable — `sparsesvd`'s last release is from
    2013, is sdist-only and no longer builds. It now raises a clear `ValueError`
    instead of `NameError`.
-   **`low_memory` / `low_memory_chunk` kwargs removed** from `count_pairs`. The
    dense `(V+1)x(V+1)` outer product they existed to work around is gone;
    subsampling is now applied via sparse diagonal scaling, which is exact and
    O(nnz).
-   **`evaluation.read_test_data(lang, type)` is now `read_test_data(lang, kind)`**
    and returns a `list` of `importlib.resources` `Traversable`s rather than a
    generator of `pathlib.Path`s. Positional callers are unaffected; keyword
    callers break, as does code calling `pathlib`-only methods on the results.
    `.read_text()`, `.name` and `.suffix` all still work.
-   **Other keyword-argument renames** (all shadowed a builtin; only the first is
    a plausible keyword call in the wild): `Corpus.texts_to_file(dir=)` →
    `directory=`, `Bunch.dict_to_path(dict=)` → `params=`,
    `experiment.flatten_dict(dict=)` → `mapping=`, `utils.chunks(l=)` → `seq=`.
-   **`Bunch.pmi(keyed_vectors=True)` now returns `(vectors, results)`** instead
    of a bare `KeyedVectors`, matching `Bunch.svd(keyed_vectors=True)`. With the
    default `evaluate=True` the old form raised `IndexError`, so only
    `evaluate=False` callers are affected; they now get a one-element unpack.
-   **`subsample="prob"` produces different (and now correct) results** — see
    "Fixed". Anything trained with `subsample="prob"` needs re-running.
-   **`texts_to_sents` gained an `n_process` parameter, defaulting to 1.**
    Multi-process spaCy piping is now opt-in and must only be enabled by callers
    that are not already inside a process pool and whose module is protected by
    an `if __name__ == "__main__"` guard.
-   **Analogy evaluation results change from a constant 0.0 to real numbers**, and
    `"score"` changes from a raw hit count to an accuracy ratio. Stored
    `results.db` rows and plots comparing old and new analogy scores are not
    comparable. See "Fixed" below.
-   **Invalid arguments now raise instead of being silently ignored under `-O`.**
    Argument validation moved from `assert` to explicit `raise`. In particular
    `dynamic_window=0` now raises where it previously skipped validation (`0 ==
    False` made the old guard truthy).
-   **`save_matrix` writes `scipy.sparse.save_npz` format.** `load_matrix` still
    reads the old hand-rolled layout, so existing caches keep working.
-   **`from hyperhyper import *` no longer exports `logging`, `NullHandler` or the
    submodules**, now that `__init__.py` defines `__all__`.
-   `Bunch.pmi` / `Bunch.svd` now carry their real `__name__`, `__doc__` and
    signature (the `record` decorator uses `functools.wraps`). Only breaks code
    introspecting the former `wrapper`.
-   **`PPMIEmbedding.most_similar()` and `SVDEmbedding.most_similar()` now return
    `[(index, score), ...]`** instead of `[(score, index), ...]`, matching
    `most_similar_vectors()` in both classes. Indices are plain `int` and scores
    plain `float`. Callers unpacking the old order get silently transposed
    results — this needs a code change, not a config change.
-   **The subsampling threshold is now `subsample_factor x total token count`**
    (word2vec's `sample * train_words`) instead of `x sentence count`. Every
    embedding trained with `subsample="deter"` or `"prob"` changes. At 1M
    sentences / 20M tokens with the default `1e-5`, the threshold goes from 10 to
    200; rare words previously dragged into an `f^-1/2` reweighting are now left
    alone, as in Levy & Goldberg 3.1. To reproduce old numbers, divide your
    `subsample_factor` by your average sentence length.
-   **`PPMIEmbedding(..., neg=None)` now clips negatives to 0** like every other
    `neg`. It previously returned *signed* PMI and leaked `-inf` for any stored
    zero, which destroyed the affected row in `normalize()`. `neg=1` (the
    default) is unchanged.
-   **All on-disk caches are invalidated.** Cache files are now named
    `<params>_v2-<blake2b digest>.npz`, and the key is the *full effective*
    parameter set (defaults resolved from `count_pairs`' signature, `**kwargs`
    included) rather than the arguments the caller happened to spell. Old
    entries are never found again and are recomputed; they are not deleted, so
    remove them by hand to reclaim space. This is deliberate — the old names
    could not distinguish `window=2` from `window=10`.
-   **`results.db` rows written by older versions are not comparable.** `record`
    now writes every parameter, including positional ones and unspoken defaults,
    and attributes loose `**kwargs` to `pair_args__*` instead of sibling columns.
    Old rows lack `dim`/`cds` and may carry a contradictory `window` column.
-   **`Bunch(path, corpus=None, force_overwrite=True)` now raises `ValueError`**
    instead of deleting the bunch and then failing to reload it.
-   **`results(order=...)` is validated and `limit` is coerced to `int`.**
    Anything outside `column [asc|desc]` raises `ValueError`.
-   **Evaluation metrics change across the board**, because the numbers they
    replaced were wrong (see "Fixed"): word-similarity scores rise, analogy
    accuracies rise, `oov` rises on datasets with multi-word or
    unanswerable-after-lemmatization rows, and `fullscore` is now
    `score - abs(score) * oov`. The last differs from the old
    `score * (1 - oov)` only where `score < 0`. Results recorded by earlier
    versions cannot be compared against new ones.

### Fixed

-   **Similarity/analogy micro and macro averages double-counted shared items.**
    The WordSim353 similarity/relatedness files are subsets of `ws353`, and many
    other bundled sets partially overlap (verified: 462 of the 6097 English
    similarity pair-rows were duplicates across files, and German/analogy sets
    overlap too), yet the aggregates pooled every file — so those judgements were
    weighted two-to-three times. Each unique pair (similarity) or quadruple
    (analogy) is now counted once; a set fully redundant with an earlier one is
    excluded from micro **and** macro but still reported individually. Per-dataset
    `score`/`oov`/`fullscore` are unchanged. This moves the reported micro/macro
    numbers — the old ones were wrong.
-   **The bundled evaluation datasets carried bad rows.** Conflicting duplicate
    pairs (the same pair with two different gold scores, across ~7 similarity
    files), exact duplicate rows, self-pairs, and ~1000 multi-word/hyphenated
    entries the scorer silently could not represent (which inflated `oov`). All
    removed — rows only, no gold score altered or averaged — with the reason
    recorded as `#`-comment provenance in each file. Row-count deltas per file are
    in the commit. Old numbers are preserved in git history, not a parallel suite,
    since this release already changes evaluation results wholesale.

-   **`svd()` was non-deterministic when `dim` exceeded the matrix rank.** Past
    the numerical rank the extra singular values are ~0 and their singular
    vectors are arbitrary null-space directions; under the default `eig=0` those
    got full weight, and `scipy`'s ARPACK seeds them from a random start vector,
    so the same corpus and parameters gave different neighbours run to run.
    Reachable through the default `bunch.svd()` (`dim=500`) on any small-vocab or
    heavily-pruned corpus. `calc_svd` now drops components below the numerical
    rank tolerance, which makes `eig=0` deterministic and the three impls agree;
    the normal `dim < rank` path is unchanged.
-   **`dim >= vocab` crashed the default `svd()`** with an opaque
    `ValueError: k must satisfy 0 < k < min(A.shape)` on `scipy`, while `gensim`
    and `scikit` silently returned different dimensions. `dim` is now clamped to
    `min(shape) - 1` in `calc_svd`; an all-zero matrix (e.g. a huge `neg`) raises
    a clear error instead of `ArpackError: Starting vector is zero`.
-   **An empty corpus raised `AttributeError`** from `count_matrix.nnz` on a
    `None`; it now raises a clear `ValueError` naming the cause.
-   **The cache digest was unstable across runs for `set`-valued parameters.**
    `dict_to_path` fell back to `repr` for non-JSON values, and set repr order is
    hash-randomized, so the same params hashed differently across processes and
    silently missed the cache. `_canonical` now sorts sets and rejects
    non-serialisable values outright. Not reachable through the documented
    scalar parameters; hardening only.
-   **Analogy evaluation was always exactly 0.0.** Three separate bugs:
    `most_similar_vectors` returns `[(idx, score), ...]`, so the `b_ in guesses`
    membership test could never be true; the 3CosAdd arithmetic was inverted
    (the datasets are `a a_ b b_`, so the query is `a_ - a + b`); and the score
    was a raw hit count rather than an accuracy.
-   **`SVDEmbedding.most_similar_vectors` did not exist**, so
    `Bunch.eval_analogy` raised `AttributeError` for SVD embeddings.
-   **`seed=` was a silent no-op in worker processes.** `random.seed()` was called
    in the parent only; under `spawn` (macOS default, and `forkserver` on Linux
    from 3.14) workers got a fresh entropy-seeded RNG, making `subsample="prob"`
    and `dynamic_window="prob"` non-reproducible. Each worker now derives a
    per-file RNG from a stable string seed. This also fixes pre-existing
    nondeterminism from futures completing out of order.
-   **`.todense()` produced a `numpy.matrix`**, which propagated into
    `KeyedVectors.vectors` and made `most_similar()` fail outright.
-   **`calc_pmi` silently upcast float32 counts to float64** via `dok_matrix`,
    doubling memory on the largest object in the pipeline.
-   **`np.reciprocal` on integer counts performed floor division**, silently
    returning an all-zero PMI matrix. Reciprocals are now guarded, which also
    handles zero row/column sums that previously became `inf` -> `nan`.
-   Zero-norm rows in `SVDEmbedding.normalize` produced `nan` (and a
    `RuntimeWarning`) rather than staying zero.
-   **The database retry loop was unbounded.** A permanent failure (schema
    mismatch, read-only filesystem) hung the process forever, printing the same
    traceback every 10 seconds. It is now a bounded exponential backoff that
    catches `SQLAlchemyError` specifically, and the SQLite busy timeout is set to
    30s, which removes the reason the loop existed.
-   **`read_test_data` handed back paths whose files had already been deleted.**
    `as_file` extracts the dataset directory to a temporary location and removes
    it again when the context exits, so returning paths from inside the `with`
    produced dangling names under any non-filesystem loader (zipimport, a
    `.pyz`/pex bundle, a zipped Lambda layer). Verified against a zipimported
    build: reading the first returned path raised `FileNotFoundError`. It now
    returns `Traversable`s and reads them lazily, which needs no extraction.
-   **`subsample="prob"` was inverted.** The map it built was the word2vec
    *discard* probability (`1 - sqrt(t/count)`), but `iterate_tokens` uses it as
    a *keep* probability. Frequent words were therefore kept ~99% of the time
    and rare words discarded ~71% of the time — the opposite of subsampling, and
    the opposite of `subsample="deter"`, which always used the correct factor.
    Both branches now share `subsample_keep_probabilities()`. **This changes
    every result computed with `subsample="prob"`.**
-   **`eval_similarity` / `eval_analogies` raised `ZeroDivisionError`** when no
    dataset had an in-vocabulary row — the normal case for the small corpora
    this package targets, and reachable straight from `Bunch.svd()` and
    `Bunch.pmi()`, which evaluate by default. They now return `nan` micro/macro
    scores with an empty `results` list and log a warning.
-   **`Corpus.from_text_files` was not reproducible**, seed or no seed. Per-file
    vocabularies were merged in worker-completion order, and `filter_extremes`
    breaks document-frequency ties by insertion order, so each run kept a
    different *set* of tokens. Results are now buffered and merged in sorted
    filename order, and the input glob is sorted.
-   **`Bunch.pmi(keyed_vectors=True)` crashed** with the default `evaluate=True`:
    it returned a bare `KeyedVectors` where the `record` decorator expected the
    `(embedding, results)` tuple that `Bunch.svd` returns, and then indexed a
    vector with a string. See "Breaking" — it is now symmetric with `Bunch.svd`.
-   **`nlp.pipe(..., n_process=-1)` made `texts_to_sents` unusable.** spaCy's
    child processes re-import the parent `__main__`, so an unguarded script or a
    notebook cell re-entered the function and spawned pools until it hung; and
    because `texts_to_sents` is itself called from inside a
    `ProcessPoolExecutor`, `Corpus.from_text_files` fanned out to `cpu_count²`
    spaCy processes. It is now `n_process=1` by default and opt-in per call.
-   Mutable default arguments (`pair_args={}`, `impl_args={}`, `query={}`) and an
    in-place mutation in `Bunch.dict_to_path` that could have corrupted the
    shared `default_pair_args` for the process lifetime.
-   **Word-similarity gold scores were ranked as strings, not numbers.** The
    third dataset column was never converted to `float`, and `spearmanr`
    column-stacks its arguments — so a float array stacked with a string array
    promoted *both* columns to strings and ranked them lexicographically
    (`"10" < "2.5" < "9"`). A perfect embedding, which must score exactly 1.0,
    reported 0.66-0.75 depending on the file; worst on `en/bruni_men.txt`, whose
    gold values run `0.000000`-`50.000000` with varying integer width and which
    is half of all English rows, so it dominated the micro average.
-   **The matrix cache ignored `**kwargs`, silently serving a wrong matrix.**
    `Bunch.pmi(window=10)` returned the matrix cached for `window=2`; a sweep
    recorded several database rows that all held the first window's embedding,
    and the answer depended on call order. Same hole in `svd_matrix`.
    `dict_to_path` could also collide two different parameter sets onto one
    filename (case was folded, the `_` separator was forgeable).
-   **Analogy rows unanswerable by construction were scored as wrong** instead of
    skipped. After the default lemmatizing preprocessing `writes` -> `write`
    makes the answer equal to one of the question words, which affects 31% of
    `en/google.txt` and 80% of `en/msr.txt` rows — capping `msr` accuracy near
    0.20 invisibly. They now count as OOV.
-   **`to_item` substituted a different word** rather than skipping multi-token
    entries: `["vice", "president"]` was scored as `vice`. Affects hyphenated and
    multi-word entries (20 in `en/luong_rare`, 10 in `de/gur350`).
-   **`PPMIEmbedding.__init__` destroyed the matrix it was given** (`self.m =
    matrix` without a copy, then rebinding `.data`). Building a second embedding
    from the same matrix silently returned an all-zero `log(log(...))` instead of
    raising. `bunch.pmi_matrix()` hands that matrix to the user.
-   **`results_from_db` crashed on any string filter and interpolated raw SQL.**
    `{"impl": "scipy"}` raised `no such column: scipy`, making every
    string-valued column unfilterable; `order` and `limit` were injectable.
-   **Parallel pair counting was not reproducible.** Partial float32 matrices
    were summed in completion order; merging in a fixed order makes the cached
    `.npz` bit-identical between runs.
-   **`impl="scipy"` returned singular values ascending** while `gensim` and
    `scikit` returned them descending. Similarity and analogy scores were
    unaffected (column scaling plus row normalization), but `s[0]`, `ut[:, :k]`
    slices and the arrays persisted by `svd_matrix` were reversed.
-   **One corpus could not feed two bunches.** `texts_to_file` overwrote
    `corpus.texts` with the paths it had just written, so the second bunch
    pickled those paths as if they were token lists and the workers died with
    `'PosixPath' object is not iterable`.
-   **`tokenize_texts_parallel` nested process pools.**
    `Corpus.from_text_files(preproc_func=tokenize_texts_parallel)` runs the
    preprocessing inside a pool of `workers` processes, each of which then
    started its own pool -- asking for `workers**2` processes. It now stays
    serial inside a worker.

### Performance

Measured by profiling a 250k-sentence corpus, which refuted the assumption the
work started from: `count_pairs` is 8.6% of a `bunch.svd()` and the per-token
loop ~24% of that, so ~2% end to end. The two real costs were the SVD (54%)
and evaluation (28%).

-   **Evaluation no longer spawns a process pool per preprocessing call.** It
    spawned 12, each child re-importing the package, to perform 0.07s of
    tokenization. `tokenize_texts_parallel` now falls back to the serial
    tokenizer below `PARALLEL_MIN_CHARS`. **`eval_similarity` 40.78s ->
    0.087s (468x)**, with every per-dataset score verified bit-identical.
-   **`import spacy` is lazy.** It cost 2.2s on every pool spawn and is only
    needed by `texts_to_sents`.
-   **The default `text_chunk_size` is derived rather than fixed.** It is a
    size where the pool needs a count: 250k sentences produced 3 chunks, so
    10 cores delivered 1.8x. **`count_pairs` 8.58s -> 5.10s.** Explicit
    `text_chunk_size` values are unchanged in meaning; values `< 1` now raise.
-   **`count_pairs` stays serial when a pool would not pay for itself**,
    decided by timing a probe chunk rather than by a constant -- the same
    corpus runs at 4.3 or 0.54 us/token depending on `subsample`, an 8x spread
    no fixed threshold can straddle. Incidentally takes the equivalence test
    suite from 478.9s to 8.7s.
-   `merge_order()` makes the summation order an explicit contract, so which
    worker finishes when no longer influences the result at all.

### Reproducibility

-   **Counting results no longer depend on the machine's core count.** Both the
    partial-matrix merge order (grouped by `2 * workers + 1`) and the automatic
    chunk size (`workers * 4` chunks) were derived from the local core count, so
    an 8-core and a 16-core machine produced matrices differing in the last bits
    -- the merge summed in a different order, and the chunking split the corpus
    differently. `merge_order()` is now `sorted(paths)` and the chunk count is a
    fixed `TARGET_TEXT_CHUNKS`; the pool size stays the core count. Building the
    same bunch on two machines now yields byte-identical matrices. The frozen
    equivalence reference was re-taken for this; results move by at most 2-3
    float32 ulps (max 2.4e-4) versus the previous core-count-dependent output.
-   **`default_pair_args` is derived from the `count_pairs` signature** instead
    of a hand-written copy that had already drifted: it omitted `seed` and
    `min_count`, so a run using their defaults recorded no value for them and
    `results(query={"min_count": 0})` matched nothing.

### Changed

-   **`from_texts` / `from_text_files` default to the lightweight tokenizer, not
    spaCy** (ADR 0002). They used to default to `texts_to_sents` (spaCy
    sentence-splitting + lemmatization, needs a model); now all constructors
    default to `tokenize_texts_parallel_v2`, so the package is lightweight by
    default everywhere. `texts_to_sents` stays available as an explicit
    `preproc_func`. This changes the vocabulary — and therefore evaluation numbers —
    for **new** corpora built with the default; existing bunches are unaffected.


-   Packaging moved from Poetry to PEP 621 + hatchling; `poetry.lock` replaced by
    `uv.lock`.
-   CI moved from Travis to GitHub Actions, testing 3.10-3.13.
-   Linting and formatting via ruff.
-   `print()` calls replaced with module loggers, restoring the `NullHandler`
    contract that `__init__.py` sets up.
-   Worker pool sizing respects CPU affinity and cgroup quotas rather than using
    the raw host core count — including the two `ProcessPoolExecutor`s in
    `corpus.py` and the one in `pair_counts.py`, which previously took the bare
    `os.cpu_count()` default.
-   `nlp.pipe` now runs batched (`batch_size=1000`). It is still single-process
    by default; see `n_process` under "Breaking".
-   `calc_pmi` no longer forces float32. It promotes integer counts to float but
    preserves a float64 input, which previously would have been computed at
    float32 precision and returned with a float64 dtype.
-   `subsample="deter"` and `subsample="prob"` now derive their factor from one
    shared function, so they cannot drift apart again.
