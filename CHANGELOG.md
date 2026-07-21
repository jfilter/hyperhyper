# Changelog

## Unreleased

Modernization of the package for current Python and dependency versions.
Everything below is user-visible; several items change numeric results.

### Added

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
