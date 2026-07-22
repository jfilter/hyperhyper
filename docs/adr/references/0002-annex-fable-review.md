# ADR 0002 — Annex: Fable review, verbatim

> Independent review by the Fable model (read-only; verified the tokenizer's edge
> behavior in-repo), 2026-07-22, on the tokenization + on-disk-format brief.
> Preserved verbatim as source material for
> [0002-tokenization-and-storage.md](../0002-tokenization-and-storage.md). Not all
> recommendations were adopted — see the ADR's "Where the reviews diverged".

---

I've read the pipeline end to end (`preprocessing.py`, `corpus.py`, `bunch.py`,
`evaluation.py`, `utils.py`, the bundled datasets, ADR 0001, `pyproject.toml`) and
empirically verified the tokenizer's edge behavior. Two verified facts sharpen
both questions before the recommendations:

- **NFC is a live bug, not a nicety.** Verified in-repo: `tokenize_string("café")`
  returns `["café"]` for NFC input but `["cafe"]` for NFD input — combining marks
  are `\W`, so `strip_non_alphanum` deletes the accent. Same surface word, two
  different vocab entries, silently. Also verified: `"3.50"` tokenizes to
  `["0", "00"]` (digit-norm runs before punctuation stripping), `don't` (U+2019)
  splits just like `don't`, and underscores survive (`co_op` stays whole) because
  gensim's filter is `\W`-based.
- **Pickle-by-reference is the migration constraint that dominates Question A.**
  `corpus.pkl` (gensim `SaveLoad`) stores `preproc_fun` as a module+qualname
  reference. Unpickling rebinds to the *current* definition. So if you "fix"
  `tokenize_string` in place, every previously built bunch silently changes its
  evaluation preprocessing on reopen (`Bunch.eval_sim` runs
  `self.corpus.preproc_fun` on test words). Any semantic change must land under a
  **new function name**.

# QUESTION A — tokenization

**Verdict: a regex/whitespace tokenizer is still the right default in 2026 for
this package.** Count-based PPMI+SVD on small corpora doesn't benefit from subword
or learned tokenization; a 10-line deterministic tokenizer is exactly what
"reproducible, inspectable" means. But the *current* one is the wrong 10 lines,
and it's already 80% pluggable — finish that.

Ranked plan:

**A1 (do now, zero behavior change): inline the three gensim.parsing helpers.**
`strip_tags`, `strip_non_alphanum`, `preprocess_string` are ~6 lines of regex.
gensim stays a hard dependency regardless (`Vocab` extends `Dictionary`,
`KeyedVectors`), so this isn't about dependency count — it's about pinning
tokenizer *semantics* in-repo where they're testable and can't drift under a
gensim upgrade. Cheap, aligned with "inspectable".

**A2 (do now): formalize pluggability — it already exists, it's just undocumented
and under-specified.** `Corpus.from_sents(preproc_func=...)` /
`from_text_files(preproc_func=...)` already accept any callable, and evaluation
correctly reuses `corpus.preproc_fun`, so a custom tokenizer is automatically
applied consistently to corpus *and* test words. What's missing is the contract:
document that `preproc_func` is `Callable[[list[str]], list[list[str]]]` and **must
be a picklable top-level function** (it goes into `corpus.pkl` and across process
pools; a lambda breaks both). This is where spaCy/HF belongs: as a user-supplied
callable (the `texts_to_sents` spaCy path is already exactly this pattern). Do NOT
build a tokenizer class hierarchy, registry, or language auto-detection.

**A3 (the real fix, next minor version): a `tokenize_string_v2` under a new name,
flipped in as the default.** Spec:
1. `unicodedata.normalize("NFC", text)` first — stdlib, fixes the café/cafe split.
2. Lowercase.
3. Digit→"0": **keep it, but as a parameter.** It's the hyperwords/Levy-Goldberg-
   Dagan convention this package reimplements, and collapsing numerals genuinely
   helps tiny corpora. It is surprising, so document it loudly and make it a flag;
   don't drop it from the default.
4. Replace destructive substitution with an extraction pattern:
   `re.findall(r"\w+(?:['’\-]\w+)*", text)` — keeps `city's`, `ice-cream`, `don't`
   whole; still splits everything else on non-word chars. Deliberately do NOT add
   `.` to the joiner set: it would rescue `3.50` and `u.s.a` but glue `word.Next`
   in scraped text with missing spaces — worse trade. No English clitic stripping
   (`'s` removal) — keep the default language-neutral.

**Migration:** new names all the way up (`tokenize_texts_parallel_v2` too), flip
the `preproc_func=` default in `Corpus.from_sents`. Old bunches keep pointing at
the old names via pickle and are bit-for-bit unaffected; new bunches get v2. Old
function stays forever. CHANGELOG marks it a breaking *evaluation* change (vocab
changes → all scores move). Record the tokenizer's qualname in the results DB so
recorded numbers are attributable.

**A4 (consequence): re-curate the datasets after v2.** This matters: the
curated-v2 cleaning dropped ~1000 hyphenated/multi-word rows because *the v1
tokenizer* couldn't represent them — the datasets were bent around a tokenizer
deficiency. Under v2, `ice-cream` is one token and most of those rows become
scoreable again. Restore the tokenizer-induced drops as a curated-v3 (keeping the
legacy suite per ADR 0001). True multi-word entries (`vice president`) stay
dropped — the unigram vocab can never score them, no tokenizer or file format
changes that.

# QUESTION B — on-disk formats

**B1 evaluation datasets: keep whitespace `.txt`; add one line of parser: real `#`
comments.** Verdict against migration:
- The comment leak is real but the fix is trivial: skip
  `line.lstrip().startswith("#")` in `setup_test_tokens` *before* the field-count
  filter. That retires the fragile invariant `tests/test_datasets.py` currently has
  to pin ("no comment line may split into exactly N fields") and makes the
  provenance headers already shipped in curated-v2 robust by construction.
  Collision risk is nil (no legitimate row starts with `#`; the tokenizer would
  destroy a literal `#` word anyway). Optionally also skip `:`-prefixed lines for
  word2vec-format analogy section headers.
- Multi-word entries are a **scorer** limitation, not a format limitation —
  `to_item` demands a single vocab token. JSONL/TSV would let you *write* `vice
  president` but nothing could *score* it. Migrating formats buys representational
  power the evaluator cannot consume, at the cost of the de-facto
  hyperwords/word2vec interop, glanceability, and every existing user file.
- Provenance: the `#`-header convention already in the curated files is
  proportionate (ADR 0001 point 7). Formalize the key-value comment style in docs;
  no code.
- **Do NOT:** CSV (quoting rules for data that never contains delimiters), JSONL
  (kills glanceability and interop for zero scoring benefit), or a TSV header row
  (breaks every downstream space-delimited parser, including users' hyperwords-era
  files).

**B2 corpus pickles: acceptable now, replaceable later — and the review's premise
is slightly off in an interesting way.** The `texts_N.pkl` chunks are not lists of
token-lists for most of their life: `TransformToIndicesClosure`/`_texts_to_ids`
overwrite them with lists of stdlib `array('H'/'L')` **integer id sequences**. So a
"readable JSONL" version would be columns of vocab ids — not meaningfully more
inspectable. Pickle of stdlib types is fast, version-stable, and this is a derived,
regenerable cache inside a bunch directory. Verdict: keep for now; document that
bunch directories are trusted local caches (never load a bunch from an untrusted
source — that's the honest pickle-security mitigation). *If* you touch this area
later, the natural non-pickle format is a `.npz` per chunk (concatenated
uint16/uint32 ids + offsets array, `allow_pickle=False`), read-old-pickle-fallback
exactly like the existing legacy-matrix fallback in `load_matrix`. Worth doing
opportunistically, not now. The bigger fragility is `corpus.pkl` (gensim class +
function reference) — leave its format alone (it's entangled with gensim
`SaveLoad`), but it's the reason A3 must use new function names.

**B3 matrices: confirmed correct, nothing to do.** scipy `save_npz`/`load_npz` is
non-pickle, compressed, and the standard; the legacy hand-rolled-CSR fallback in
`hyperhyper/utils.py` is the right back-compat pattern. `np.load` calls use the
safe `allow_pickle=False` default. CSV for sparse matrices would be absurd, agreed.

# Priority order

1. `#`-comment support in `setup_test_tokens` (one line; defuses a landmine).
2. Inline the gensim.parsing regexes (pins semantics, no behavior change).
3. Document the `preproc_func` contract (pluggability is already built; make it
   official).
4. `tokenize_string_v2` (NFC + extraction pattern + digit-flag) under new names,
   default-flipped in the next minor.
5. Curated-v3: restore the ~1000 tokenizer-induced dropped rows.
6. (Opportunistic) npz id-chunks replacing `texts_N.pkl`; trust-model note for
   bunch dirs.

**Not doing, deliberately:** heavy default tokenizer, tokenizer abstraction layer,
clitic/compound language rules, CSV/TSV/JSONL dataset migration, header rows,
corpus.pkl reformat, uncompressed or dense matrix formats. Each would spend the
package's "lightweight, reproducible" identity to solve a problem it doesn't have.
