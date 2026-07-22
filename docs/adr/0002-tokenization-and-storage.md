# ADR 0002 — Tokenization and on-disk storage

- **Status:** Proposed
- **Date:** 2026-07-22
- **Deciders:** maintainer
- **Informed by:** two independent model reviews, preserved verbatim as annexes
  ([Fable](references/0002-annex-fable-review.md),
  [GPT/codex](references/0002-annex-codex-review.md)), plus direct verification of
  the load-bearing claims below.

## Context

Two questions, prompted by real pain from ADR 0001's data work: is the tokenizer
right, and why is the data stored the way it is ("why not CSV?").

**Verified facts** (checked in-repo, not assumed):

1. **NFC is a live bug.** `tokenize_string("café")` returns `["café"]` for NFC
   input but `["cafe"]` for NFD — combining marks are `\W`, so `strip_non_alphanum`
   deletes the accent. The same surface word becomes two vocab entries, silently.
2. **The tokenizer splits intra-word punctuation destructively.** `city's` →
   `["city","s"]`, `ice-cream` → `["ice","cream"]`, `3.50` → `["0","00"]`. This is
   why ADR 0001's cleaning had to *drop* ~1000 possessive/hyphenated evaluation
   rows — the data was bent around a tokenizer deficiency.
3. **`preproc_fun` is pickled by reference.** `Corpus(SaveLoad)` stores it as a
   module+qualname reference; unpickling rebinds to the *current* definition. So
   changing `tokenize_string` in place would silently alter every existing bunch's
   evaluation preprocessing on reopen. **Any semantic change must use a new name.**
4. **The lightweight tokenizer is not the default where it matters.** `from_texts`
   and `from_text_files` default to `preproc_func=texts_to_sents` (spaCy
   lemmatization, needs a model); only `from_sents` defaults to the lightweight
   path. The heavy default sits on the most-used constructors — against the
   package's "lightweight" identity.
5. **The evaluation parser discards silently.** `setup_test_tokens` keeps only
   lines that split into exactly N fields and drops everything else with no
   warning — which is both how the comment-leak was dangerous and how a typo'd row
   vanishes unnoticed.

## Decision

### Tokenization — both reviews agreed; adopted

- **A new tokenizer under a new name (`tokenize_string_v2` and the `_v2` wrappers),
  made the default via the constructors — never an in-place edit** (fact 3). The
  old functions stay forever so existing bunches are bit-for-bit unaffected.
  Semantics:
  1. `unicodedata.normalize("NFC", ...)` first (fixes fact 1; stdlib).
  2. Lowercase (configurable, default on).
  3. Extraction, not destruction: `re.findall(r"\w+(?:['’\-]\w+)*", ...)` — keeps
     `city's`, `ice-cream`, `don't` whole, splits on everything else. Canonicalize
     `’`→`'` and Unicode hyphens→`-` first. Deliberately do **not** rescue dotted
     forms (`u.s.a`, `3.50`) — adding `.` to the joiner glues `word.Next` in
     scraped text, a worse trade.
  4. Digit normalization is a **parameter**, no longer silently baked in.
- **Inline the three `gensim.parsing` regex helpers** (A1). Not about dependency
  count (gensim stays, via `Vocab`/`KeyedVectors`) — about owning and testing the
  tokenizer's semantics in-repo so they cannot drift under a gensim upgrade.
- **Fix the constructor defaults**: `from_texts` / `from_text_files` default to the
  lightweight tokenizer, not spaCy (fact 4). `texts_to_sents` stays as an opt-in
  callable.
- **Formalize `preproc_func` as the pluggability contract** (A2): document it as a
  picklable top-level `Callable[[list[str]], list[list[str]]]`. This is where
  spaCy/HF belong — user-supplied, not a built-in tokenizer hierarchy. No registry,
  no language auto-detection.
- **Record the tokenizer identity** (qualname + options) in the results DB, so
  scores computed under v1 and v2 are attributable and don't collide.

### Storage — the reviews diverged; decided toward the proportionate path

- **Evaluation datasets: keep the whitespace `.txt` format. Do NOT migrate to
  TSV/CSV/JSONL now.** Harden the parser instead:
  - Skip lines whose first non-space char is `#` *before* the field-count filter —
    a real comment convention, retiring ADR 0001's fragile "headers must not split
    into N fields" invariant (one line).
  - **Warn (with filename:line) on a malformed non-comment, non-blank line instead
    of dropping it silently** (fact 5) — this captures Codex's strongest
    robustness point without a format migration.
  - Rationale for not migrating: multi-word entries are a **scorer** limitation
    (`to_item` needs one vocab token), not a format one — TSV would let us *write*
    `vice president` but nothing could *score* it. Against that, `.txt` keeps the
    de-facto hyperwords/word2vec interop, glanceability, and every existing user
    file. See "Where the reviews diverged".
- **Corpus training chunks: keep pickle.** They are ragged `array('H'/'L')` integer
  id sequences (not readable tokens), a derived regenerable cache inside a bunch
  directory. Document that a bunch directory is a **trusted local cache — never
  load one from an untrusted source** (the honest pickle-security answer). Make
  `allow_pickle=False` explicit on numpy loads and prefer atomic writes.
- **Matrices: keep `.npz`** (compressed sparse / dense). Both reviews and the
  earlier audit agree; CSV for a sparse matrix is absurd.

### Consequence — re-curate after v2

Once v2 lands, most of the ~1000 rows ADR 0001 dropped as "multi-word" become
single tokens again (`ice-cream`). Restore them as a **curated-v3** pass, keeping
the provenance trail. Genuinely multi-word entries (`vice president`) stay dropped
— the unigram vocab can never score them, regardless of tokenizer or format.

## Where the reviews diverged

The one real disagreement was the evaluation-data format:

- **GPT/codex:** migrate to strict UTF-8 **TSV** (header row, `# key: value`
  preamble, quoted fields, `.tsv` extension), read both `.tsv` and legacy `.txt`,
  write TSV. Its strongest arguments: a real comment/metadata mechanism, malformed
  rows that *error with filename:line* instead of vanishing, and forward-fit for
  richer future data.
- **Fable:** keep `.txt`; the comment leak is a one-line fix and multi-word is
  unscoreable, so TSV buys representational power the evaluator cannot use, at the
  cost of interop and every existing file.

**Resolved toward Fable, taking Codex's robustness point.** We do not migrate, but
we do add both the `#`-comment convention and the warn-on-malformed behavior — so
the "silent discard" danger is fixed without spending the interop/glanceability
the format has. If a later phase (ADR 0001's P4 domain proxy tasks) needs a
genuinely richer row shape, that new `kind` may define its own format then, decided
on its own merits rather than pre-migrating `ws`/`analogy` for it.

## Two judgment calls worth the maintainer's eye

1. **Digit normalization default. DECIDED (2026-07-22): keep digits by default.**
   Fable kept digit→`0` (the Levy-Goldberg-Dagan / hyperwords convention this
   package reimplements); Codex defaulted to keeping digits (for domain corpora
   where `IL-6`, years, model numbers carry meaning). Both agreed it must become an
   explicit parameter. The maintainer chose **keep digits by default** — the stated
   audience is small *domain* corpora — with legacy digit→`0` available as an
   option and documented as the paper's convention.
2. **Not migrating to TSV.** The proportionate choice above; recorded explicitly so
   a future maintainer sees it was a decision, not an oversight.

## Consequences

- The v2 tokenizer changes vocabulary, so **all** similarity/analogy numbers move
  for corpora built under it. Old bunches are unaffected (old function names). A
  breaking *evaluation* change for new bunds; goes in the CHANGELOG, and the
  tokenizer identity is recorded per result.
- Evaluation gains a real comment convention and stops hiding malformed rows.
- The package's "lightweight by default" claim becomes true (no spaCy default).
- curated-v3 recovers evaluation coverage the v1 tokenizer had cost us.

## Roadmap (priority order)

1. `#`-comment support + warn-on-malformed in `setup_test_tokens` (one small,
   high-value change; defuses a landmine and stops silent data loss).
2. Inline the `gensim.parsing` regexes (pins semantics, zero behavior change).
3. Fix the `from_texts` / `from_text_files` spaCy default → lightweight tokenizer.
4. Document the `preproc_func` contract.
5. `tokenize_string_v2` (NFC + extraction + digit parameter) under new names,
   default-flipped; record tokenizer identity in the results DB.
6. curated-v3: restore the tokenizer-induced dropped rows.
7. (Opportunistic, not now) npz id-chunks replacing `texts_N.pkl`; atomic writes;
   bunch-directory trust note.

## Alternatives considered

- **TSV/CSV/JSONL migration of eval data** — rejected now as disproportionate (see
  divergence); revisit only if a richer `kind` forces it.
- **A tokenizer class hierarchy / registry / language auto-detection** — rejected;
  `preproc_func` already is the hook.
- **A heavy default tokenizer (spaCy/HF)** — rejected; betrays the package identity.
- **Rewriting `corpus.pkl` / chunk pickles now** — deferred; it's a regenerable
  local cache, and a holistic bunch-format revision is the right scope for that,
  not a piecemeal change.
