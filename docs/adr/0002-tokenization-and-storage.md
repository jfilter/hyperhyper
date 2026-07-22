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

- **Evaluation datasets: migrate to TSV, reading legacy `.txt` for
  backward-compatibility.** *(Reversed 2026-07-22 — this section originally decided
  to keep `.txt`; see the amendment below.)*
  - Canonical format: UTF-8 `.tsv` with a `# key: value` metadata preamble, a
    required header row, tab-delimited fields, parsed with the stdlib `csv` module.
    Malformed rows **raise** (with file:line), not warn.
  - Legacy `.txt` (whitespace) is still read via the hardened parser below —
    users' existing files keep working; the reader dispatches on file extension,
    never by sniffing. A directory with both `foo.tsv` and `foo.txt` is an error.
  - The parser hardening decided here still shipped and is now the **legacy `.txt`
    read path**: skip `#`/`:` comment lines before the field-count filter (retiring
    ADR 0001's fragile "headers must not split into N fields" invariant), and warn
    (file:line) on a malformed legacy row instead of dropping it silently (fact 5).

  > **Amendment (2026-07-22): reversed from "keep `.txt`" to "migrate to TSV".**
  > The original decision rested on a hyperwords/word2vec "space-delimited interop"
  > premise. Checking the files disproved it: they are already *inconsistently*
  > delimited — `en/ws/ws353.txt` and `en/analogy/msr.txt` are tab-separated,
  > `de/ws/gur350.txt` is space-separated, all parsed leniently by `.split()` and
  > misnamed `.txt`. The data is already half-TSV. A declared TSV format with a
  > real parser *removes* that mess, gives headers + a metadata preamble + strict
  > errors-not-silent-drops, and fits the richer rows the ADR-0001 P4 domain-proxy
  > datasets will need (e.g. synonym multiple-choice: target + gold + distractors).
  > The one thing TSV does *not* buy — scoring multi-word entries — remains a
  > *scorer* limitation (unigram vocab), unchanged by the format. GPT/codex
  > recommended TSV from the start; Fable's "keep `.txt`" rested on the interop
  > premise the files contradict. Maintainer confirmed the migration.
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

**Initially resolved toward Fable, then reversed to Codex's TSV (2026-07-22).** The
first call kept `.txt` and only hardened the parser. On a second look the interop
premise behind "keep `.txt`" proved false — the bundled files are already a
tab/space mix (see the amendment under Decision) — so the maintainer took Codex's
recommendation: migrate to TSV, keep reading legacy `.txt`. The parser hardening
still shipped and became the legacy read path, so it was not wasted.

## Two judgment calls worth the maintainer's eye

1. **Digit normalization default. DECIDED (2026-07-22): keep digits by default.**
   Fable kept digit→`0` (the Levy-Goldberg-Dagan / hyperwords convention this
   package reimplements); Codex defaulted to keeping digits (for domain corpora
   where `IL-6`, years, model numbers carry meaning). Both agreed it must become an
   explicit parameter. The maintainer chose **keep digits by default** — the stated
   audience is small *domain* corpora — with legacy digit→`0` available as an
   option and documented as the paper's convention.
2. **Evaluation-data format. DECIDED (2026-07-22): migrate to TSV**, reading legacy
   `.txt` for backward-compatibility. Initially this ADR decided to keep `.txt`;
   that was reversed once the "space-delimited interop" premise was found false (the
   files are already a tab/space mix) and because ADR-0001 P4 needs richer rows. See
   the amendment under Decision.

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
7. (Opportunistic) npz id-chunks replacing `texts_N.pkl`; atomic writes;
   bunch-directory trust note. **Partly done 2026-07-22:** atomic writes and the
   trust note shipped, together with an explicit `allow_pickle=False` on every
   numpy load. Replacing the chunk pickles with npz is still deferred -- they are
   a regenerable derived cache, and a holistic bunch-format revision remains the
   right scope for that.

## Alternatives considered

- **TSV/CSV/JSONL migration of eval data** — rejected now as disproportionate (see
  divergence); revisit only if a richer `kind` forces it.
- **A tokenizer class hierarchy / registry / language auto-detection** — rejected;
  `preproc_func` already is the hook.
- **A heavy default tokenizer (spaCy/HF)** — rejected; betrays the package identity.
- **Rewriting `corpus.pkl` / chunk pickles now** — deferred; it's a regenerable
  local cache, and a holistic bunch-format revision is the right scope for that,
  not a piecemeal change.
