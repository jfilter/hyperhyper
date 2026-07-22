# ADR 0001 — Annex: Fable review, verbatim

> Independent review by the Fable model (read-only, inspected the code and data),
> 2026-07-22, in response to the shared brief. Preserved verbatim as source
> material for the decision recorded in
> [0001-evaluation-data-strategy.md](../0001-evaluation-data-strategy.md).
> Not all recommendations were adopted — see the ADR's "Alternatives considered".

---

I inspected the actual eval code (`hyperhyper/evaluation.py`), the bundled data
under `hyperhyper/evaluation_datasets/`, and re-measured the quality problems
before writing this. Ground truth I verified: conflicting-duplicate pairs exist
not only in `de/ws/schm280.txt` (5, e.g. `küste wald` scored both 4.2 and 5.4)
but also `de/ws/simlex999.txt` (5), `de/ws/ws353sim.txt` (2), `en/ws/ws353.txt`
(2), and 1 each in `de/ws/ws353rel.txt`, `en/ws/luong_rare.txt`,
`en/ws/radinsky_mturk.txt`; hyphenated entries: 10 in `de/ws/gur350.txt`, 20 in
`en/ws/luong_rare.txt`. Also relevant: `read_test_data()` only reads
package-bundled data — users cannot currently point the evaluator at their own
domain sets, and `setup_test_tokens()` drops any line without exactly 3/4
fields, which means comment/provenance headers in data files are already safe.

## Plan: better evaluation data for hyperhyper (prioritized)

### P0 — Clean and lint the existing datasets (cheapest, highest certainty)

Do this before generating anything new; it fixes measured bugs in the gold data itself.

1. **Resolve conflicting duplicates deterministically.** For each of the ~17
   duplicate pairs across 7 files: consult the upstream original (sources are
   listed in `de/README.txt`); if the duplicate is an upstream artifact, keep the
   upstream-canonical row; if genuinely conflicting with no canonical answer,
   average the scores. Record every decision as `#`-comment lines at the top of
   the file (the parser ignores them) — provenance lives in-file, survives
   vendoring.
2. **Drop multi-token/hyphenated rows from the files** (gur350, luong_rare). The
   code already refuses to score them (`to_item` returns `None`), but they now
   silently inflate the OOV penalty in `penalize_oov`. Removing them makes `oov`
   mean what it says. Note the removals in the file header.
3. **Add a dataset lint test** in the test suite that asserts, for every bundled
   file: correct field count, score parses as float, no case-insensitive
   duplicate pairs, no self-pairs (`w w`), single-token entries, and for
   analogies no row where `b_ ∈ {a, a_, b}` pre-lemmatization. This becomes the
   acceptance gate for all future datasets (including generated ones), so quality
   regressions become impossible rather than merely unlikely.
4. **Decide and document a casing policy per language.** DE files mix cased
   (`China Yuan …` in `analogy/open.txt`) and lowercased (`ws/schm280.txt`)
   conventions; today the preprocessing pipeline papers over this. Normalize the
   files to one convention per language and state it in the README.

### P1 — Unlock user-supplied / domain datasets (small code change, biggest leverage for the actual use case)

The stated use case is small domain corpora, but `read_test_data()` hardcodes
`files(evaluation_datasets)`. No amount of bundled data fixes that.

1. Add an optional `data_dir` / extra-datasets parameter to `eval_similarity` /
   `eval_analogies` so users can drop their own `ws/` and `analogy/` files in the
   same 3/4-column format. Bundled data stays the default; this is additive and
   dependency-free.
2. Ship a **coverage report tool**: given a corpus vocabulary and a dataset,
   report the fraction of rows fully in-vocabulary *before* training. Users learn
   instantly which bundled sets are usable for their domain and whether a custom
   set is warranted. This is "better evaluation data" achieved with zero new data.
3. Expose the P0 linter as a small CLI/script so users can validate their own
   datasets against the same gate.

### P2 — Generate NEW ANALOGY sets with an LLM + independent verification (the right task to generate)

**Why analogy, not similarity:** analogy gold answers are facts (capital:country,
plural, currency, comparative), checkable against sources that are independent of
any LLM. Similarity gold is aggregated human judgement; an LLM assigning scores
collapses the benchmark into "does PPMI agree with an LLM" — the circularity
trap. So: generate analogies, import similarities (P3).

Pipeline design (lives in a separate offline directory, e.g. `bench/datagen/` in
the repo root — never a runtime dependency, no torch/transformers/network in the
package):

1. **Schema first.** Define relation types explicitly, restricted to relations
   with *unique* answers (3CosAdd needs one correct `b_`): capital–country,
   country–currency, country–demonym adjective, comparative/superlative,
   singular–plural, masculine–feminine profession nouns, verb inflection.
   Explicitly exclude many-to-many relations (hypernymy, synonymy, "opposite" in
   the loose sense — `de/analogy/open.txt`'s currency section is fine, but
   free-association relations are not).
2. **Generate PAIRS, not quadruples.** The LLM proposes `(a, a_)` pairs per
   relation; quadruples are formed combinatorially afterwards (as the Google set
   does). Verifying N pairs instead of N² quads is what makes verification
   tractable.
3. **Verify against non-LLM sources.** Factual relations: Wikidata SPARQL
   (capitals, currencies, demonyms). Morphological relations: Wiktionary dumps /
   UniMorph tables. Only where no structured source exists, use a *second,
   different-vendor* LLM strictly as a verifier with a yes/no fact question; any
   disagreement drops the pair. Target 100% verified for factual sets — drop,
   never "fix".
4. **Filter for usability:** single-token under the package's own preprocessing;
   frequency-banded using a reference frequency list (e.g. wordfreq) so sets
   aren't dominated by rare words; injectivity check within each set (no two pairs
   sharing an answer word in a way that makes rows ambiguous).
5. **Freeze and label.** Commit the generated `.txt` with an in-file comment
   header: generator model + version + date, prompt hash, verification source and
   pass rate, license. The *frozen file* is the benchmark artifact of record; the
   generation script is kept for audit but never re-run at install/test time
   (regeneration would silently break score comparability, which this project has
   been hardened against).
6. **Contamination check:** diff generated pairs against `google.txt`/`msr.txt`
   (the LLM will happily regurgitate famous benchmark content); dedupe or flag
   overlaps in the header.

Targets, in order: (a) 2–3 new languages chosen by user demand (FR, ES are the
obvious first), (b) a small "domain template" showing how to build a domain
analogy set (e.g. org–abbreviation, term–unit) using the same pipeline — but be
honest in docs that many domains lack clean 1:1 relations, which leads to P3/P4.

### P3 — More similarity data: import human-rated sets, never generate scores

1. **New languages:** Multi-SimLex (12+ languages, human-rated), translated
   WordSim353/RG65 variants, subject to license review per dataset. This is how
   similarity coverage grows without the validity trap.
2. **Domain similarity:** human-rated domain sets already exist — biomedical
   UMNSRS, MayoSRS, Bio-SimLex/Bio-SimVerb. Import (or document how to fetch, if
   licenses forbid bundling) rather than generate. For domains without existing
   sets, the answer is P4, not LLM scores.
3. An LLM may assist only with *mechanical* roles for similarity data: format
   conversion, tokenization checks, flagging suspect rows for human review — never
   producing or altering a gold score.

### P4 — Objective proxy tasks for arbitrary domains (v2, design now, build later)

For a user's niche corpus, graded similarity gold is unobtainable, but
*set-membership* tasks are objective and verifiable:

- **Synonym multiple choice (TOEFL-style):** gold synonym drawn from a domain
  glossary/Wiktionary; 3 frequency-matched distractors. LLM proposes candidates;
  gold-ness is checked against the glossary, so verification is membership, not
  judgement. Scored by cosine ranking — fits the existing vector API.
- **Category purity:** category → members from a taxonomy; score whether the
  embedding separates categories. Same verification property.

These would need a third `kind` next to `ws/` and `analogy/`, so they are v2 —
but they are the *correct* long-term answer for domain evaluation, and P1's
`data_dir` mechanism should be designed so a new `kind` slots in.

## Validation protocol for any generated dataset (gate before merging)

1. **Structural lint** (P0 tool) passes.
2. **Fact verification rate**: 100% for structured-source relations; ≥ agreed
   threshold with dual-LLM verification elsewhere, disagreements dropped.
3. **Human spot check**: random 50–100 rows, maintainer-verified, ≥98% correct —
   feasible precisely because analogy rows are objectively right/wrong.
4. **Convergent validity**: score a battery of models (several hyperhyper configs
   on a public corpus, plus a pretrained fastText for that language) on the new
   set; require (a) accuracy well above chance and below ceiling for the good
   models, and (b) Spearman ≥ ~0.7 between model rankings on the new set vs.
   established sets of the same language. A set that ranks models randomly is
   noise regardless of how correct its facts are.
5. **Difficulty calibration**: report per-relation accuracy and word-frequency
   bands in the header; prune relations at floor (0%) or ceiling (100% for every
   model) since they carry no signal.

## What NOT to do

- **No LLM-assigned similarity scores as gold**, ever — including "rescoring",
  gap-filling, or LLM translation of human sets *while keeping the original
  scores* (translation shifts word senses; the ratings don't transfer).
- **No LLM in the metric path** (no LLM-as-judge at eval time) and **no ML/network
  dependency at runtime** — generation stays in `bench/datagen/`, artifacts are
  frozen text in git.
- **No regenerate-on-demand benchmarks** — a benchmark that changes between runs
  destroys the reproducibility this project was hardened for.
- **No silent fixes** to existing gold data — every dedup/drop decision is
  recorded in the file header and covered by the lint test.
- **No many-to-many analogy relations** (hypernymy, loose antonymy) — they make
  3CosAdd rows unanswerable by construction, the same failure mode the code
  already documents for lemma-collapsed rows.
- **Don't bulk-bundle huge generated sets** — the data ships inside the wheel;
  keep new sets in the low thousands of rows and license-clean.

**Ranked by value:** P0 (fixes measured gold-data bugs, half a day) > P1 (unlocks
the actual domain use case with ~no new data) > P2 (LLM+Wikidata analogy
pipeline, new languages) > P3 (import human similarity sets) > P4 (objective
domain proxy tasks) — with the validation protocol as standing infrastructure
applied from P2 onward.

---

## Note on one point not adopted

Fable's P0 step 1 allows *averaging* two conflicting aggregate scores when no
canonical upstream answer exists. The ADR overrides this with GPT's stricter rule
— **drop** such pairs rather than average two numbers whose provenance we do not
understand — because averaging fabricates a gold value. See the ADR's Decision 2.
