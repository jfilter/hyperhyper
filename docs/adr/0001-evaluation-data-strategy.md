# ADR 0001 — Evaluation data strategy

- **Status:** Proposed
- **Date:** 2026-07-22
- **Deciders:** maintainer
- **Informed by:** an internal review with two independent models, plus direct
  verification of the claims below against the code and data. Both reviews are
  preserved verbatim as annexes:
  [Fable](references/0001-annex-fable-review.md),
  [GPT/codex](references/0001-annex-codex-review.md).

## Context

`hyperhyper` evaluates embeddings on two intrinsic tasks with bundled datasets
under `hyperhyper/evaluation_datasets/{en,de}/`:

- **Word similarity** (`ws/`): `word1 word2 human_score`, scored by Spearman
  correlation between the embedding cosine and the human score.
- **Word analogy** (`analogy/`): `a a_ b b_`, scored by 3CosAdd / 3CosMul
  accuracy (does `a_ - a + b` land on `b_`).

The question that prompted this: *with modern AI, should we generate more of
these datasets?* Answering it well requires separating three things that are
easy to conflate — generating data, fixing the data we have, and letting users
bring their own.

Three facts were **verified**, not assumed, before writing this:

1. **A live aggregation bug.** `ws353_similarity.txt` (203 pairs) and
   `ws353_relatedness.txt` (251 pairs) are *complete subsets* of `ws353.txt`
   (351 pairs) — verified: 203/203 and 251/251 pairs overlap. `eval_similarity`
   iterates every file in `ws/` and micro-averages line-weighted across all of
   them, so those judgements are counted two-to-three times in the English
   word-similarity micro score. This is a correctness bug in the metric, not
   merely a data-quality nit.
2. **Users cannot evaluate on their own data.** `read_test_data(lang, kind)`
   hardcodes `files(evaluation_datasets)`. There is no parameter to point the
   evaluator at a user's own `ws/` or `analogy/` files.
3. **Conflicting-duplicate and multi-token rows** exist across ~7 similarity
   files (e.g. `de/schm280` ×5, `de/simlex999` ×5, `en/ws353` ×2), plus ~30
   hyphenated/multi-word rows the scorer silently cannot represent.

### The central tension

The package's stated purpose is embeddings for **small, domain-specific**
corpora. A generated *general-language* analogy set (capitals, currencies,
plurals) will not overlap a domain corpus's vocabulary, so it produces all-OOV,
useless scores for the actual use case. It is valuable only for the
general-language benchmark and for **new-language** coverage.

Therefore the AI-generation the question asked about is, for *this* package, the
**lowest-value** item — the highest-value work is fixing the live bug, cleaning
the data, and letting users supply their own. This ADR records that inversion
deliberately.

## Decision

1. **Generate analogies; never generate similarity scores.** Analogy gold is a
   verifiable fact (capital:country, plural, currency); an LLM assigning
   *similarity* scores turns the benchmark from "agreement with humans" into
   "agreement with a language model" — a circular construct. Similarity coverage
   grows only by importing existing human-rated sets (e.g. Multi-SimLex) or by
   running a real human study, never by LLM scoring, rescoring, or
   score-preserving translation.

2. **Fix and freeze before generating.** Preserve today's files as a `legacy`
   suite so historical results stay reproducible, then ship a cleaned
   `curated-v2`. Fix the aggregation bug (mutually-exclusive suite configs so a
   parent set and its subsets are not pooled). Resolve conflicting duplicates by
   *upstream authority → pooled raw ratings → drop*; **do not average two
   unexplained aggregate scores** (that fabricates a number). Record every
   decision as `#`-comment lines in the file itself; the parser already ignores
   them.

3. **A dataset linter is the acceptance gate** for every dataset, bundled or
   generated: column count, score parses, no case-insensitive duplicate or self
   pairs, single-token entries, and no analogy row whose answer collapses onto a
   query word. It runs in the test suite.

4. **Let users bring their own data.** Add an optional `data_dir` to
   `eval_similarity` / `eval_analogies`, and a **coverage report** (fraction of a
   dataset's rows that are in-vocabulary, computable before training). This
   serves the actual use case with zero new data.

5. **Generation is offline and out of the runtime.** Any generation tooling
   lives in `bench/datagen/`; it must never become a runtime dependency (no
   torch/transformers/network in the installed package). Released datasets are
   frozen, checksummed `.txt` files committed to git. A regenerated file is a
   *new version*, never a reproduction — regenerate-on-demand benchmarks are
   forbidden because they break the reproducibility the project is built on.

6. **When generating analogies:** generate base *pairs* and compile quadruples
   deterministically (verify N pairs, not N²); one-to-one relations only;
   **verify against a non-LLM authority** (Wikidata/Wiktionary/UniMorph snapshot)
   with an LLM only as a candidate proposer and cross-checker — LLM agreement is
   a diagnostic, not proof; cap questions per base pair (no full Cartesian
   product); label morphological sets as surface-form (incompatible with the
   lemmatizing preprocessor); and diff against `google.txt`/`msr.txt` for
   contamination.

7. **Provenance is proportionate.** Each generated set carries an in-file header
   recording generator model + version, date, verification source, and pass
   rate — enough to audit and to reproduce the *build* from captured candidates.
   We deliberately do **not** adopt a full research-grade manifest/human-study
   apparatus (raw-response archives, Cohen's κ gates, 12-rater panels): it is
   disproportionate to this package's scale and would not be maintained. Those
   belong to a similarity human-study only if one is ever funded.

## Consequences

- Existing recorded results change once: `curated-v2` and the aggregation fix
  move the similarity numbers (the English micro score in particular). The
  `legacy` suite preserves the old numbers for anyone who needs continuity. This
  goes in the CHANGELOG as a breaking evaluation change.
- The metric becomes correct (no triple-counted pairs) and honest (`oov` stops
  being inflated by rows the scorer never scores).
- Users of domain corpora get a usable evaluation path (`data_dir` + coverage)
  without waiting on any generated data.
- New AI-generated analogy sets are scoped to new-language coverage, where they
  are genuinely useful and verifiable — not sold as improving the small-domain
  use case they cannot serve.

## Roadmap (priority order)

- **P0 — Clean + version + fix the aggregation bug.** No AI. Highest certainty;
  fixes a live metric bug.
- **P1 — `data_dir` + coverage report.** Small code change; biggest leverage for
  the real use case.
- **P2 — Offline analogy pipeline** (`bench/datagen/`), scoped first to EN/DE
  `curated-v2` regeneration and then new languages (ES, FR, then a
  morphology-rich one), gated by the linter and factual verification.
- **P3 — Import human similarity sets** for new languages (license-checked).
- **P4 — Objective domain proxy tasks** (synonym multiple-choice from a glossary,
  category purity) — a new `kind` alongside `ws/`/`analogy/`, designed so
  `data_dir` accommodates it. The correct long-term answer for domain evaluation.

## Alternatives considered

- **LLM-scored similarity data** — rejected (validity trap above).
- **Mass-translating the Google analogies to new languages** — rejected;
  translation shifts morphology, sense, and gender, and can change the relation.
- **Full research-grade provenance/manifest system** (per the GPT review) —
  rejected as disproportionate; principles kept, apparatus dropped.
- **Do nothing / keep the package lightweight** — partially adopted: we add no
  runtime dependency, but the live aggregation bug and the missing `data_dir`
  are real defects worth fixing regardless of the generation question.
