# ADR 0001 — Annex: GPT (codex) review, verbatim

> Independent review by GPT via the `codex` CLI (read-only), 2026-07-22, in
> response to the shared brief. Preserved verbatim as source material for the
> decision recorded in [0001-evaluation-data-strategy.md](../0001-evaluation-data-strategy.md).
> Not all recommendations were adopted — see the ADR's "Alternatives considered".

The highest-value strategy is:

1. Clean and version the existing benchmarks.
2. Generate substantially more analogy data, especially corpus-conditioned and domain-specific sets.
3. Expand similarity data only through existing human-rated resources or new human annotation—not LLM scoring.
4. Treat LLMs as candidate generators and reviewers, never as the source of truth.

I would allocate roughly 70% of new-data effort to analogies, 20% to cleaning/provenance, and 10% initially to preparing properly funded human similarity studies.

## P0 — Establish a trustworthy `curated-v2` suite

Do this before adding data. Preserve the current files as a named `legacy` suite so historical results remain reproducible; do not silently change their contents under the same dataset/version.

Create an offline validator and a manifest for every dataset. The packaged runtime should still consume simple three- or four-column text files; manifests, evidence, generation logs, and heavyweight tooling remain outside the runtime path.

Each manifest should record:

- Dataset name and immutable version.
- Language, task, and analogy relation types.
- Human-rated, imported, deterministic-generated, or LLM-proposed status.
- Upstream source, citation, license, retrieval date, and source checksum.
- Normalization and compatible preprocessing profiles.
- Number of raw, rejected, and released rows, with rejection reasons.
- Split membership and overlap with other bundled sets.
- For generated data: prompts, exact model IDs, parameters, timestamps, raw responses, evidence sources, reviewer decisions, and final checksum.

The validator should reject a release on:

- Wrong column counts or non-finite/out-of-range scores.
- Non-NFC Unicode or unexpected normalization collisions.
- Symmetric duplicate similarity pairs, including reversed order and case variants.
- Exact or inverse duplicate analogy questions.
- Conflicting scores for the same canonical similarity pair.
- Multi-token items after the declared preprocessing pipeline.
- Analogy terms that are not four distinct tokens before and after preprocessing.
- An expected answer that collapses onto an excluded query word.
- Missing source or incompatible licensing information.

The current data already justifies this phase. A raw audit finds conflicting duplicate pairs beyond the four in `de/schm280`: five in German SimLex999, conflicts in German WS353 subsets, English WS353, Rare Words, and MTurk. It also finds 30 hyphen-containing similarity rows and exact analogy duplicates in English/German Google and German `open`/`opposite`.

For conflicts:

- First resolve against the authoritative upstream release.
- If raw individual ratings show that two rows are independent judgments of the same pair, pool the underlying ratings and recompute one gold score.
- Otherwise, do not average two unexplained aggregate numbers. Remove both from `curated-v2` and record the unresolved conflict.
- Keep the untouched upstream form in the archival source area.

Also stop aggregating overlapping parent and subset datasets as though they were independent. For example, WS353 and its similarity/relatedness subsets should belong to mutually exclusive suite configurations. Otherwise the same judgments receive extra weight.

## P1 — Build an offline analogy-generation pipeline

This is the primary source of new evaluation data because analogy answers can be independently verified. An LLM may propose `a → a′` relation pairs, but acceptance must not depend on the LLM’s confidence or consensus alone.

The pipeline should operate on base relation pairs first:

- `athens → greece`
- `baghdad → iraq`

Only after verification should it deterministically form analogy questions. Store base pairs as the canonical data; compile them into the existing four-token format for packaging.

Use a controlled relation catalog. Each relation must have:

- A precise definition and direction.
- Cardinality constraints, preferably one-to-one.
- A verification procedure.
- A stability policy and snapshot date.
- Language-specific lexical and inflection rules.
- Minimum number of independent base pairs.

Good initial relations include capitals, currencies where unambiguous, demonyms, singular/plural, comparative/superlative, and carefully defined domain taxonomies. Avoid vague relations such as “associated with,” many-to-many relations such as country–language, disputed geopolitical facts, and rapidly changing facts unless the dataset is explicitly snapshot-versioned.

Generation procedure:

1. Freeze the input vocabulary, token counts, preprocessing configuration, and corpus checksum.
2. Admit only terms above a declared frequency threshold and represented as exactly one token by the same preprocessing used for training.
3. Ask two independent generators—preferably different model families—to propose relation pairs independently.
4. Canonicalize and merge their candidates.
5. Blindly have each generator classify the other’s candidates as valid, invalid, or uncertain and identify the exact relation.
6. Verify every accepted pair against a non-LLM authority:
   - Versioned knowledge-base snapshot for factual relations.
   - Dictionary or morphological lexicon for inflection.
   - Domain ontology, standard, or pinned API documentation for domain relations.
7. Send disagreements, ambiguities, and source conflicts to a qualified human reviewer.
8. Deterministically construct balanced analogy questions using a recorded seed.

LLM agreement is a diagnostic, not evidence of truth. Report candidate overlap, acceptance agreement, relation-label agreement, and Cohen’s kappa. If only one model family is used, label the dataset `single-generator` and require stronger human review.

Do not form every possible pair of base pairs. That creates thousands of highly dependent questions and makes accuracy look more certain than it is. Cap questions per base pair, balance relation sizes, and retain grouping metadata so confidence intervals can be clustered by base pair.

Morphological analogies must be marked as surface-form datasets. They are incompatible with lemmatizing preprocessing: the repository already notes that lemmatization collapses large parts of Google and MSR into unanswerable questions. Do not interpret such rows as ordinary OOV. The suite should either select a compatible surface-form profile or decline to score that relation.

## P2 — Make corpus-conditioned domain packs a first-class product

A fixed general-language benchmark and a corpus-conditioned benchmark answer different questions and should always be reported separately:

- Fixed suite: comparable across users and releases.
- Corpus-conditioned suite: meaningful for this particular small corpus.

The corpus-conditioned builder may use only vocabulary membership, token frequency, language, and declared domain to select candidates. It must not choose test items because `hyperhyper` gets them right or wrong.

For each domain pack:

- Store the corpus hash, vocabulary hash, preprocessing profile, and frequency threshold.
- Require all four terms to survive the exact corpus preprocessing.
- Split at the base-pair/entity level, not after expanding to quadruples.
- Provide `dev` for relation and hyperparameter decisions and a sealed `test` split.
- Require at least roughly 12 independent base pairs per relation; otherwise report that the corpus cannot support that relation rather than manufacturing a tiny score.
- Report accuracy alongside coverage, item count, relation-macro accuracy, and clustered confidence intervals.

Pilot domains should be selected by availability of authoritative structured sources, not by how easy they are for an LLM to discuss. Good pilots are versioned biomedical/taxonomic ontologies and pinned software/API vocabularies. Defer legal and policy packs until jurisdiction, effective date, multi-word terminology, and expert review are handled properly.

Multi-word domain terms should not be replaced with their first token or casually joined with underscores. Support them only in a separately labelled phrase-aware pack when the training preprocessor produces the identical atomic token.

## P3 — Add languages deliberately

Do not mass-translate Google analogies. Translation changes morphology, sense, grammatical gender, lexical frequency, and sometimes the relation itself.

Recommended sequence:

1. Finish English and German `curated-v2` and corpus-conditioned pilots.
2. Add Spanish and French as high-resource pilots with native reviewers and strong lexical/knowledge-base coverage.
3. Add one morphology-rich language, such as Polish, to test surface-form relation handling.
4. Expand further according to user demand and availability of native/domain reviewers.

For each language:

- Generate natively using a language-specific relation inventory.
- Verify lexical forms with native dictionaries or morphological analyzers.
- Have native speakers audit all relation definitions and a stratified item sample.
- Publish analogy-only language packs when that is all that can be supported honestly; do not create LLM-scored similarity files merely to make every language directory symmetrical.
- Label translated-and-verified data separately from natively generated data.

## P4 — Expand similarity only with humans

An LLM must never supply, revise, adjudicate, or fill missing similarity gold scores. LLM-generated scores would change the construct from “agreement with humans” to “agreement with a language model.”

There are two valid routes:

1. Import additional established human-rated datasets, after checking licensing, provenance, overlap, token compatibility, and native rather than machine-translated rating methodology.
2. Run new human annotation studies.

For a new domain similarity set:

- Select candidates from the corpus vocabulary using frequency strata, lexical resources, random pairs, and several unrelated baseline models. Do not select only examples where `hyperhyper` behaves interestingly.
- Define clearly whether the task is similarity or broader relatedness.
- Use at least 12 independent raters per pair; use domain experts where specialist senses matter.
- Randomize order and pair direction.
- Include hidden repeats, anchors, and attention checks.
- Predeclare annotator-exclusion and adjudication rules.
- Release anonymized raw ratings, aggregate score, variance or confidence interval, and annotation instructions.
- Measure split-half reliability and Krippendorff’s alpha or ICC.
- Include blinded anchor pairs from an established human set. Aim for anchor Spearman correlation around 0.8 or better, while investigating genuine domain-sense differences rather than forcing agreement.
- Make dev/test splits entity-disjoint where feasible.

LLMs may suggest candidate terms, flag spelling problems, or identify potentially ambiguous senses. Those flags should be reviewed by humans and must never determine the numeric gold score.

## P5 — Validate generated releases before bundling

Use a candidate pool several times larger than the final release and apply predeclared gates.

For analogies:

- Every fact passes deterministic or source-backed verification.
- Independent verifier agreement is at least substantial, preferably κ ≥ 0.8.
- Human-review a stratified 10% or at least 100 items per language/domain; review all items for small or high-risk packs.
- Require estimated factual precision of at least 98%, with the 95% confidence lower bound above 95%.
- Reject an entire relation if errors are systematic, even when overall precision passes.
- Test the generator on held-out existing analogy relations, then remove exact public-benchmark copies from the new test set.

For difficulty calibration, use a frozen panel containing random/unigram baselines, several PPMI configurations, and at least one unrelated embedding family—offline only. Publish easy/core/challenge strata based on median panel performance. Avoid a set where almost every system scores below 5% or above 95%, but do not delete verified hard items merely because `hyperhyper` misses them.

For new human similarity sets:

- Check inter-rater reliability and anchor correlation.
- Compare the ranking of a fixed panel of embeddings on the new and established human sets.
- Investigate large reversals by language, frequency, ambiguity, and domain rather than tuning the data until the ranking looks familiar.

A model-panel correlation is a quality diagnostic, not proof. Factual correctness or human judgment remains the gold authority.

## Reproducibility and contamination policy

Keep generation in a separate locked development environment or companion repository. No LLM client, transformer, knowledge-base client, or morphology package should become a runtime dependency. Runtime evaluation should read committed, checksummed artifacts without network access.

Because hosted LLM calls are not bit-reproducible, preserve their raw responses. The deterministic build should regenerate the released `.txt` files from captured candidate records, evidence, reviewer decisions, and a fixed seed. A future API rerun is a new dataset version, not a reproduction of the old one.

Protect against evaluation leakage by:

- Publishing immutable `dev` and `test` splits.
- Splitting analogy base pairs and entities before quadruple construction.
- Deduplicating against bundled and major public benchmarks.
- Scanning a frozen calibration corpus for exact benchmark templates and reporting matches.
- Never using test performance to select items.
- Reporting corpus-conditioned results only against the exact pack and corpus snapshot that generated them.

## Explicit non-goals

Do not:

- Ask an LLM for similarity scores.
- Treat two agreeing LLMs as independent factual proof.
- machine-translate human scores without native rerating.
- silently repair conflicting scores by averaging aggregates.
- ship multi-token entries that the declared preprocessor cannot represent.
- expand a few base pairs into enormous Cartesian products.
- mix overlapping datasets into one aggregate.
- use changing facts without source versions and dates.
- select questions to improve `hyperhyper`’s reported score.
- generate data during installation, training, or evaluation.
- publish generated data without prompts, evidence, labels, licenses, and checksums.

The first concrete release should therefore be `legacy` plus a cleaned `curated-v2`, followed by an offline English/German corpus-conditioned analogy builder. Only after that pipeline meets its factual-precision and human-audit gates should it be used for Spanish/French and domain packs. Human-rated domain similarity should be a separate, slower project rather than an LLM shortcut.